# coding=utf-8
# Copyright 2023 The LAION-AI Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch CLAP model."""
import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat, reduce, pack, unpack

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from configuration_clamp import (
    ClampAudioConfig,
    ClampConfig,
    ClampTextConfig,
    ClampMotionConfig,
)
from transformers.models.clap import ClapPreTrainedModel, ClapModel
from transformers.models.clap.modeling_clap import (
    ClapProjectionLayer,
    ClapOutput,
    ClapTextModel,
    ClapAudioModel,
    ClapAudioModelWithProjection,
    ClapTextModelWithProjection,
    CLAP_INPUTS_DOCSTRING,
    CLAP_TEXT_INPUTS_DOCSTRING,
    CLAP_AUDIO_INPUTS_DOCSTRING,
)


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html#CLIP-loss-function
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(
        self, n_in, n_state, dilation=1, activation="silu", norm=None, dropout=None
    ):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-6, affine=True
            )
            self.norm2 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-6, affine=True
            )
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(
            n_state,
            n_in,
            1,
            1,
            0,
        )

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = [
            ResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate**depth,
                activation=activation,
                norm=norm,
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# from vector_quantize_pytorch import ResidualVQ

# Borrow from vector_quantize_pytorch


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(logits, temperature=1.0, stochastic=False, dim=-1, training=True):
    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)

    return ind


def batched_sample_vectors(samples, num):
    def sample_vectors(samples, num):
        num_samples, device = samples.shape[0], samples.device
        if num_samples >= num:
            indices = torch.randperm(num_samples, device=device)[:num]
        else:
            indices = torch.randint(0, num_samples, (num,), device=device)

        return samples[indices]

    return sample_vectors(samples, num)


def batched_bincount(x, *, minlength):
    dtype, device = x.dtype, x.device
    target = torch.zeros(minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
    samples,
    num_clusters,
    num_iters=10,
    sample_fn=batched_sample_vectors,
):
    dim, dtype, device = (
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        dists = -torch.cdist(samples, means, p=2)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    return means, bins


class QuantizeEMAReset(nn.Module):
    def __init__(self, config: ClampMotionConfig, mu=0.99):
        super(QuantizeEMAReset, self).__init__()
        self.nb_code = config.codebook_size
        self.code_dim = config.codebook_dim

        self.mu = mu  ##TO_DO
        requires_projection = False
        self.project_in = (
            nn.Linear(config.hidden_size, self.code_dim)
            if requires_projection
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(self.code_dim, config.hidden_size)
            if requires_projection
            else nn.Identity()
        )
        self.codebook = nn.Embedding(self.nb_code, self.code_dim)

    def quantize(self, x, sample_codebook_temp=0.0):
        # N X C -> C X N
        k_w = self.codebook.weight.t()
        # x: NT X C
        # NT X N
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )  # (N * L, b)

        # code_idx = torch.argmin(distance, dim=-1)

        code_idx = gumbel_sample(
            -distance,
            dim=-1,
            temperature=sample_codebook_temp,
            stochastic=True,
            training=self.training,
        )

        return code_idx

    def dequantize(self, code_idx):
        x = self.codebook(code_idx)
        # F.embedding(code_idx, self.codebook)

        return x

    @torch.no_grad()
    def encode(self, x, mask=None, temperature=0.0):
        shape = x.shape

        need_transpose = True if (shape[-1] != self.code_dim) else False

        if need_transpose:
            x = rearrange(x, "n c t -> (n t) c")
            N, width, T = shape
        else:
            x = rearrange(x, "n t c -> (n t) c")
            N, T, width = shape

        x = self.project_in(x)

        # if self.training and not self.init:
        #     self.init_codebook(x)

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)

        x_d = self.project_out(x_d)
        if need_transpose:
            x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        else:
            x_d = x_d.view(N, T, -1).contiguous()
        return x_d

    def forward(self, x, return_idx=False, temperature=0.0):
        shape = x.shape

        need_transpose = True if (shape[-1] != self.code_dim) else False

        if need_transpose:
            x = rearrange(x, "n c t -> (n t) c")
            N, width, T = shape
        else:
            x = rearrange(x, "n t c -> (n t) c")
            N, T, width = shape

        x = self.project_in(x)

        # if self.training and not self.init:
        #     self.init_codebook(x)

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)  ## N T C
        print(x_d.shape)

        # if self.training:
        #     perplexity = self.update_codebook(x, code_idx)
        # else:
        #     perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        x_d = self.project_out(x_d)

        if need_transpose:
            x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        else:
            x_d = x_d.view(N, T, -1).contiguous()

        # Postprocess
        code_idx = code_idx.view(N, T).contiguous()

        # print(code_idx[0])
        if return_idx:
            return x_d, code_idx, commit_loss
        return x_d, commit_loss


CLAP_MOTION_START_DOCSTRING = r"""
    This model inherits from [`ClapModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ClampConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CLAP_MOTION_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class ClampMotionProjectionLayer(nn.Module):
    def __init__(self, config: Union[ClampMotionConfig]):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        input_dim = config.codebook_dim
        projection_dim = config.projection_dim

        self.project_in = (
            nn.Linear(input_dim, hidden_size)
            if input_dim != hidden_size
            else nn.Identity()
        )

        self.linear1 = nn.Linear(hidden_size, projection_dim)
        self.activation = ACT2FN[config.projection_hidden_act]
        self.linear2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, hidden_states):

        hidden_states = self.project_in(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MotionEncoder(nn.Module):
    def __init__(
        self,
        config: ClampMotionConfig,
    ):
        super().__init__()

        blocks = []
        filter_t, pad_t = config.conv_stride * 2, config.conv_stride // 2
        blocks.append(nn.Conv1d(config.motion_dim, config.hidden_size, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(int(np.log2(config.down_sampling_ratio))):
            input_dim = config.hidden_size
            block = nn.Sequential(
                nn.Conv1d(
                    input_dim, config.hidden_size, filter_t, config.conv_stride, pad_t
                ),
                Resnet1D(
                    config.hidden_size,
                    config.num_hidden_layers,
                    config.dilation_growth_rate,
                    activation=config.hidden_act,
                ),
            )
            blocks.append(block)

        if config.codebook_dim != config.hidden_size:
            blocks.append(nn.Conv1d(config.hidden_size, config.codebook_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x, mask=None):
        return self.model(x)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPTextModelOutput with CLIP->Clap
class ClampMotionModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    motion_embeds: Optional[torch.FloatTensor] = None
    motion_quantized: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    # last_hidden_state: torch.FloatTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->Clap, vision->audio, Vision->Audio, image->audio
class ClampOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for audio-text similarity.
        logits_per_audio:(`torch.FloatTensor` of shape `(audio_batch_size, text_batch_size)`):
            The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the audio-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, audio_batch_size)`):
            The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-audio
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`ClapTextModel`].
        audio_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The audio embeddings obtained by applying the projection layer to the pooled output of [`ClapAudioModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ClapTextModel`].
        audio_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ClapAudioModel`].
    """

    loss: Optional[torch.FloatTensor] = None

    logits_per_text_vs_audio: torch.FloatTensor = None
    logits_per_audio_vs_text: torch.FloatTensor = None

    logits_per_text_vs_motion: torch.FloatTensor = None
    logits_per_motion_vs_text: torch.FloatTensor = None

    logits_per_audio_vs_motion: torch.FloatTensor = None
    logits_per_motion_vs_audio: torch.FloatTensor = None

    text_embeds: torch.FloatTensor = None
    audio_embeds: torch.FloatTensor = None
    motion_embeds: torch.FloatTensor = None

    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None
    motion_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            (
                self[k]
                if k not in ["text_model_output", "audio_model_output"]
                else getattr(self, k).to_tuple()
            )
            for k in self.keys()
        )


class ClampMotionPooler(nn.Module):
    def __init__(self, config: ClampMotionConfig):
        super().__init__()
        self.dim = config.codebook_dim
        self.pool_type = config.pool_type
        # self.dense = nn.Linear(config.codebook_dim, config.codebook_dim)

        self.activation = nn.Tanh()

    def forward(
        self, quantized_motion: torch.Tensor, motion_mask: torch.IntTensor = None
    ) -> torch.Tensor:

        # if quantized_motion.shape[-1] != self.dim:
        #     quantized_motion = quantized_motion.permute(0, 2, 1).contiguous()

        if self.pool_type == "mean":

            pooled_output = torch.mean(quantized_motion, dim=1)
        elif self.pool_type == "max":
            pooled_output = torch.max(quantized_motion, dim=1)

        pooled_output = self.activation(pooled_output)
        return pooled_output


class ClampMotionModel(ClapPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    config_class = ClampMotionConfig

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->ClapText
    def __init__(self, config: ClampMotionConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.encoder = MotionEncoder(config)

        self.quantizer = QuantizeEMAReset(config).eval()

        self.pooler = ClampMotionPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def codebook(self):
        return self.quantizer.codebook

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).contiguous()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1).contiguous()
        return x

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        motion_mask: Optional[torch.IntTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], ClampMotionModelOutput]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_features is not None:
            input_shape = input_features.size()

        else:
            raise ValueError("You have to specify input_features")

        batch_size, seq_length, motion_dim = input_shape
        device = input_features.device

        input_features = self.preprocess(input_features).float()
        encoded_features = self.encoder(input_features)

        if motion_mask is not None:

            downsampled_motion_mask = torch.nn.functional.max_pool1d(
                motion_mask.float(),
                1,
                stride=self.config.down_sampling_ratio,
            )
            encoded_features = encoded_features * downsampled_motion_mask[:, None, :]
        quantized_features = self.quantizer.encode(
            encoded_features,
            downsampled_motion_mask if motion_mask is not None else None,
            temperature=self.config.codebook_sampling_temperature,
        )
        if motion_mask is not None:

            quantized_features = (
                quantized_features * downsampled_motion_mask[:, None, :]
            )
        encoded_features = self.postprocess(encoded_features)
        quantized_features = self.postprocess(quantized_features)

        pooled_output = (
            self.pooler(quantized_features) if self.pooler is not None else None
        )

        if not return_dict:
            return (encoded_features, pooled_output, quantized_features)

        return ClampMotionModelOutput(
            motion_embeds=encoded_features,
            motion_quantized=quantized_features,
            pooler_output=pooled_output,
        )


@add_start_docstrings(
    """
    CLAP Motion Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    CLAP_MOTION_START_DOCSTRING,
)
class ClampMotionModelWithProjection(ClapPreTrainedModel):
    config_class = ClampMotionConfig

    def __init__(self, config: ClampMotionConfig):
        super().__init__(config)
        self.motion_model = ClampMotionModel(config)
        self.motion_projection = ClapProjectionLayer(config)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CLAP_MOTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=ClampMotionModelOutput, config_class=ClampMotionConfig
    )
    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        motion_mask: Optional[torch.IntTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ClampMotionModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapTextModelWithProjection

        >>> model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        motion_outputs = self.motion_model(
            input_features=input_features,
            motion_mask=motion_mask,
            return_dict=return_dict,
        )

        pooled_output = (
            motion_outputs[1] if not return_dict else motion_outputs.pooler_output
        )

        motion_embeds = self.motion_projection(pooled_output)

        if not return_dict:
            outputs = (motion_embeds, motion_outputs[1], motion_outputs[2])
            return tuple(output for output in outputs if output is not None)

        return ClampMotionModelOutput(
            motion_embeds=motion_embeds,
            motion_quantized=motion_outputs.motion_quantized,
            pooler_output=pooled_output,
        )


@add_start_docstrings(CLAP_MOTION_START_DOCSTRING)
class ClampModel(ClapPreTrainedModel):
    config_class = ClampConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: ClampConfig):
        super().__init__(config)

        if not isinstance(config.motion_config, ClampMotionConfig):
            raise ValueError(
                "config.motion_config is expected to be of type ClampMotionConfig but is of type"
                f" {type(config.motion_config)}."
            )

        if not isinstance(config.text_config, ClampTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type ClampTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.audio_config, ClampAudioConfig):
            raise ValueError(
                "config.audio_config is expected to be of type ClampAudioConfig but is of type"
                f" {type(config.audio_config)}."
            )

        text_config = config.text_config
        audio_config = config.audio_config
        motion_config = config.motion_config

        self.logit_scale_a = nn.Parameter(
            torch.ones([]) * np.log(config.logit_scale_init_value)
        )
        self.logit_scale_t = nn.Parameter(
            torch.ones([]) * np.log(config.logit_scale_init_value)
        )
        self.logit_scale_m = nn.Parameter(
            torch.ones([]) * np.log(config.logit_scale_init_value)
        )

        self.projection_dim = config.projection_dim

        self.text_model = ClapTextModel(text_config)
        self.text_projection = ClapProjectionLayer(text_config)

        self.audio_model = ClapAudioModel(audio_config)
        self.audio_projection = ClapProjectionLayer(audio_config)

        self.motion_model = ClampMotionModel(motion_config)
        self.motion_projection = ClampMotionProjectionLayer(motion_config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CLAP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`ClapTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapModel

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLAP model's config for some fields (if specified) instead of those of audio & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = (
            text_outputs[1] if return_dict is not None else text_outputs.pooler_output
        )
        text_features = self.text_projection(pooled_output)
        text_features = F.normalize(text_features, dim=-1)

        return text_features

    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    def get_audio_features(
        self,
        input_features: Optional[torch.Tensor] = None,
        is_longer: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            audio_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The audio embeddings obtained by
            applying the projection layer to the pooled output of [`ClapAudioModel`].

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor, ClapModel
        >>> import torch

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        >>> random_audio = torch.rand((16_000))
        >>> inputs = feature_extractor(random_audio, return_tensors="pt")
        >>> audio_features = model.get_audio_features(**inputs)
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            return_dict=return_dict,
        )

        pooled_output = (
            audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        )

        audio_features = self.audio_projection(pooled_output)
        audio_features = F.normalize(audio_features, dim=-1)

        return audio_features

    @add_start_docstrings_to_model_forward(CLAP_MOTION_INPUTS_DOCSTRING)
    def get_motion_features(
        self,
        input_features: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`ClapTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapModel

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLAP model's config for some fields (if specified) instead of those of audio & text components.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        motion_outputs = self.motion_model(
            input_features=input_features,
            return_dict=return_dict,
        )

        pooled_output = (
            motion_outputs[1] if not return_dict else motion_outputs.pooler_output
        )

        motion_features = self.motion_projection(pooled_output)
        motion_features = F.normalize(motion_features, dim=-1)

        return motion_features

    @add_start_docstrings_to_model_forward(CLAP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapOutput, config_class=ClampConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        input_motion_features: Optional[torch.FloatTensor] = None,
        motion_mask: Optional[torch.IntTensor] = None,
        is_longer: Optional[torch.BoolTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ClapOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, ClapModel

        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

        >>> input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

        >>> inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

        >>> outputs = model(**inputs)
        >>> logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
        >>> probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLAP model's config for some fields (if specified) instead of those of audio & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        logits_per_text_vs_audio = None
        logits_per_audio_vs_text = None
        logits_per_text_vs_motion = None
        logits_per_motion_vs_text = None
        logits_per_audio_vs_motion = None
        logits_per_motion_vs_audio = None
        text_embeds = None
        audio_embeds = None
        motion_embeds = None
        text_outputs = None
        audio_outputs = None
        motion_outputs = None

        mode = []

        if input_ids is not None:
            mode.append("text")
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            text_embeds = (
                text_outputs[1] if not return_dict else text_outputs.pooler_output
            )
            # normalized features
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        if input_features is not None:
            mode.append("audio")
            audio_outputs = self.audio_model(
                input_features=input_features,
                is_longer=is_longer,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            audio_embeds = (
                audio_outputs[1] if not return_dict else audio_outputs.pooler_output
            )
            # normalized features
            audio_embeds = self.audio_projection(audio_embeds)
            audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)

        if input_motion_features is not None:
            mode.append("motion")
            motion_outputs = self.motion_model(
                input_features=input_motion_features,
                motion_mask=motion_mask,
                return_dict=return_dict,
            )
            motion_embeds = (
                motion_outputs[1] if not return_dict else motion_outputs.pooler_output
            )
            # normalized features
            motion_embeds = self.motion_projection(motion_embeds)
            motion_embeds = motion_embeds / motion_embeds.norm(
                p=2, dim=-1, keepdim=True
            )

        assert len(mode) > 1, "You need to input atleast 2 modalities"

        # cosine similarity as logits
        logit_scale_text = self.logit_scale_t.exp()
        logit_scale_audio = self.logit_scale_a.exp()
        logit_scale_motion = self.logit_scale_m.exp()

        if input_ids is not None and input_features is not None:
            logits_per_text_vs_audio = (
                torch.matmul(text_embeds, audio_embeds.t()) * logit_scale_text
            )
            logits_per_audio_vs_text = (
                torch.matmul(audio_embeds, text_embeds.t()) * logit_scale_audio
            )

        if input_ids is not None and input_motion_features is not None:
            logits_per_text_vs_motion = (
                torch.matmul(text_embeds, motion_embeds.t()) * logit_scale_text
            )
            logits_per_motion_vs_text = (
                torch.matmul(motion_embeds, text_embeds.t()) * logit_scale_motion
            )

        if input_motion_features is not None and input_features is not None:

            logits_per_audio_vs_motion = (
                torch.matmul(audio_embeds, motion_embeds.t()) * logit_scale_audio
            )
            logits_per_motion_vs_audio = (
                torch.matmul(motion_embeds, audio_embeds.t()) * logit_scale_motion
            )

        loss = None

        if return_loss:

            loss_list = []

            if input_ids is not None and input_features is not None:

                txt_aud_loss = contrastive_loss(logits_per_text_vs_audio)
                aud_txt_loss = contrastive_loss(logits_per_audio_vs_text.t())

                loss_list.append(txt_aud_loss)
                loss_list.append(aud_txt_loss)

            if input_ids is not None and input_motion_features is not None:

                txt_mot_loss = contrastive_loss(logits_per_text_vs_motion)
                mod_txt_loss = contrastive_loss(logits_per_motion_vs_text.t())

                loss_list.append(txt_mot_loss)
                loss_list.append(mod_txt_loss)

            if input_motion_features is not None and input_features is not None:

                aud_mot_loss = contrastive_loss(logits_per_audio_vs_motion)
                mot_aud_loss = contrastive_loss(logits_per_motion_vs_audio.t())

                loss_list.append(aud_mot_loss)
                loss_list.append(mot_aud_loss)

            loss = sum(loss_list) / len(loss_list)

        if not return_dict:
            output = (
                logits_per_text_vs_audio,
                logits_per_audio_vs_text,
                logits_per_text_vs_motion,
                logits_per_motion_vs_text,
                logits_per_audio_vs_motion,
                logits_per_motion_vs_audio,
                text_embeds,
                audio_embeds,
                motion_embeds,
                text_outputs,
                audio_outputs,
                motion_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return ClampOutput(
            loss=loss,
            logits_per_text_vs_audio=logits_per_text_vs_audio,
            logits_per_audio_vs_text=logits_per_audio_vs_text,
            logits_per_text_vs_motion=logits_per_text_vs_motion,
            logits_per_motion_vs_text=logits_per_motion_vs_text,
            logits_per_audio_vs_motion=logits_per_audio_vs_motion,
            logits_per_motion_vs_audio=logits_per_motion_vs_audio,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            motion_embeds=motion_embeds,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
            motion_model_output=motion_outputs,
        )
