# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for CLAP."""


import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import os
from transformers.audio_utils import fram_wave, get_mel_filter_banks, power_to_db, stft
from transformers.feature_extraction_sequence_utils import (
    SequenceFeatureExtractor,
    FeatureExtractionMixin,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging

# from transformers import FeatureExtractionMixin

logger = logging.get_logger(__name__)


class ClampFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CLAP feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the *Short Time
    Fourier Transform* (STFT) which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, defaults to 64):
            The feature dimension of the extracted Mel spectrograms. This corresponds to the number of mel filters
            (`n_mels`).
        sampling_rate (`int`, defaults to 48_000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz). This only serves
            to warn users if the audio fed to the feature extractor does not have the same sampling rate.
        hop_length (`int`, defaults to 480):
            Length of the overlaping windows for the STFT used to obtain the Mel Spectrogram. The audio will be split
            in smaller `frames` with a step of `hop_length` between each frame.
        max_length_s (`int`, defaults to 10):
            The maximum input lenght of the model in seconds. This is used to pad the audio.
        fft_window_size (`int`, defaults to 1024):
            Size of the window (in samples) on which the Fourier transform is applied. This controls the frequency
            resolution of the spectrogram. 400 means that the fourrier transform is computed on windows of 400 samples.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the attention masks coresponding to the input.
        frequency_min (`float`, *optional*, default to 0):
            The lowest frequency of interest. The STFT will not be computed for values below this.
        frequency_max (`float`, *optional*, default to 14_000):
            The highest frequency of interest. The STFT will not be computed for values above this.
        top_db (`float`, *optional*):
            The highest decibel value used to convert the mel spectrogram to the log scale. For more details see the
            `audio_utils.power_to_db` function
        truncation (`str`, *optional*, default to `"fusions"`):
            Truncation pattern for long audio inputs. Two patterns are available:
                - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and a
                  downsampled version of the entire mel spectrogram.
            If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a copy
            of the original mel obtained from the padded audio.
                - `rand_trunc` will select a random crop of the mel spectrogram.
        padding (`str`, *optional*, defaults to `"repeatpad"`):
               Padding pattern for shorter audio inputs. Three patterns were originally implemented:
                - `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
                - `repeat`: the audio is repeated and then cut to fit the `max_length`
                - `pad`: the audio is padded.
    """

    model_input_names = [
        "input_features",
        "is_longer",
        "input_motion_features",
        "motion_mask",
    ]

    def __init__(
        self,
        feature_size=64,
        sampling_rate=48_000,
        hop_length=480,
        max_length_s=10,
        fft_window_size=1024,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        frequency_min: float = 0,
        frequency_max: float = 14_000,
        top_db: int = None,
        truncation: str = "fusion",
        padding: str = "repeatpad",
        motion_padding: str = "longest",
        motion_max_length_s: int = 10,
        fps: int = 30,
        motion_type: str = "full",
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.top_db = top_db
        self.truncation = truncation
        self.padding = padding
        self.fft_window_size = fft_window_size
        self.nb_frequency_bins = (fft_window_size >> 1) + 1
        self.hop_length = hop_length
        self.max_length_s = max_length_s
        self.nb_max_samples = max_length_s * sampling_rate
        self.sampling_rate = sampling_rate
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max
        self.mel_filters = get_mel_filter_banks(
            nb_frequency_bins=self.nb_frequency_bins,
            nb_mel_filters=feature_size,
            frequency_min=frequency_min,
            frequency_max=frequency_max,
            sample_rate=sampling_rate,
            norm=None,
            mel_scale="htk",
        )
        self.mel_filters_slaney = get_mel_filter_banks(
            nb_frequency_bins=self.nb_frequency_bins,
            nb_mel_filters=feature_size,
            frequency_min=frequency_min,
            frequency_max=frequency_max,
            sample_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.fps = fps
        self.motion_type = motion_type
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.body_mean, self.hand_mean, self.full_mean = self.hmldata_process(self.mean)
        self.body_std, self.hand_std, self.full_std = self.hmldata_process(self.std)
        self.motion_padding = motion_padding
        self.motion_max_length = motion_max_length_s * fps
        self.use_rotation = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance, excpet for the
            mel filter banks, which do not need to be saved or printed as they are too long.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        if "mel_filters" in output:
            del output["mel_filters"]
        if "mel_filters_slaney" in output:
            del output["mel_filters_slaney"]
        return output

    def _np_extract_fbank_features(
        self, waveform: np.array, mel_filters: Optional[np.array] = None
    ) -> np.ndarray:
        """
        Compute the log-Mel spectrogram of the provided `waveform` using the `hanning` window. In CLAP, two different
        filter banks are used depending on the truncation pattern:
            - `self.mel_filters`: they correspond to the defaults parameters of `torchaduio` which can be obtained from
              calling `torchaudio.transforms.MelSpectrogram().mel_scale.fb`. These filters are used when `truncation`
              is set to `"fusion"`.
            - `self.mel_filteres_slaney` : they correspond to the defaults parameters of `torchlibrosa` which used
              `librosa.filters.mel` when computing the mel spectrogram. These filters were only used in the original
              implementation when the truncation mode is not `"fusion"`.
        """
        window = np.hanning(self.fft_window_size + 1)[:-1]
        frames = fram_wave(waveform, self.hop_length, self.fft_window_size)
        spectrogram = stft(frames, window, fft_window_size=self.fft_window_size)

        magnitudes = np.abs(spectrogram) ** 2
        mel_spectrogram = np.matmul(mel_filters.T, magnitudes)
        log_mel_spectrogram = power_to_db(mel_spectrogram).T
        log_mel_spectrogram = np.asarray(log_mel_spectrogram, np.float32)
        return log_mel_spectrogram

    def _random_mel_fusion(self, mel, total_frames, chunk_frames):
        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
        if len(ranges[1]) == 0:
            # if the audio is too short, we just use the first chunk
            ranges[1] = [0]
        if len(ranges[2]) == 0:
            # if the audio is too short, we just use the first chunk
            ranges[2] = [0]
        # randomly choose index for each part
        idx_front = np.random.choice(ranges[0])
        idx_middle = np.random.choice(ranges[1])
        idx_back = np.random.choice(ranges[2])

        mel_chunk_front = mel[idx_front : idx_front + chunk_frames, :]
        mel_chunk_middle = mel[idx_middle : idx_middle + chunk_frames, :]
        mel_chunk_back = mel[idx_back : idx_back + chunk_frames, :]

        mel = torch.tensor(mel[None, None, :])
        mel_shrink = torch.nn.functional.interpolate(
            mel,
            size=[chunk_frames, 64],
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        mel_shrink = mel_shrink[0][0].numpy()
        mel_fusion = np.stack(
            [mel_chunk_front, mel_chunk_middle, mel_chunk_back, mel_shrink], axis=0
        )
        return mel_fusion

    def _get_input_mel(
        self,
        waveform: np.array,
        max_length,
        truncation,
        padding,
        subset_index=None,
    ) -> np.array:
        """
        Extracts the mel spectrogram and prepares it for the mode based on the `truncation` and `padding` arguments.
        Four different path are possible:
            - `truncation="fusion"` and the length of the waveform is greater than the max length: the mel spectrogram
              will be computed on the entire audio. 3 random crops and a dowsampled version of the full mel spectrogram
              are then stacked together. They will later be used for `feature_fusion`.
            - `truncation="rand_trunc"` and the length of the waveform is smaller than the max length: the audio is
              padded based on `padding`.
            - `truncation="fusion"` and the length of the waveform is smaller than the max length: the audio is padded
              based on `padding`, and is repeated `4` times.
            - `truncation="rand_trunc"` and the length of the waveform is greater than the max length: the mel
              spectrogram will be computed on a random crop of the waveform.

        """
        if waveform.shape[0] > max_length:
            if truncation == "rand_trunc":
                longer = True
                # random crop to max_length (for compatibility) -> this should be handled by self.pad
                overflow = len(waveform) - max_length
                idx = (
                    subset_index
                    if subset_index is not None
                    else np.random.randint(0, overflow + 1)
                )
                waveform = waveform[idx : idx + max_length]
                input_mel = self._np_extract_fbank_features(
                    waveform, self.mel_filters_slaney
                )[None, :]

            elif truncation == "fusion":
                mel = self._np_extract_fbank_features(waveform, self.mel_filters)
                chunk_frames = (
                    max_length // self.hop_length + 1
                )  # the +1 related to how the spectrogram is computed
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is larger than max_length but smaller than max_length+hop_length.
                    # In this case, we just use the whole audio.
                    input_mel = np.stack([mel, mel, mel, mel], axis=0)
                    longer = False
                else:
                    input_mel = self._random_mel_fusion(mel, total_frames, chunk_frames)
                    longer = True
            else:
                raise NotImplementedError(
                    f"data_truncating {truncation} not implemented"
                )

        else:
            longer = False
            # only use repeat as a new possible value for padding. you repeat the audio before applying the usual max_length padding
            if waveform.shape[0] < max_length:
                if padding == "repeat":
                    n_repeat = int(max_length / len(waveform))
                    waveform = np.stack(np.tile(waveform, n_repeat + 1))[:max_length]
                if padding == "repeatpad":
                    n_repeat = int(max_length / len(waveform))
                    waveform = np.stack(np.tile(waveform, n_repeat))
                waveform = np.pad(
                    waveform,
                    (0, max_length - waveform.shape[0]),
                    mode="constant",
                    constant_values=0,
                )

            if truncation == "fusion":
                input_mel = self._np_extract_fbank_features(waveform, self.mel_filters)
                input_mel = np.stack(
                    [input_mel, input_mel, input_mel, input_mel], axis=0
                )
            else:
                input_mel = self._np_extract_fbank_features(
                    waveform, self.mel_filters_slaney
                )[None, :]

        return input_mel, longer

    def inv_transform(self, data: torch.Tensor) -> torch.Tensor:

        assert (
            data.shape[-1] == self.body_mean.shape[-1]
            or data.shape[-1] == self.hand_mean.shape[-1]
            or data.shape[-1] == self.full_mean.shape[-1]
        ), f"shape mismatch between input data {data.shape} and means {self.body_mean.shape} {self.hand_mean.shape} {self.full_mean.shape}"

        if data.shape[-1] == self.body_mean.shape[-1]:
            return data * (
                torch.Tensor(self.body_std).to(data.device) - 1e-8
            ) + torch.Tensor(self.body_mean).to(data.device)

        elif data.shape[-1] == self.hand_mean.shape[-1]:
            return data * (
                torch.Tensor(self.hand_std).to(data.device) - 1e-8
            ) + torch.Tensor(self.hand_mean).to(data.device)

        elif data.shape[-1] == self.full_mean.shape[-1]:
            return data * (
                torch.Tensor(self.full_std).to(data.device) - 1e-8
            ) + torch.Tensor(self.full_mean).to(data.device)

    def transform(self, data: np.array) -> np.array:
        assert (
            data.shape[-1] == self.body_mean.shape[-1]
            or data.shape[-1] == self.hand_mean.shape[-1]
            or data.shape[-1] == self.full_mean.shape[-1]
        ), f"shape mismatch between input data {data.shape} and means {self.body_mean.shape} {self.hand_mean.shape} {self.full_mean.shape}"

        if data.shape[-1] == self.body_mean.shape[-1]:
            return (data - self.body_mean) / (self.body_std + 1e-8)

        elif data.shape[-1] == self.hand_mean.shape[-1]:
            return (data - self.hand_mean) / (self.hand_std + 1e-8)

        elif data.shape[-1] == self.full_mean.shape[-1]:
            return (data - self.full_mean) / (self.full_std + 1e-8)

    def hmldata_process(
        self,
        hml_data: np.array,
        joint_num=52,
        body_joints=22,
        hand_joints=30,
    ):
        root_params = hml_data[..., :4]
        local_pos = hml_data[..., 4 : 4 + (joint_num - 1) * 3]
        local_rots = hml_data[
            ..., 4 + (joint_num - 1) * 3 : 4 + (joint_num - 1) * 3 + (joint_num - 1) * 6
        ]
        local_vels = hml_data[
            ...,
            4
            + (joint_num - 1) * 3
            + (joint_num - 1) * 6 : 4
            + (joint_num - 1) * 3
            + (joint_num - 1) * 6
            + joint_num * 3,
        ]
        foot = hml_data[..., -4:]

        local_rots_body = local_rots[..., : (body_joints - 1) * 6]
        local_rots_hand = local_rots[..., -hand_joints * 6 :]

        local_pos_body = local_pos[..., : (body_joints - 1) * 3]
        local_pos_hand = local_pos[..., -hand_joints * 3 :]

        local_vel_body = local_vels[..., : (body_joints) * 3]
        local_vel_hand = local_vels[..., -hand_joints * 3 :]

        if self.use_rotation:
            body_params = np.concatenate(
                [root_params, local_pos_body, local_rots_body, local_vel_body, foot], -1
            )
            hand_params = np.concatenate(
                [local_pos_hand, local_rots_hand, local_vel_hand], -1
            )

        if not self.use_rotation:
            body_params = np.concatenate(
                [root_params, local_pos_body, local_vel_body, foot], -1
            )
            hand_params = np.concatenate([local_pos_hand, local_vel_hand], -1)

            hml_data = np.concatenate(
                [
                    hml_data[..., : 4 + (joint_num - 1) * 3],
                    hml_data[..., 4 + (joint_num - 1) * 3 + (joint_num - 1) * 6 :],
                ],
                -1,
            )

        return body_params, hand_params, hml_data

    def _get_motion_padded(
        self,
        max_length: int,
        motion_list: List[np.ndarray],
        down_sampling_factor=4,
        truncation="rand_trunc",
        padding="longest",
        subset_index=None,
    ):
        motions = []
        masks = []
        if padding == "longest" or max_length is None:
            max_length_ = (
                max([motion.shape[0] for motion in motion_list]) // down_sampling_factor
            ) * down_sampling_factor
            max_length = min(max_length_, max_length)

        for full_motion in motion_list:
            full_motion = (full_motion - self.mean) / (self.std + 1e-8)
            body_params, hand_params, full_params = self.hmldata_process(full_motion)
            if self.motion_type == "body":
                motion = body_params
            elif self.motion_type == "hand":
                motion = hand_params
            else:
                motion = full_params

            seq_len = motion.shape[0]

            if seq_len > max_length:
                # if truncation == "rand_trunc":
                # random crop to max_length (for compatibility) -> this should be handled by self.pad
                overflow = seq_len - max_length
                idx = (
                    subset_index
                    if subset_index is not None
                    else np.random.randint(0, overflow + 1)
                )
                motion = motion[idx : idx + max_length]
                mask = np.array([1] * max_length)
                motions.append(motion[None, ...])
                masks.append(mask[None, ...])

            else:

                motion = motion[
                    : (seq_len // down_sampling_factor) * down_sampling_factor
                ]
                seq_len = (seq_len // down_sampling_factor) * down_sampling_factor
                if padding == "repeat":
                    n_repeat = int(max_length / seq_len)
                    motion = np.stack(np.tile(motion, n_repeat + 1))[:max_length]
                elif padding == "repeatpad":
                    n_repeat = int(max_length / seq_len)
                    motion = np.stack(np.tile(motion, n_repeat))

                # if padding == "longest":

                pad_motion = np.concatenate(
                    [motion, np.zeros((max_length - motion.shape[0], motion.shape[-1]))]
                )
                mask = np.array(
                    [1] * motion.shape[0] + [0] * (max_length - motion.shape[0])
                )

                motions.append(pad_motion[None, ...])
                masks.append(mask[None, ...])

        padded_motion = np.concatenate(motions, 0)
        attention_mask = np.concatenate(masks, 0)

        return padded_motion, attention_mask

    def __call__(
        self,
        raw_speech: Optional[
            Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]
        ] = None,
        raw_motion: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        truncation: str = None,
        padding: Optional[str] = None,
        max_length: Optional[int] = None,
        motion_padding: Optional[str] = None,
        max_motion_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            truncation (`str`, *optional*):
                Truncation pattern for long audio inputs. Two patterns are available:
                    - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and
                      a downsampled version of the entire mel spectrogram.
                If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a
                copy of the original mel obtained from the padded audio.
                    - `rand_trunc` will select a random crop of the mel spectrogram.
            padding (`str`, *optional*):
               Padding pattern for shorter audio inputs. Three patterns were originally implemented:
                    - `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
                    - `repeat`: the audio is repeated and then cut to fit the `max_length`
                    - `pad`: the audio is padded.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.np.array` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
        """
        truncation = truncation if truncation is not None else self.truncation
        padding = padding if padding else self.padding
        motion_padding = motion_padding if motion_padding else self.motion_padding

        max_length = max_length if max_length is not None else self.nb_max_samples
        max_motion_length = (
            max_motion_length
            if max_motion_length is not None
            else self.motion_max_length
        )

        subset_idx_motion, subset_idx_audio = None, None

        if raw_motion is not None and raw_speech is not None:
            common_seconds = min(
                raw_motion.shape[0] // self.fps,
                raw_speech.shape[0] // self.sampling_rate,
            )
            raw_motion = raw_motion[: int(common_seconds * self.fps)]
            raw_speech = raw_speech[: int(common_seconds * self.sampling_rate)]

            if common_seconds > max_length // self.sampling_rate:
                subset_idx_motion = np.random.randint(
                    0, raw_motion.shape[0] - max_motion_length + 1
                )
                subset_idx_audio = np.random.randint(
                    0, raw_speech.shape[0] - max_length + 1
                )

        input_features = {}

        if raw_speech is not None:

            if sampling_rate is not None:
                if sampling_rate != self.sampling_rate:
                    raise ValueError(
                        f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                        f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                        f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                    )
            else:
                logger.warning(
                    "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                    "Failing to do so can result in silent errors that might be hard to debug."
                )

            is_batched = bool(
                isinstance(raw_speech, (list, tuple))
                and (
                    isinstance(raw_speech[0], np.ndarray)
                    or isinstance(raw_speech[0], (tuple, list))
                )
            )

            if is_batched:
                raw_speech = [
                    np.asarray(speech, dtype=np.float64) for speech in raw_speech
                ]
            elif not is_batched and not isinstance(raw_speech, np.ndarray):
                raw_speech = np.asarray(raw_speech, dtype=np.float64)
            elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(
                np.float64
            ):
                raw_speech = raw_speech.astype(np.float64)

            # always return batch
            if not is_batched:
                raw_speech = [np.asarray(raw_speech)]

            # convert to mel spectrogram, truncate and pad if needed.
            padded_inputs = []
            for waveform in raw_speech:
                if waveform is None:
                    padded_inputs.append((np.zeros(1, max_length, 64), False))

                padded_inputs.append(
                    self._get_input_mel(
                        waveform,
                        max_length,
                        truncation,
                        padding,
                        subset_index=subset_idx_audio,
                    )
                )

            input_mel = []
            is_longer = []
            for mel, longer in padded_inputs:
                input_mel.append(mel)
                is_longer.append(longer)

            if truncation == "fusion" and sum(is_longer) == 0:
                # if no audio is longer than 10s, then randomly select one audio to be longer
                rand_idx = np.random.randint(0, len(input_mel))
                is_longer[rand_idx] = True

            if isinstance(input_mel[0], List):
                input_mel = [
                    np.asarray(feature, dtype=np.float64) for feature in input_mel
                ]

            # is_longer is a list of bool
            is_longer = [[longer] for longer in is_longer]

            input_features["input_features"] = np.concatenate(input_mel, 0)
            input_features["is_longer"] = is_longer

            # = {"input_features": input_mel, "is_longer": is_longer}

        if raw_motion is not None:
            is_batched = bool(
                isinstance(raw_motion, (list, tuple))
                and (isinstance(raw_motion[0], np.ndarray))
            )

            if is_batched:
                raw_motion = [
                    np.asarray(motion, dtype=np.float64) for motion in raw_motion
                ]
            elif not is_batched and not isinstance(raw_motion, np.ndarray):
                raw_motion = np.asarray(raw_motion, dtype=np.float64)
            # always return batch
            if not is_batched:
                raw_motion = [np.asarray(raw_motion)]

            padded_motion, attention_mask = self._get_motion_padded(
                max_length=max_motion_length,
                motion_list=raw_motion,
                padding=motion_padding,
                subset_index=subset_idx_motion,
            )

            input_features["input_motion_features"] = padded_motion
            input_features["motion_mask"] = attention_mask

        input_features = BatchFeature(input_features)

        if return_tensors is not None:
            input_features = input_features.convert_to_tensors(return_tensors)

        return input_features
