import itertools
import os
from collections import Counter
from math import sqrt
from pathlib import Path
import random

import numpy as np
import torch
import transformers
import wandb


from clamp_dataset import load_dataset, simple_collate
from functools import partial

from modeling_clamp import ClampModel
from configuration_clamp import ClampConfig

from feature_extraction_clamp import ClampFeatureExtractor
from transformers import RobertaTokenizer
from processing_clamp import ClampProcessor

from optimizer import get_optimizer
from torch import nn
from tqdm import tqdm
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from training_config import cfg, get_cfg_defaults

from yacs.config import CfgNode


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


# main trainer class


class ClampTrainer(nn.Module):
    def __init__(
        self,
        args: CfgNode,
    ):
        super().__init__()
        self.model_name = args.model_name

        transformers.set_seed(42)

        self.args = args
        self.training_args = args.train
        self.dataset_args = args.dataset

        self.num_train_steps = self.training_args.num_train_iters
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clamp_config = ClampConfig.from_pretrained(
            "/srv/hays-lab/scratch/sanisetty3/music_motion/clamp/clamp"
        )
        self.clamp_model = ClampModel.from_pretrained(
            "/srv/hays-lab/scratch/sanisetty3/music_motion/clamp/clamp"
        )

        freeze_parameters = [
            p
            for n, p in self.clamp_model.named_parameters()
            if "text_model" in n
            or "text_projection" in n
            or "audio_model" in n
            or "audio_projection" in n
            or "motion_model" in n
            or "motion_model.quantizer.codebook.weight" in n
            or "logit_scale_a" in n
            or "logit_scale_t" in n
        ]
        print("Freeze Text and Audio!!!!")

        for k in freeze_parameters:
            k.requires_grad = False

        clamp_feature_extractor = ClampFeatureExtractor.from_pretrained(
            "/srv/hays-lab/scratch/sanisetty3/music_motion/clamp/clamp/"
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            "/srv/hays-lab/scratch/sanisetty3/music_motion/clamp/clamp"
        )
        self.clamp_processor = ClampProcessor(clamp_feature_extractor, tokenizer)

        self.register_buffer("steps", torch.Tensor([0]))

        self.grad_accum_every = self.training_args.gradient_accumulation_steps

        self.optim = get_optimizer(
            self.clamp_model.named_parameters(),
            freeze_parameters,
            lr=self.training_args.learning_rate,
            wd=self.training_args.weight_decay,
        )

        total = sum(p.numel() for p in self.clamp_model.parameters() if p.requires_grad)
        print("Total training params: %.2fM" % (total / 1e6))

        self.lr_scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.num_train_steps,
        )

        train_ds, sampler_train, weights_train = load_dataset(
            dataset_root=self.dataset_args.dataset_root,
            split="train",
            weight_scale=[1, 2, 0.8, 1, 1, 1, 1, 0.5, 1, 2, 1, 2, 2, 1, 1, 1, 1],
        )
        test_ds, _, _ = load_dataset(
            dataset_root=self.dataset_args.dataset_root,
            split="test",
        )

        self.print(
            f"training with training {len(train_ds)} and test dataset of and {len(test_ds)} samples"
        )

        # dataloader

        self.dl = DataLoader(
            train_ds,
            batch_size=self.training_args.train_bs,
            sampler=sampler_train,
            shuffle=False if sampler_train else True,
            collate_fn=partial(simple_collate, clamp_processor=self.clamp_processor),
        )
        self.valid_dl = DataLoader(
            test_ds,
            batch_size=self.training_args.eval_bs,
            shuffle=False,
            collate_fn=partial(simple_collate, clamp_processor=self.clamp_processor),
        )

        self.dl_iter = cycle(self.dl)
        # self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = self.training_args.save_steps
        self.log_losses_every = self.training_args.logging_steps
        self.evaluate_every = self.training_args.evaluate_every
        self.calc_metrics_every = self.training_args.evaluate_every
        self.wandb_every = self.training_args.wandb_every

        # if self.is_main:
        wandb.login()
        wandb.init(project=self.model_name)

    def print(self, msg):
        # self.accelerator.print(msg)
        print(msg)

    @property
    def device(self):
        return torch.device("cuda")

    def save(self, path, loss=None):
        pkg = dict(
            model=self.clamp_model.state_dict(),
            optim=self.optim.state_dict(),
            steps=self.steps,
            total_loss=self.best_loss if loss is None else loss,
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cuda")
        self.clamp_model.load_state_dict(pkg["model"])

        self.optim.load_state_dict(pkg["optim"])
        self.steps = pkg["steps"]
        self.best_loss = pkg["total_loss"]

    def train_step(self):
        steps = int(self.steps.item())

        self.clamp_model = self.clamp_model.train()

        # logs

        logs = {}

        for _ in range(self.grad_accum_every):
            batch = next(self.dl_iter)

            output = self.clamp_model(**batch, return_loss=True)

            loss = output.loss / self.grad_accum_every
            # logits_per_text_vs_motion = output.logits_per_text_vs_motion
            # probs_tvm = logits_per_text_vs_motion.softmax(dim=-1)
            # logits_per_motion_vs_text = output.logits_per_motion_vs_text
            # probs_mvt = logits_per_motion_vs_text.softmax(dim=-1)

            loss.backward()

            accum_log(
                logs,
                dict(
                    loss=loss.detach().cpu(),
                    # probs_tvm=probs_tvm.detach().cpu(),
                    # probs_mvt=probs_mvt.detach().cpu(),
                ),
            )

        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()

        # build pretty printed losses

        losses_str = f"{steps}: model total contrastive loss: {logs['loss'].float():.3}"

        # log
        if steps % self.wandb_every == 0:
            for key, value in logs.items():
                wandb.log({f"train_loss/{key}": value})

            self.print(losses_str)

        if steps % self.evaluate_every == 0:
            self.validation_step()
            # self.sample_render_hmlvec(os.path.join(self.output_dir, "samples"))

        # if self.is_main and not (steps % self.save_model_every) and steps > 0:
        if not (steps % self.save_model_every):
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            model_path = os.path.join(
                self.output_dir, "checkpoints", f"clamp_motion.{steps}.pt"
            )
            self.save(model_path)
            print(float(logs["loss"]), self.best_loss)

            if float(logs["loss"]) <= self.best_loss:
                model_path = os.path.join(self.output_dir, f"clamp_motion.pt")
                self.save(model_path)
                self.best_loss = logs["loss"]

            self.print(
                f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}'
            )

        self.steps += 1
        return logs

    def validation_step(self):
        self.clamp_model.eval()
        val_loss_ae = {}

        self.print(f"validation start")

        with torch.no_grad():
            for batch in tqdm(
                (self.valid_dl),
                position=0,
                leave=True,
            ):
                output = self.clamp_model(**batch, return_loss=True)

                loss = output.loss
                logits_per_text_vs_motion = output.logits_per_text_vs_motion
                probs_tvm = logits_per_text_vs_motion.softmax(dim=-1)
                logits_per_motion_vs_text = output.logits_per_motion_vs_text
                probs_mvt = logits_per_motion_vs_text.softmax(dim=-1)

                loss_dict = {
                    "total_loss": loss.detach().cpu(),
                    "probs_tvm": torch.diag(probs_tvm.detach().cpu()),
                    "probs_mvt": torch.diag(probs_mvt.detach().cpu()),
                }

                val_loss_ae.update(loss_dict)

                sums_ae = dict(Counter(val_loss_ae) + Counter(loss_dict))
                means_ae = {
                    k: sums_ae[k] / float((k in val_loss_ae) + (k in loss_dict))
                    for k in sums_ae
                }
                val_loss_ae.update(means_ae)

        for key, value in val_loss_ae.items():
            wandb.log({f"val_loss_vqgan/{key}": value})

        print(
            f"val/total_loss ",
            val_loss_ae["total_loss"],
        )

        print(
            "val/rec_loss",
            val_loss_ae["probs_tvm"],
        )

        self.clamp_model.train()

    def train(self, resume=False):
        self.best_loss = float("inf")
        print(self.output_dir)

        if resume:
            save_dir = self.args.output_dir
            save_path = os.path.join(save_dir, "clamp_motion.pt")
            print("resuming from ", save_path)
            self.load(save_path)

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            # log_fn(logs)

        self.print("training complete")


if __name__ == "__main__":
    nme = "clamp_enc"
    path = f"/srv/hays-lab/scratch/sanisetty3/music_motion/clamp/checkpoints/{nme}/{nme}.yaml"
    cfg = get_cfg_defaults()
    print("loading config from:", path)
    cfg.merge_from_file(path)
    cfg.freeze()
    print("output_dir: ", cfg.output_dir)

    trainer = ClampTrainer(
        args=cfg,
    ).cuda()

    trainer.train(cfg.train.resume)
