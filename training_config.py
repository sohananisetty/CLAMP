"""
Default config
"""

# import argparse
# import yaml
import os
from glob import glob

from yacs.config import CfgNode as CN

cfg = CN()
cfg.abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cfg.device = "cuda"
cfg.model_name = "clamp"
cfg.pretrained_modelpath = os.path.join(cfg.abs_dir, f"clamp/")
cfg.output_dir = os.path.join(cfg.abs_dir, "checkpoints/")
cfg.dataset = CN()
cfg.dataset.dataset_root = "/srv/hays-lab/scratch/sanisetty3/motionx"
cfg.dataset.fps = 30
cfg.train = CN()
cfg.train.resume = False
cfg.train.seed = 42
cfg.train.fp16 = False
cfg.train.num_train_iters = 500000  #'Number of training steps
cfg.train.save_steps = 5000
cfg.train.logging_steps = 10
cfg.train.wandb_every = 100
cfg.train.evaluate_every = 5000
cfg.train.eval_bs = 20
cfg.train.train_bs = 24
cfg.train.gradient_accumulation_steps = 4
cfg.train.motion_max_length_s = 10
cfg.train.motion_min_length_s = 3
## optimization
cfg.train.learning_rate = 2e-4
cfg.train.weight_decay = 0.0
cfg.train.warmup_steps = 4000
cfg.train.gamma = 0.05
cfg.train.lr_scheduler_type = "cosine"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()
