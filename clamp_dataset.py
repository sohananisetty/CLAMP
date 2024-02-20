import codecs as cs
import os
import random
from glob import glob
from os.path import join as pjoin

import clip
import numpy as np
import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from typing import Tuple, Optional, List, Dict
import itertools
from processing_clamp import ClampProcessor
import json
import torchaudio


genre_dict = {
    "mBR": "Break",
    "mPO": "Pop",
    "mLO": "Lock",
    "mMH": "Middle_Hip-hop",
    "mLH": "LAstyle Hip-hop",
    "mHO": "House",
    "mWA": "Waack",
    "mKR": "Krump",
    "mJS": "Street_Jazz",
    "mJB": "Ballet_Jazz",
}

dataset_names_default = [
    "animation",
    "humanml",
    "perform",
    "GRAB",
    "idea400",
    "humman",
    "beat",
    "game_motion",
    "music",
    "aist",
    "fitness",
    "moyo",
    "choreomaster",
    "dance",
    "kungfu",
    "EgoBody",
    "HAA500",
]


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(
            f"Impossible to convert from {channels} to {target_channels}"
        )
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def load_dataset(
    dataset_root,
    dataset_names=dataset_names_default,
    motion_min_length_s=2,
    motion_max_length_s=10,
    split: str = "train",
    weight_scale: Optional[List[int]] = None,
    # dataset_class=CLAMPDataset,
):
    if weight_scale is None:
        weight_scale = [1] * len(dataset_names)
    assert len(dataset_names) == len(weight_scale), "mismatch in size"
    dataset_list = []
    weights = []
    for dataset_name in dataset_names:
        dataset_list.append(
            CLAMPDataset(
                dataset_name,
                dataset_root=dataset_root,
                split=split,
                motion_min_length_s=motion_min_length_s,
                motion_max_length_s=motion_max_length_s,
            )
        )

    concat_dataset = torch.utils.data.ConcatDataset(dataset_list)

    if split != "train" or len(dataset_names) == 1:
        return concat_dataset, None, None

    for i, ds in enumerate(dataset_list):
        weights.append(
            [weight_scale[i] * concat_dataset.__len__() / (ds.__len__())] * ds.__len__()
        )

    weights = list(itertools.chain.from_iterable(weights))

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights)
    )

    return concat_dataset, sampler, weights


class CLAMPDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        motion_min_length_s=2,
        motion_max_length_s=10,
        sampling_rate: int = 48000,
        fps: int = 30,
        split: str = "train",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.fps = fps

        self.min_motion_length = motion_min_length_s * fps
        self.max_motion_length = motion_max_length_s * fps

        data_root = dataset_root

        self.text_dir = os.path.join(data_root, "texts/semantic_labels")
        self.face_text_dir = os.path.join(data_root, "texts/face_texts")
        self.motion_dir = os.path.join(data_root, "motion_data/new_joint_vecs")
        self.audio_dir = os.path.join(data_root, "audio")

        self.sampling_rate = sampling_rate

        split_file = os.path.join(data_root, f"motion_data/{split}.txt")

        self.id_list = []
        self.text_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                if dataset_name + "/" in line:
                    # try:
                    motion = np.load(os.path.join(self.motion_dir, line.strip()))
                    if motion.shape[0] < self.min_motion_length:
                        continue

                    if self.dataset_name == "humanml":
                        name_list, txt_list = self.load_humanml(line.strip())

                    if self.dataset_name == "beat":
                        name_list, txt_list = self.load_beat(line.strip())

                    else:
                        name_list, txt_list = self.load_txt(line.strip())

                    self.id_list.extend(name_list)
                    self.text_list.extend(txt_list)

                    # except:
                    #     continue

        print(
            f"Total number of motions {dataset_name}: {len(self.id_list)} and texts {len(self.text_list)}"
        )

    def __len__(self) -> int:
        return len(self.id_list)

    def load_txt(self, name):
        name = name.split(".")[0]
        new_name = f"{name}_0_0"
        name_list = []
        txt_list = []

        with open(os.path.join(self.text_dir, name + ".txt"), "r") as f:
            for line in f.readlines():
                name_list.append(new_name)
                txt_list.append(line.strip())

        return name_list, txt_list

    def load_humanml(self, name):
        name = name.split(".")[0]
        # data_dict = {}
        name_list = []
        txt_list = []
        with open(os.path.join(self.text_dir, name + ".txt"), "r") as f:
            for line in f.readlines():
                line_split = line.strip().split("#")
                caption = line_split[0]
                tokens = line_split[1].split(" ")
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                new_name = f"{name}_{int(f_tag * self.fps)}_{int(to_tag * self.fps)}"

                name_list.append(new_name)
                txt_list.append(caption)

        return name_list, txt_list

    def load_beat(self, name):
        name = name.split(".")[0]
        new_name = f"{name}_0_0"
        name_list = []
        txt_list = []
        with open(
            os.path.join(
                self.text_dir.replace("semantic_labels", "body_texts"), name + ".json"
            ),
            "r",
        ) as outfile:
            frame_texts = json.load(outfile)

        items = list(frame_texts.values())
        # sentence = (" ".join(list(dict.fromkeys(items)))).strip()
        name_list.append(new_name)
        txt_list.append(" ".join(items))

        return name_list, txt_list

    def mask_augment(self, motion, perc_n=0.0, perc_d=0.0):
        n, d = motion.shape
        num_masked_n = int(n * perc_n)
        num_masked_d = int(d * perc_d)

        n_ind = list(np.random.choice(np.arange(n), num_masked_n))
        d_ind = list(np.random.choice(np.arange(d), num_masked_d))

        motion[n_ind, :] = 0
        motion[:, d_ind] = 0

        return motion

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        # print(self.id_list[item])
        name, f_, to_ = self.id_list[item].rsplit("_", 2)
        f_, to_ = int(f_), int(to_)
        motion = np.load(os.path.join(self.motion_dir, name + ".npy"))
        text = self.text_list[item]
        try:
            wav, sr = torchaudio.load(os.path.join(self.audio_dir, name + ".wav"))
            audio_data = convert_audio(wav, sr, self.sampling_rate, 1)

            # audio_data, _ = librosa.load(
            #     os.path.join(self.audio_dir, name + ".wav"),
            #     sr=48000,
            # )
        except:
            audio_data = None
        if to_ - f_ > self.min_motion_length:
            motion = motion[f_:to_]
            text = text[f_:to_]

        text = (" ".join(list(dict.fromkeys(text.split(" "))))).strip()

        return {
            "name": name,
            "motion": motion,
            "text": text,
            "audio": audio_data,
        }


def simple_collate(
    samples: List[Tuple[torch.Tensor, str]], clamp_processor: ClampProcessor
) -> Dict[str, torch.Tensor]:
    motions = []
    texts = []
    audios = []
    names = []

    for sample in samples:
        names.append(sample["name"])
        motions.append(sample["motion"])
        texts.append(sample["text"])
        audios.append(sample["audio"])

    # if any(elem is None for elem in audios):
    #     audios = None
    # if None in texts:
    #     texts = None
    # if any(elem is None for elem in motions):
    #     motions = None

    inputs = clamp_processor(
        text=texts, audios=audios, motions=motions, return_tensors="pt", padding=True
    )
    inputs["names"] = np.array(names)
    inputs["texts"] = np.array(texts)

    return inputs
