# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import lightning.pytorch as pl
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import omegaconf
import torch
import torch.nn as nn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from solo.data.temp_dali_fix import TempDALIGenericIterator
from solo.utils.misc import omegaconf_select


class RandomGrayScaleConversion:
    def __init__(self, prob: float = 0.2, device: str = "gpu"):
        """Converts image to greyscale with probability.

        Args:
            prob (float, optional): probability of conversion. Defaults to 0.2.
            device (str, optional): device on which the operation will be performed.
                Defaults to "gpu".
        """

        self.prob = prob
        self.grayscale = ops.ColorSpaceConversion(
            device=device, image_type=types.RGB, output_type=types.GRAY
        )

    def __call__(self, images):
        do_op = fn.random.coin_flip(probability=self.prob, dtype=types.DALIDataType.BOOL)
        if do_op:
            out = self.grayscale(images)
            out = fn.cat(out, out, out, axis=2)
        else:
            out = images
        return out


class RandomColorJitter:
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        prob: float = 0.8,
        device: str = "gpu",
    ):
        """Applies random color jittering with probability.

        Args:
            brightness (float): brightness value for samplying uniformly
                in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): contrast value for samplying uniformly
                in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): saturation value for samplying uniformly
                in [max(0, 1 - saturation), 1 + saturation].
            hue (float): hue value for samplying uniformly in [-hue, hue].
            prob (float, optional): probability of applying jitter. Defaults to 0.8.
            device (str, optional): device on which the operation will be performed.
                Defaults to "gpu".
        """

        assert 0 <= hue <= 0.5

        self.prob = prob

        self.color = ops.ColorTwist(device=device)

        # look at torchvision docs to see how colorjitter samples stuff
        # for bright, cont and sat, it samples from [1-v, 1+v]
        # for hue, it samples from [-hue, hue]

        self.brightness = 1
        self.contrast = 1
        self.saturation = 1
        self.hue = 0

        if brightness:
            self.brightness = ops.random.Uniform(range=[max(0, 1 - brightness), 1 + brightness])

        if contrast:
            self.contrast = ops.random.Uniform(range=[max(0, 1 - contrast), 1 + contrast])

        if saturation:
            self.saturation = ops.random.Uniform(range=[max(0, 1 - saturation), 1 + saturation])

        if hue:
            # dali uses hue in degrees for some reason...
            hue = 360 * hue
            self.hue = ops.random.Uniform(range=[-hue, hue])

    def __call__(self, images):
        do_op = fn.random.coin_flip(probability=self.prob, dtype=types.DALIDataType.BOOL)
        if do_op:
            out = self.color(
                images,
                brightness=self.brightness() if callable(self.brightness) else self.brightness,
                contrast=self.contrast() if callable(self.contrast) else self.contrast,
                saturation=self.saturation() if callable(self.saturation) else self.saturation,
                hue=self.hue() if callable(self.hue) else self.hue,
            )
        else:
            out = images
        return out


class RandomGaussianBlur:
    def __init__(self, prob: float = 0.5, window_size: int = 23, device: str = "gpu"):
        """Applies random gaussian blur with probability.

        Args:
            prob (float, optional): probability of applying random gaussian blur. Defaults to 0.5.
            window_size (int, optional): window size for gaussian blur. Defaults to 23.
            device (str, optional): device on which the operation will be performe.
                Defaults to "gpu".
        """

        self.prob = prob
        # gaussian blur
        self.gaussian_blur = ops.GaussianBlur(device=device, window_size=(window_size, window_size))
        self.sigma = ops.random.Uniform(range=[0, 1])

    def __call__(self, images):
        do_op = fn.random.coin_flip(probability=self.prob, dtype=types.DALIDataType.BOOL)
        if do_op:
            sigma = self.sigma() * 1.9 + 0.1
            out = self.gaussian_blur(images, sigma=sigma)
        else:
            out = images
        return out


class RandomSolarize:
    def __init__(self, threshold: int = 128, prob: float = 0.0):
        """Applies random solarization with probability.

        Args:
            threshold (int, optional): threshold for inversion. Defaults to 128.
            prob (float, optional): probability of solarization. Defaults to 0.0.
        """

        self.prob = prob
        self.threshold = threshold

    def __call__(self, images):
        do_op = fn.random.coin_flip(probability=self.prob, dtype=types.DALIDataType.BOOL)
        if do_op:
            inverted_img = types.Constant(255, dtype=types.UINT8) - images
            mask = images >= self.threshold
            out = mask * inverted_img + (True ^ mask) * images
        else:
            out = images
        return out


class NormalPipelineBuilder:
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        device: str,
        validation: bool = False,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
        data_fraction: float = -1.0,
    ):
        """Initializes the pipeline for validation or linear eval training.

        If validation is set to True then images will only be resized to 256px and center cropped
        to 224px, otherwise random resized crop, horizontal flip are applied. In both cases images
        are normalized.

        Args:
            data_path (str): directory that contains the data.
            batch_size (int): batch size.
            device (str): device on which the operation will be performed.
            validation (bool): whether it is validation or training. Defaults to False. Defaults to
                False.
            device_id (int): id of the device used to initialize the seed and for parent class.
                Defaults to 0.
            shard_id (int): id of the shard (chuck of samples). Defaults to 0.
            num_shards (int): total number of shards. Defaults to 1.
            num_threads (int): number of threads to run in parallel. Defaults to 4.
            seed (int): seed for random number generation. Defaults to 12.
            data_fraction (float): percentage of data to use. Use all data when set to -1.0.
                Defaults to -1.0.
        """

        super().__init__()

        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.seed = seed + device_id

        self.device = device
        self.validation = validation

        # manually load files and labels
        labels = sorted(Path(entry.name) for entry in os.scandir(data_path) if entry.is_dir())
        data = [
            (data_path / label / file, label_idx)
            for label_idx, label in enumerate(labels)
            for file in sorted(os.listdir(data_path / label))
        ]
        files, labels = map(list, zip(*data))

        # sample data if needed
        if data_fraction > 0:
            assert data_fraction < 1, "data_fraction must be smaller than 1."

            from sklearn.model_selection import train_test_split

            files, _, labels, _ = train_test_split(
                files, labels, train_size=data_fraction, stratify=labels, random_state=42
            )

        self.reader = ops.readers.File(
            files=files,
            labels=labels,
            shard_id=shard_id,
            num_shards=num_shards,
            shuffle_after_epoch=not self.validation,
        )
        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )

        # crop operations
        if self.validation:
            self.resize = ops.Resize(
                device=self.device,
                resize_shorter=256,
                interp_type=types.INTERP_CUBIC,
            )
            # center crop and normalize
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(224, 224),
                mean=[v * 255 for v in IMAGENET_DEFAULT_MEAN],
                std=[v * 255 for v in IMAGENET_DEFAULT_STD],
            )
        else:
            self.resize = ops.RandomResizedCrop(
                device=self.device,
                size=224,
                random_area=(0.08, 1.0),
                interp_type=types.INTERP_CUBIC,
            )
            # normalize and horizontal flip
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                mean=[v * 255 for v in IMAGENET_DEFAULT_MEAN],
                std=[v * 255 for v in IMAGENET_DEFAULT_STD],
            )

        self.coin05 = ops.random.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

    @pipeline_def
    def pipeline(self):
        """Defines the computational pipeline for dali operations."""

        # read images from memory
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)

        # crop into large and small images
        images = self.resize(images)

        if self.validation:
            # crop and normalize
            images = self.cmn(images)
        else:
            # normalize and maybe apply horizontal flip with 0.5 chance
            images = self.cmn(images, mirror=self.coin05())

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return (images, labels)


def build_transform_pipeline_dali(dataset, cfg, dali_device):
    """Creates a pipeline of transformations given a dataset and an augmentation Cfg node.
    The node needs to be in the following format:
        crop_size: int
        [OPTIONAL] mean: float
        [OPTIONAL] std: float
        rrc:
            enabled: bool
            crop_min_scale: float
            crop_max_scale: float
        color_jitter:
            prob: float
            brightness: float
            contrast: float
            saturation: float
            hue: float
        grayscale:
            prob: float
        gaussian_blur:
            prob: float
        solarization:
            prob: float
        equalization:
            prob: float
        horizontal_flip:
            prob: float
    """

    MEANS_N_STD = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        "imagenet100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "radimagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    }

    mean, std = MEANS_N_STD.get(
        dataset, (cfg.get("mean", IMAGENET_DEFAULT_MEAN), cfg.get("std", IMAGENET_DEFAULT_STD))
    )

    augmentations = []
    if cfg.rrc.enabled:
        augmentations.append(
            ops.RandomResizedCrop(
                device=dali_device,
                size=cfg.crop_size,
                random_area=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                interp_type=types.INTERP_CUBIC,
            )
        )
    else:
        augmentations.append(
            ops.Resize(
                device=dali_device,
                size=(cfg.crop_size, cfg.crop_size),
                interp_type=types.INTERP_CUBIC,
            )
        )

    if cfg.color_jitter.prob:
        augmentations.append(
            RandomColorJitter(
                brightness=cfg.color_jitter.brightness,
                contrast=cfg.color_jitter.contrast,
                saturation=cfg.color_jitter.saturation,
                hue=cfg.color_jitter.hue,
                prob=cfg.color_jitter.prob,
                device=dali_device,
            )
        )

    if cfg.grayscale.prob:
        augmentations.append(RandomGrayScaleConversion(prob=cfg.grayscale.prob, device=dali_device))

    if cfg.gaussian_blur.prob:
        augmentations.append(RandomGaussianBlur(prob=cfg.gaussian_blur.prob, device=dali_device))

    if cfg.solarization.prob:
        augmentations.append(RandomSolarize(prob=cfg.solarization.prob))

    if cfg.equalization.prob:
        raise NotImplementedError(
            "Equalization is not available for DALI"
            "turn it off by setting augmentations.equalization.enabled to False."
        )

    coin = None
    if cfg.horizontal_flip.prob:
        coin = ops.random.CoinFlip(probability=cfg.horizontal_flip.prob)

    cmn = ops.CropMirrorNormalize(
        device=dali_device,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        mean=[v * 255 for v in mean],
        std=[v * 255 for v in std],
    )

    class AugWrapper:
        def __init__(self, augmentations, cmn, coin) -> None:
            self.augmentations = augmentations
            self.cmn = cmn
            self.coin = coin

        def __call__(self, images):
            for aug in self.augmentations:
                images = aug(images)

            if self.coin:
                images = self.cmn(images, mirror=self.coin())
            else:
                images = self.cmn(images)
            return images

        def __repr__(self) -> str:
            return f"{self.augmentations}"

    return AugWrapper(augmentations=augmentations, cmn=cmn, coin=coin)


class Wrapper(TempDALIGenericIterator):
    def __init__(self, dataset_size: int, *args, **kwargs):
        """Wrapper to have dataset size.

        Args:
            dataset_size (int): number of samples in the dataset.
        """

        super().__init__(*args, **kwargs)
        self.dataset_size = dataset_size

    def __next__(self):
        batch = super().__next__()[0]
        x, target = batch["x"], batch["label"]
        target = target.squeeze(-1).long()
        # PyTorch Lightning does double buffering
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1316,
        # and as DALI owns the tensors it returns the content of it is trashed so the copy needs,
        # to be made before returning.
        x = x.detach().clone()
        target = target.detach().clone()
        return x, target


class ClassificationDALIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        train_data_path: Union[str, Path],
        val_data_path: Union[str, Path],
        batch_size: int,
        num_workers: int = 4,
        data_fraction: float = -1.0,
        dali_device: str = "gpu",
    ):
        """DataModule for classification data using Nvidia DALI.

        Args:
            dataset (str): dataset name.
            train_data_path (Union[str, Path]): path where the training data is located.
            val_data_path (Union[str, Path]): path where the validation data is located.
            batch_size (int): batch size..
            num_workers (int, optional): number of parallel workers. Defaults to 4.
            data_fraction (float, optional): percentage of data to use.
                Use all data when set to -1.0. Defaults to -1.0.
            dali_device (str, optional): device used by the dali pipeline.
                Either 'gpu' or 'cpu'. Defaults to 'gpu'.
        """

        super().__init__()

        self.dataset = dataset

        # paths
        self.train_data_path = Path(train_data_path)
        self.val_data_path = Path(val_data_path)

        self.num_workers = num_workers

        self.batch_size = batch_size

        self.data_fraction = data_fraction

        self.dali_device = dali_device
        assert dali_device in ["gpu", "cpu"]

        # handle custom data by creating the needed pipeline
        if dataset in ["imagenet100", "imagenet", "radimagenet"]:
            self.pipeline_class = NormalPipelineBuilder
        elif dataset == "custom":
            self.pipeline_class = CustomNormalPipelineBuilder
        else:
            raise ValueError(dataset, "is not supported, used [imagenet, imagenet100, radimagenet or custom]")

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg.dali = omegaconf_select(cfg, "dali", {})
        cfg.dali.device = omegaconf_select(cfg, "dali.device", "gpu")
        return cfg

    def setup(self, stage: Optional[str] = None):
        # extra info about training
        self.device_id = self.trainer.local_rank
        self.shard_id = self.trainer.global_rank
        self.num_shards = self.trainer.world_size

        # get current device
        if torch.cuda.is_available() and self.dali_device == "gpu":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        train_pipeline_builder = self.pipeline_class(
            self.train_data_path,
            validation=False,
            batch_size=self.batch_size,
            device=self.dali_device,
            device_id=self.device_id,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            num_threads=self.num_workers,
            data_fraction=self.data_fraction,
        )
        train_pipeline = train_pipeline_builder.pipeline(
            batch_size=train_pipeline_builder.batch_size,
            num_threads=train_pipeline_builder.num_threads,
            device_id=train_pipeline_builder.device_id,
            seed=train_pipeline_builder.seed,
        )
        train_pipeline.build()

        self.train_loader = Wrapper(
            pipelines=train_pipeline,
            dataset_size=train_pipeline.epoch_size("Reader"),
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        )

        val_pipeline_builder = self.pipeline_class(
            self.val_data_path,
            validation=True,
            batch_size=self.batch_size,
            device=self.dali_device,
            device_id=self.device_id,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            num_threads=self.num_workers,
        )
        val_pipeline = val_pipeline_builder.pipeline(
            batch_size=val_pipeline_builder.batch_size,
            num_threads=val_pipeline_builder.num_threads,
            device_id=val_pipeline_builder.device_id,
            seed=val_pipeline_builder.seed,
        )
        val_pipeline.build()

        self.val_loader = Wrapper(
            pipelines=val_pipeline,
            dataset_size=val_pipeline.epoch_size("Reader"),
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )

    def train_dataloader(self) -> TempDALIGenericIterator:
        return self.train_loader

    def val_dataloader(self) -> TempDALIGenericIterator:
        return self.val_loader
