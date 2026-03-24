"""
Data loading and dataset-profile helpers for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.crop import CenterCropTransform





def get_transform_name(transform_type: int) -> str:
    transform_names = {
        0: "center_crop_arr+HFlip (JiT)",
        1: "Resize+RandomCrop",
        2: "RandomResizedCrop+HFlip",
    }
    return transform_names.get(transform_type, "unknown")


def prepare_dataloader(
    data_path,
    image_size,
    batch_size,
    num_workers,
    rank,
    world_size,
    transform_type: int = 0,
    rrc_scale_min: float = 0.8,
    rrc_scale_max: float = 1.0,
):


    if transform_type == 1:
        first_crop_size = 384 if image_size == 256 else int(image_size * 1.5)
        transform = transforms.Compose([
            transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
        ])
    elif transform_type == 2:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(rrc_scale_min, rrc_scale_max),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            CenterCropTransform(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    dataset = datasets.ImageFolder(str(data_path), transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, sampler

