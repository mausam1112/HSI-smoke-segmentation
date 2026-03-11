"""
src/data/dataset.py
───────────────────
PyTorch Dataset for the HSSDataset segmentation benchmark.

Expected on-disk layout
-----------------------
    <dataset_root>/
    ├── <image_dir>/*.npy          (hyperspectral cubes, float32)
    ├── <mask_dir>/*.png           (binary segmentation masks)
    └── split/<data_version>/
        ├── train.txt
        ├── val.txt
        └── test.txt

Each .txt file lists base filenames (no extension), one per line.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class HSSSegmentationDataset(Dataset):
    """
    Loads hyperspectral images (.npy) and binary segmentation masks (.png)
    for the HSSDataset using pre-defined train/val/test split files.

    Parameters
    dataset_root : str
        Root directory of the dataset.
    image_dir : str
        Sub-folder containing the .npy hyperspectral files.
    mask_dir : str
        Sub-folder containing the .png mask files.
    image_size : tuple[int, int]
        Target spatial size (H, W) for the mask resize.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    img_ext : str
        File extension for image files (default ``.npy``).
    mask_ext : str
        File extension for mask files (default ``.png``).
    img_channels : int
        Expected number of spectral channels (used for documentation only).
    data_version : str
        Sub-folder under ``split/`` (e.g. ``"original"``).
    """

    def __init__(
        self,
        dataset_root: str = "./data/HSSDataset/HSS_VOC",
        image_dir: str = "npy_multichannel",
        mask_dir: str = "Mask",
        image_size: tuple[int, int] = (216, 408),
        split: str = "train",
        img_ext: str = ".npy",
        mask_ext: str = ".png",
        img_channels: int = 25,
        mask_channels: int = 1,
        data_version: str = "original",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_dir = self.dataset_root / image_dir
        self.mask_dir = self.dataset_root / mask_dir
        self.image_size = image_size
        self.split = split
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.img_channels = img_channels
        self.mask_channels = mask_channels
        self.data_version = data_version

        self.image_paths: list[Path] = []
        self.mask_paths: list[Path] = []
        self._load_paths()


    def _load_paths(self) -> None:
        split_file = self.dataset_root / "split" / self.data_version / f"{self.split}.txt"
        print(f"[Dataset] Reading {self.split} split from: {split_file}")

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        filenames = split_file.read_text().splitlines()
        filenames = [f.strip() for f in filenames if f.strip()]

        missing_imgs, missing_masks = 0, 0
        for base in filenames:
            img_path = self.image_dir / (base + self.img_ext)
            mask_path = self.mask_dir / (base + self.mask_ext)
            if img_path.exists() and mask_path.exists():
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
            else:
                if not img_path.exists():
                    missing_imgs += 1
                if not mask_path.exists():
                    missing_masks += 1

        if missing_imgs or missing_masks:
            print(
                f"[Dataset] WARNING: {missing_imgs} missing images, "
                f"{missing_masks} missing masks — skipped."
            )
        print(f"[Dataset] {len(self.image_paths)} valid pairs for split='{self.split}'.")


    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load hyperspectral cube: (H+1, W+1, C) → crop → (C, H, W)
        image = np.load(image_path).astype(np.float32)[:-1, :-1, :]
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)

        # Load mask, resize with +1 border then crop to match image
        target_h, target_w = self.image_size
        mask = Image.open(mask_path).convert("L")
        mask = TF.resize(
            mask,
            (target_h + 1, target_w + 1),
            interpolation=TF.InterpolationMode.NEAREST,
        )
        mask = np.array(mask)[:-1, :-1]
        mask = (mask > 0).astype(np.uint8)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return image, mask
