"""
Dataset Loader for Document Forgery Detection
===============================================
Supports:
  - Loading raw images (authentic + forged)
  - On-the-fly ELA computation
  - Data augmentation via albumentations
  - Train/Val/Test splitting
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.ela import compute_ela


# ─── Augmentation Pipelines ────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(noise_scale_factor=0.1, p=0.3),
        A.CLAHE(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─── Dataset Class ──────────────────────────────────────────────────────────────

class ForgeryDataset(Dataset):
    """
    Expects folder structure:
        root/
          authentic/   ← label 0
          forged/      ← label 1

    Returns (original_tensor, ela_tensor, label) for each sample.
    """

    EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")

    def __init__(self, root_dir: str, transform=None, ela_quality: int = 90,
                 ela_scale: int = 15, use_ela: bool = True):
        """
        Args:
            root_dir   : Dataset root containing 'authentic' and 'forged' sub-dirs.
            transform  : Albumentations transform pipeline.
            ela_quality: JPEG quality for ELA computation.
            ela_scale  : Amplification scale for ELA.
            use_ela    : Whether to return ELA alongside original image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale
        self.use_ela = use_ela

        self.samples = []   # list of (image_path, label)
        self._load_samples()

    def _load_samples(self):
        class_map = {"authentic": 0, "forged": 1}
        for class_name, label in class_map.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[Dataset] Warning: {class_dir} not found, skipping.")
                continue
            for ext in self.EXTENSIONS:
                for path in glob.glob(os.path.join(class_dir, "**", ext), recursive=True):
                    self.samples.append((path, label))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {self.root_dir}")
        authentic_count = sum(1 for _, l in self.samples if l == 0)
        forged_count = sum(1 for _, l in self.samples if l == 1)
        print(f"           Authentic: {authentic_count} | Forged: {forged_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load original image
        image = np.array(Image.open(path).convert("RGB"))

        # Load/compute ELA
        if self.use_ela:
            ela = compute_ela(path, self.ela_quality, self.ela_scale)
        else:
            ela = image.copy()

        # Apply transforms
        if self.transform:
            aug_orig = self.transform(image=image)["image"]
            aug_ela  = self.transform(image=ela)["image"]
        else:
            aug_orig = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            aug_ela  = torch.from_numpy(ela.transpose(2, 0, 1)).float() / 255.0

        return aug_orig, aug_ela, torch.tensor(label, dtype=torch.long)


# ─── DataLoader Factory ─────────────────────────────────────────────────────────

def build_dataloaders(root_dir: str, batch_size: int = 32, image_size: int = 224,
                      val_ratio: float = 0.15, test_ratio: float = 0.10,
                      num_workers: int = 0, use_ela: bool = True):
    """
    Build train / val / test DataLoaders with stratified-like splitting.

    Returns:
        train_loader, val_loader, test_loader
    """
    full_dataset = ForgeryDataset(
        root_dir=root_dir,
        transform=get_train_transforms(image_size),
        use_ela=use_ela,
    )

    n = len(full_dataset)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Override val/test transforms (no augmentation)
    val_ds.dataset.transform  = get_val_transforms(image_size)
    test_ds.dataset.transform = get_val_transforms(image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
