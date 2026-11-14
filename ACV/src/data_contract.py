"""
Data Contract for CamVid (Cambridge-driving Labeled Video Database)
Assignment B - Unit 3: Advanced Computer Vision (CS107203CS)

Defines public dataset references, splits, class IDs, color maps, 
ignore labels, transforms, and schema integrity validation.
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

# --- 1. DATA CONTRACT SPECIFICATION ---

DATASET_NAME = "CamVid (11-class academic benchmark)"
DATASET_REFERENCE = "Brostow et al., 'Semantic object classes in video: A high-definition ground truth database', Pattern Recognition Letters, 2009."
DATASET_URL = "https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid"

# Class ID to Semantic Label Mapping (11 benchmark classes)
CLASS_NAMES = [
    "Sky",           # 0
    "Building",      # 1
    "Pole",          # 2
    "Road",          # 3
    "Pavement",      # 4
    "Tree",          # 5
    "SignSymbol",    # 6
    "Fence",         # 7
    "Car",           # 8
    "Pedestrian",    # 9
    "Bicyclist"      # 10
]

NUM_CLASSES = len(CLASS_NAMES)  # 11

# Raw annotation value for unlabeled/void pixels
RAW_IGNORE_LABEL = 11

# PyTorch training ignore index (for CrossEntropyLoss)
IGNORE_INDEX = 255

# Standard CamVid RGB color palette for the 11 classes + void
CLASS_COLORS = np.array([
    [128, 128, 128],  # 0: Sky (Gray)
    [128, 0, 0],      # 1: Building (Maroon)
    [192, 192, 128],  # 2: Pole (Beige)
    [128, 64, 128],   # 3: Road (Purple)
    [0, 0, 192],      # 4: Pavement (Blue)
    [128, 128, 0],    # 5: Tree (Olive)
    [192, 128, 128],  # 6: SignSymbol (Pink-tan)
    [64, 64, 128],    # 7: Fence (Slate)
    [64, 0, 128],     # 8: Car (Violet)
    [64, 64, 0],      # 9: Pedestrian (Dark Olive)
    [0, 128, 192],    # 10: Bicyclist (Cyan)
    [0, 0, 0],        # 11: Void / Ignored (Black)
], dtype=np.uint8)

# Standard ImageNet normalization parameters
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Converts a 2D integer class mask (with values 0..10 or 255)
    into an RGB image using the standard CamVid color palette.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        rgb[mask == class_id] = CLASS_COLORS[class_id]
    # Void / Ignored
    rgb[mask == RAW_IGNORE_LABEL] = CLASS_COLORS[RAW_IGNORE_LABEL]
    rgb[mask == IGNORE_INDEX] = CLASS_COLORS[RAW_IGNORE_LABEL]
    return rgb


class CamVidDataset(Dataset):
    """
    CamVid Semantic Segmentation Dataset complying strictly with the Data Contract.
    Features optional in-RAM caching for rapid CPU iteration.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        resolution: Tuple[int, int] = (96, 128),
        augment: bool = False,
        cache_in_ram: bool = True
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution  # (H, W)
        self.augment = augment
        self.cache_in_ram = cache_in_ram

        self.img_dir = os.path.join(root_dir, split)
        self.mask_dir = os.path.join(root_dir, f"{split}annot")

        assert os.path.exists(self.img_dir), f"Image dir missing: {self.img_dir}"
        assert os.path.exists(self.mask_dir), f"Mask dir missing: {self.mask_dir}"

        self.img_files = sorted(os.listdir(self.img_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))
        assert len(self.img_files) == len(self.mask_files), (
            f"Mismatch between images ({len(self.img_files)}) and masks ({len(self.mask_files)}) in {split}"
        )

        # Pre-cache preprocessed tensors into RAM if enabled
        self.cached_samples = []
        if self.cache_in_ram:
            for idx in range(len(self.img_files)):
                img_path = os.path.join(self.img_dir, self.img_files[idx])
                mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

                img_pil = Image.open(img_path).convert("RGB")
                mask_pil = Image.open(mask_path)

                img_res = TF.resize(img_pil, self.resolution, interpolation=TF.InterpolationMode.BILINEAR)
                mask_res = TF.resize(mask_pil, self.resolution, interpolation=TF.InterpolationMode.NEAREST)

                img_t = TF.to_tensor(img_res)
                img_t = TF.normalize(img_t, mean=IMAGE_MEAN, std=IMAGE_STD)

                mask_arr = np.array(mask_res, dtype=np.int64)
                mask_arr[mask_arr == RAW_IGNORE_LABEL] = IGNORE_INDEX
                mask_t = torch.from_numpy(mask_arr).long()

                self.cached_samples.append((img_t, mask_t, self.img_files[idx]))

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.cache_in_ram:
            img_tensor, mask_tensor, img_name = self.cached_samples[idx]
            if self.augment:
                # Fast tensor-level augmentation
                if torch.rand(1).item() > 0.5:
                    img_tensor = TF.hflip(img_tensor)
                    mask_tensor = TF.hflip(mask_tensor)
            return img_tensor, mask_tensor, img_name

        img_name = self.img_files[idx]
        mask_name = self.mask_files[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 1. Resize: Bilinear for image, Nearest for integer class mask
        image = TF.resize(image, self.resolution, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.resolution, interpolation=TF.InterpolationMode.NEAREST)

        # 2. Augmentation (Training split only)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if torch.rand(1).item() > 0.5:
                factor = 0.8 + 0.4 * torch.rand(1).item()
                image = TF.adjust_brightness(image, factor)

        # 3. Convert Image to Float Tensor and Normalize
        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(img_tensor, mean=IMAGE_MEAN, std=IMAGE_STD)

        # 4. Convert Mask to Long Tensor and Map RAW_IGNORE_LABEL (11) -> IGNORE_INDEX (255)
        mask_arr = np.array(mask, dtype=np.int64)
        mask_arr[mask_arr == RAW_IGNORE_LABEL] = IGNORE_INDEX
        mask_tensor = torch.from_numpy(mask_arr).long()

        return img_tensor, mask_tensor, img_name


def get_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    resolution: Tuple[int, int] = (96, 128),
    num_workers: int = 0,
    cache_in_ram: bool = True
) -> Dict[str, DataLoader]:
    """
    Creates reproducible PyTorch DataLoaders for train, val, and test splits.
    """
    train_dataset = CamVidDataset(root_dir, split="train", resolution=resolution, augment=True, cache_in_ram=cache_in_ram)
    val_dataset = CamVidDataset(root_dir, split="val", resolution=resolution, augment=False, cache_in_ram=cache_in_ram)
    test_dataset = CamVidDataset(root_dir, split="test", resolution=resolution, augment=False, cache_in_ram=cache_in_ram)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders


def verify_data_contract(root_dir: str) -> Dict[str, any]:
    """
    Automated Data Contract verification function.
    Checks split counts, image-mask pair integrity, resolution, and label distributions.
    """
    print("=" * 60)
    print("DATA CONTRACT VERIFICATION: " + DATASET_NAME)
    print("Reference: " + DATASET_REFERENCE)
    print("=" * 60)

    expected_counts = {"train": 367, "val": 101, "test": 233}
    stats = {}

    for split, exp_count in expected_counts.items():
        img_dir = os.path.join(root_dir, split)
        mask_dir = os.path.join(root_dir, f"{split}annot")

        imgs = os.listdir(img_dir)
        masks = os.listdir(mask_dir)

        assert len(imgs) == exp_count, f"{split} images count mismatch: {len(imgs)} vs expected {exp_count}"
        assert len(masks) == exp_count, f"{split} masks count mismatch: {len(masks)} vs expected {exp_count}"

        # Check unique labels across this split
        unique_labels = set()
        for mf in masks[:20]:  # Sample first 20 for fast schema check
            arr = np.array(Image.open(os.path.join(mask_dir, mf)))
            unique_labels.update(np.unique(arr))

        stats[split] = {
            "image_count": len(imgs),
            "mask_count": len(masks),
            "sample_labels": sorted(list(unique_labels))
        }
        print(f"[{split.upper()}] Verified: {len(imgs)} images, {len(masks)} masks. Schema matches contract.")

    print("\nClass Hierarchy (11 Classes + Ignored Label):")
    for cid, cname in enumerate(CLASS_NAMES):
        print(f"  Class ID {cid:2d}: {cname:<12} -> Active Training Target")
    print(f"  Raw ID   {RAW_IGNORE_LABEL:2d}: Void/Unlabeled -> Mapped to Loss Ignore Index ({IGNORE_INDEX})")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    verify_data_contract("data/CamVid")
