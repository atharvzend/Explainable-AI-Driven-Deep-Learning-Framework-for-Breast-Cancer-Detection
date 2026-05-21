"""
BreakHis Dataset Loader — Combined Magnification Training
Dataset : Breast Cancer Histopathological Database (BreakHis)
Classes : benign (0), malignant (1) — binary
Mags    : 40X, 100X, 200X, 400X — ALL combined into ONE dataset for training
Split   : 80-10-10 stratified image-level (train_test_split with stratify=labels)
           → matches literature (Papers 17, 24, 37) exactly
           → stratify ensures class balance in each split
Augment : H-flip, V-flip, ±15° rotation, brightness/contrast, zoom 0.9–1.1×
Imbalance: Weighted Cross-Entropy Loss
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────
# Set via environment variable BREAKHIS_ROOT (done in Colab notebook)
BREAKHIS_ROOT  = os.environ.get(
    "BREAKHIS_ROOT",
    r"D:/breast cancer project/DataSet 2/dataset_cancer_v1/classificacao_binaria"
)
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]
CLASS_MAP      = {"benign": 0, "malignant": 1}
SEED           = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# COLLECT SAMPLES
# ─────────────────────────────────────────────
def collect_breakhis_samples(magnifications=None, root=None):
    """
    Collects image paths, labels, patient IDs, and magnification tags.

    BreakHis filename: SOB_{B/M}_{type}-{slide}-{patient_id}-{mag}-{frame}.png
    Patient ID = base.split('-')[2]  e.g. '22549AB' or '10926'
    """
    if root is None:
        root = BREAKHIS_ROOT
    if magnifications is None:
        magnifications = MAGNIFICATIONS

    image_paths, labels, patient_ids, mag_tags = [], [], [], []

    for mag in magnifications:
        mag_dir = os.path.join(root, mag)
        for class_name, label in CLASS_MAP.items():
            class_dir = os.path.join(mag_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    continue
                base       = os.path.splitext(fname)[0]
                patient_id = base.split("-")[2]
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label)
                patient_ids.append(patient_id)
                mag_tags.append(mag)

    return image_paths, labels, patient_ids, mag_tags


# ─────────────────────────────────────────────
# PATIENT-LEVEL SPLIT
# ─────────────────────────────────────────────
def split_breakhis(image_paths, labels, patient_ids, mag_tags,
                   train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                   seed=SEED):
    """
    Stratified image-level split → train / val / test (80/10/10).
    Matches literature (Papers 17, 24, 37).
    stratify=labels ensures class balance in each split.

    Returns dicts with keys: image_paths, labels, mag_tags
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Step 1: train vs temp (val + test)
    (img_train, img_temp,
     lbl_train, lbl_temp,
     mag_train, mag_temp) = train_test_split(
        image_paths, labels, mag_tags,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )

    # Step 2: val vs test
    relative_val = val_ratio / (val_ratio + test_ratio)
    (img_val, img_test,
     lbl_val, lbl_test,
     mag_val, mag_test) = train_test_split(
        img_temp, lbl_temp, mag_temp,
        test_size=(1.0 - relative_val),
        stratify=lbl_temp,
        random_state=seed
    )

    train = {"image_paths": img_train, "labels": lbl_train, "mag_tags": mag_train}
    val   = {"image_paths": img_val,   "labels": lbl_val,   "mag_tags": mag_val}
    test  = {"image_paths": img_test,  "labels": lbl_test,  "mag_tags": mag_test}

    return train, val, test


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(size=224, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class BreakHisDataset(Dataset):
    def __init__(self, image_paths, labels, mag_tags, split="train"):
        self.image_paths = image_paths
        self.labels      = labels
        self.mag_tags    = mag_tags
        self.transform   = get_transforms(split)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label, self.mag_tags[idx]


# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────
def compute_class_weights(labels, num_classes=2):
    labels  = np.array(labels)
    total   = len(labels)
    weights = []
    for c in range(num_classes):
        count  = np.sum(labels == c)
        weight = total / (num_classes * count) if count > 0 else 0.0
        weights.append(weight)
    weight_tensor = torch.FloatTensor(weights)
    print(f"  Class weights: {weight_tensor}")
    return weight_tensor


# ─────────────────────────────────────────────
# COLLATE FN
# ─────────────────────────────────────────────
def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    mags   = [b[2] for b in batch]
    return images, labels, mags


# ─────────────────────────────────────────────
# MAIN DATALOADER BUILDER
# ─────────────────────────────────────────────
def get_breakhis_dataloaders(batch_size=16, num_workers=2, seed=SEED):
    print("\n[BreakHis] Loading all magnifications combined...")

    image_paths, labels, patient_ids, mag_tags = collect_breakhis_samples()
    print(f"  Total samples   : {len(image_paths)}")
    print(f"  Unique patients : {len(set(patient_ids))}")
    for cls, idx in CLASS_MAP.items():
        print(f"    {cls}: {labels.count(idx)}")
    for mag in MAGNIFICATIONS:
        print(f"    {mag}: {mag_tags.count(mag)}")

    train_data, val_data, test_data = split_breakhis(
        image_paths, labels, patient_ids, mag_tags, seed=seed)
    print(f"  Train: {len(train_data['labels'])} | "
          f"Val: {len(val_data['labels'])} | "
          f"Test: {len(test_data['labels'])}")

    train_dataset = BreakHisDataset(**train_data, split="train")
    val_dataset   = BreakHisDataset(**val_data,   split="val")
    test_dataset  = BreakHisDataset(**test_data,  split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn)

    per_mag_loaders = {}
    for mag in MAGNIFICATIONS:
        mag_indices = [i for i, m in enumerate(test_data["mag_tags"]) if m == mag]
        if not mag_indices:
            continue
        mag_paths  = [test_data["image_paths"][i] for i in mag_indices]
        mag_labels = [test_data["labels"][i]       for i in mag_indices]
        mag_mags   = [mag] * len(mag_indices)
        mag_dataset = BreakHisDataset(
            image_paths=mag_paths, labels=mag_labels,
            mag_tags=mag_mags, split="test")
        per_mag_loaders[mag] = DataLoader(
            mag_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=True, collate_fn=collate_fn)
        print(f"  Per-mag test [{mag}]: {len(mag_labels)} images")

    class_weights = compute_class_weights(train_data["labels"])

    return train_loader, val_loader, test_loader, per_mag_loaders, class_weights
