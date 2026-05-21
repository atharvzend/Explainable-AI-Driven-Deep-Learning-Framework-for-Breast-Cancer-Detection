"""
Training Script — BreakHis Multiclass (8 tumor subtypes)
=========================================================
Trains 1 model on ALL magnifications combined (40X + 100X + 200X + 400X).
Patient-level split — no data leakage.
Output: 8 classes (adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma,
        mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma)

Usage:
  python train/train_breakhis_multiclass.py --model cnn_vit --epochs 50
  python train/train_breakhis_multiclass.py --model efficientnet_vit --epochs 50
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.breakhis_multiclass_dataset import get_multiclass_dataloaders, CLASS_NAMES
from models.cnn_vit import CNNViT
from models.efficientnet_vit import EfficientNetViT


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR_MULTICLASS",
    r"D:/breast cancer project/DataSet 2/results/breakhis_multiclass/evaluation"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CONFIG = {
    "batch_size"   : 16,
    "num_workers"  : 2,
    "lr_pretrained": 1e-5,
    "lr_new"       : 1e-4,
    "weight_decay" : 1e-4,
    "epochs"       : 50,
    "patience"     : 10,
    "grad_clip"    : 1.0,
}

NUM_CLASSES = 8


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        preds   = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if return_preds:
        return total_loss / total, correct / total, \
               np.array(all_preds), np.array(all_labels)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
def save_confusion_matrix(preds, labels, model_name, split, results_dir):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    short_names = ["Aden", "DucC", "Fibr", "LobC", "MucC", "PapC", "PhyT", "TubA"]
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(short_names, rotation=45)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(short_names)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name} ({split})\nBreakHis Multiclass (8 classes)")
    plt.tight_layout()
    path = os.path.join(results_dir, f"{model_name}_confusion_matrix_{split}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Confusion matrix saved: {path}")


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def train(model_name, epochs, batch_size):
    print(f"\n{'='*60}")
    print(f"  Model   : {model_name.upper()}")
    print(f"  Task    : Multiclass (8 tumor subtypes)")
    print(f"  Data    : BreakHis Combined (40X + 100X + 200X + 400X)")
    print(f"  Device  : {DEVICE}")
    print(f"{'='*60}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, per_mag_loaders, class_weights = \
        get_multiclass_dataloaders(batch_size=batch_size,
                                   num_workers=TRAIN_CONFIG["num_workers"])

    if model_name == "cnn_vit":
        model = CNNViT(num_classes=NUM_CLASSES)
    elif model_name == "efficientnet_vit":
        model = EfficientNetViT(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    if model_name == "cnn_vit":
        vit_params   = list(model.vit_branch.parameters())
        other_params = list(model.cnn_branch.parameters()) + \
                       list(model.fusion.parameters())
    else:
        vit_params   = list(model.vit_branch.parameters())
        other_params = list(model.effnet_branch.parameters()) + \
                       list(model.fusion.parameters())

    optimizer = optim.Adam([
        {"params": vit_params,   "lr": TRAIN_CONFIG["lr_pretrained"]},
        {"params": other_params, "lr": TRAIN_CONFIG["lr_new"]},
    ], weight_decay=TRAIN_CONFIG["weight_decay"])

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler()

    save_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pth")

    best_val_acc   = 0.0
    patience_count = 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "epoch_time_sec": []
    }
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, DEVICE, return_preds=True)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["epoch_time_sec"].append(round(epoch_time, 2))

        print(f"Epoch [{epoch:3d}/{epochs}]  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%  |  "
              f"Time: {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc"        : val_acc,
                "model_name"     : model_name,
                "dataset"        : "BreakHis_Multiclass",
                "num_classes"    : NUM_CLASSES,
            }, save_path)
            save_confusion_matrix(val_preds, val_labels, model_name, "val", RESULTS_DIR)
            print(f"  ✓ Best model saved  (val_acc={val_acc*100:.2f}%)")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= TRAIN_CONFIG["patience"]:
                print(f"\n[Early Stopping] No improvement for "
                      f"{TRAIN_CONFIG['patience']} epochs.")
                break

    total_time = time.time() - total_start

    print(f"\n--- Loading best checkpoint ---")
    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE, return_preds=True)
    save_confusion_matrix(test_preds, test_labels, model_name, "test", RESULTS_DIR)

    print(f"\n[BreakHis Multiclass | {model_name}] TEST RESULTS:")
    print(f"  Test Loss : {test_loss:.4f}")
    print(f"  Test Acc  : {test_acc*100:.2f}%  (all magnifications combined)")
    print(f"  Best Val  : {best_val_acc*100:.2f}%")
    print(f"  Total Time: {total_time/60:.1f} min")

    history["test_loss"]      = round(test_loss, 6)
    history["test_acc"]       = round(test_acc, 6)
    history["best_val_acc"]   = round(best_val_acc, 6)
    history["total_time_sec"] = round(total_time, 2)
    history["total_time_min"] = round(total_time / 60, 2)
    history["epochs_trained"] = len(history["train_loss"])
    history["model"]          = model_name
    history["task"]           = "multiclass_8"
    history["magnifications"] = "40X+100X+200X+400X"

    history_path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")

    return model, per_mag_loaders


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="cnn_vit",
                        choices=["cnn_vit", "efficientnet_vit"])
    parser.add_argument("--epochs",     type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch_size)
