"""
GradientSHAP — BreakHis Dataset
Applies GradientSHAP (Captum) to CNN+ViT and EfficientNet+ViT models.

Output: 3-panel figure per sample
  Original | GradientSHAP Attribution Map | Overlay

Saves to:
  D:/breast cancer project/breakhis_results/xai_results/{task}/{model}/shap/

Usage:
  python xai/shap_breakhis.py --task binary --model cnn_vit
  python xai/shap_breakhis.py --task binary --model efficientnet_vit
  python xai/shap_breakhis.py --task multiclass --model cnn_vit
  python xai/shap_breakhis.py --task multiclass --model efficientnet_vit
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

BINARY_CLASS_NAMES     = ["Benign", "Malignant"]
MULTICLASS_CLASS_NAMES = [
    "Adenosis", "Ductal Carcinoma", "Fibroadenoma", "Lobular Carcinoma",
    "Mucinous Carcinoma", "Papillary Carcinoma", "Phyllodes Tumor", "Tubular Adenoma"
]

SAMPLES_PER_CLASS_BINARY     = 5
SAMPLES_PER_CLASS_MULTICLASS = 3


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def denormalize(tensor):
    img = tensor.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    img = IMAGENET_STD * img + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def get_background_samples(train_loader, n_samples=50, device="cpu"):
    background, collected = [], 0
    for batch in train_loader:
        images = batch[0].to(device)
        for i in range(images.size(0)):
            if collected >= n_samples:
                break
            background.append(images[i])
            collected += 1
        if collected >= n_samples:
            break
    return torch.stack(background)


# ─────────────────────────────────────────────
# MAIN RUN FUNCTION
# ─────────────────────────────────────────────
def run_shap(model, test_loader, train_loader, model_name, class_names, save_dir,
             device, samples_per_class):
    try:
        from captum.attr import GradientShap
    except ImportError:
        print("[SHAP] captum not installed. Run: pip install captum")
        return

    os.makedirs(save_dir, exist_ok=True)
    num_classes  = len(class_names)
    class_counts = {c: 0 for c in range(num_classes)}
    total_target = samples_per_class * num_classes

    model.eval()
    model = model.to(device)

    print("[SHAP] Collecting background samples...")
    background = get_background_samples(train_loader, n_samples=50, device=device)
    print(f"[SHAP] Background shape: {background.shape}")

    explainer = GradientShap(model)
    saved     = 0

    for batch in test_loader:
        images, labels, _ = batch
        images = images.to(device)

        for i in range(images.size(0)):
            if saved >= total_target:
                break

            true_label = labels[i].item()
            if class_counts[true_label] >= samples_per_class:
                continue

            img_tensor = images[i:i+1]

            with torch.no_grad():
                pred_class = model(img_tensor).argmax(dim=1).item()

            attributions = explainer.attribute(
                img_tensor,
                baselines=background[:10],
                target=pred_class,
                n_samples=10,
                stdevs=0.01
            )   # [1, 3, 224, 224]

            attr_np  = attributions.squeeze().permute(1, 2, 0).cpu().numpy()
            attr_map = np.abs(attr_np).sum(axis=2)
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

            img_np  = denormalize(img_tensor)
            img_u8  = (img_np * 255).astype(np.uint8)
            heat_rgb = (cm.get_cmap("jet")(attr_map)[:, :, :3] * 255).astype(np.uint8)
            overlay  = (0.5 * heat_rgb + 0.5 * img_u8).astype(np.uint8)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_np)
            axes[0].set_title(f"Original\nTrue: {class_names[true_label]}", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(attr_map, cmap="RdBu_r", vmin=0, vmax=1)
            axes[1].set_title(f"GradientSHAP\nPred: {class_names[pred_class]}", fontsize=11)
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("GradientSHAP Overlay", fontsize=11)
            axes[2].axis("off")

            class_label = class_names[true_label].replace(" ", "_")
            fname = f"{model_name}_{class_label}_{class_counts[true_label]+1:02d}.png"
            path  = os.path.join(save_dir, fname)
            plt.suptitle(f"Model: {model_name}  |  BreakHis", fontsize=12)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

            class_counts[true_label] += 1
            saved += 1
            print(f"  [{saved}/{total_target}] Saved: {fname}")

        if saved >= total_target:
            break

    print(f"\n[GradientSHAP] Done — {saved} images saved to: {save_dir}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  type=str, required=True, choices=["binary", "multiclass"])
    parser.add_argument("--model", type=str, required=True, choices=["cnn_vit", "efficientnet_vit"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Task: {args.task}  |  Model: {args.model}")

    BASE_RESULTS = r"D:/breast cancer project/breakhis_results"
    SAVE_DIR     = os.path.join(BASE_RESULTS, "xai_results", args.task, args.model, "shap")

    if args.task == "binary":
        from data.breakhis_dataset import get_breakhis_dataloaders
        from models.cnn_vit import CNNViT
        from models.efficientnet_vit import EfficientNetViT

        train_loader, val_loader, test_loader, _, _ = get_breakhis_dataloaders()
        class_names      = BINARY_CLASS_NAMES
        num_classes      = 2
        samples_per_class = SAMPLES_PER_CLASS_BINARY
        ckpt_dir         = os.path.join(BASE_RESULTS, "binary")

    else:
        from data.breakhis_multiclass_dataset import get_multiclass_dataloaders
        from models.cnn_vit import CNNViT
        from models.efficientnet_vit import EfficientNetViT

        train_loader, val_loader, test_loader, _, _ = get_multiclass_dataloaders()
        class_names      = MULTICLASS_CLASS_NAMES
        num_classes      = 8
        samples_per_class = SAMPLES_PER_CLASS_MULTICLASS
        ckpt_dir         = os.path.join(BASE_RESULTS, "multiclass")

    if args.model == "cnn_vit":
        model = CNNViT(num_classes=num_classes)
    else:
        model = EfficientNetViT(num_classes=num_classes)

    ckpt_path  = os.path.join(ckpt_dir, f"{args.model}_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    run_shap(model, test_loader, train_loader, args.model, class_names, SAVE_DIR,
             device, samples_per_class)
