"""
LIME — BreakHis Dataset
Applies LIME (Local Interpretable Model-Agnostic Explanations) to
CNN+ViT and EfficientNet+ViT models.

Segmentation: SLIC superpixels (appropriate for histopathology images)

Output: 3-panel figure per sample
  Original | LIME Mask | LIME Overlay

Saves to:
  D:/breast cancer project/breakhis_results/xai_results/{task}/{model}/lime/

Usage:
  python xai/lime_breakhis.py --task binary --model cnn_vit
  python xai/lime_breakhis.py --task binary --model efficientnet_vit
  python xai/lime_breakhis.py --task multiclass --model cnn_vit
  python xai/lime_breakhis.py --task multiclass --model efficientnet_vit
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BINARY_CLASS_NAMES     = ["Benign", "Malignant"]
MULTICLASS_CLASS_NAMES = [
    "Adenosis", "Ductal Carcinoma", "Fibroadenoma", "Lobular Carcinoma",
    "Mucinous Carcinoma", "Papillary Carcinoma", "Phyllodes Tumor", "Tubular Adenoma"
]

SAMPLES_PER_CLASS_BINARY     = 5
SAMPLES_PER_CLASS_MULTICLASS = 3

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def denormalize(tensor):
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    img  = tensor.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    img  = std * img + mean
    return np.clip(img, 0, 1)


# ─────────────────────────────────────────────
# MAIN RUN FUNCTION
# ─────────────────────────────────────────────
def run_lime(model, test_loader, model_name, class_names, save_dir, device,
             samples_per_class, num_superpixels=75, num_samples=1000):
    try:
        from lime import lime_image
        from skimage.segmentation import slic, mark_boundaries
    except ImportError:
        print("[LIME] Required: pip install lime scikit-image")
        return

    os.makedirs(save_dir, exist_ok=True)
    num_classes  = len(class_names)
    class_counts = {c: 0 for c in range(num_classes)}
    total_target = samples_per_class * num_classes

    model.eval()
    model = model.to(device)

    # ── Prediction function for LIME ──
    def predict_fn(images_np):
        """images_np: [N, H, W, 3] uint8 → returns [N, num_classes] probs"""
        from PIL import Image as PILImage
        batch = []
        for img in images_np:
            img_u8 = img.astype(np.uint8) if img.max() > 1.0 else (img * 255).astype(np.uint8)
            tensor = _to_tensor(PILImage.fromarray(img_u8))
            batch.append(tensor)
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            probs = F.softmax(model(batch_tensor), dim=1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()
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
            img_np     = (denormalize(img_tensor) * 255).astype(np.uint8)

            with torch.no_grad():
                pred_class = model(img_tensor).argmax(dim=1).item()

            explanation = explainer.explain_instance(
                img_np,
                predict_fn,
                top_labels=num_classes,
                hide_color=0,
                num_samples=num_samples,
                segmentation_fn=lambda x: slic(
                    x, n_segments=num_superpixels,
                    compactness=10, sigma=1, channel_axis=-1
                )
            )

            temp_img, lime_mask = explanation.get_image_and_mask(
                pred_class,
                positive_only=True,
                num_features=10,
                hide_rest=False
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_np)
            axes[0].set_title(f"Original\nTrue: {class_names[true_label]}", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(lime_mask, cmap="RdGy", vmin=-1, vmax=1)
            axes[1].set_title(f"LIME Mask\nPred: {class_names[pred_class]}", fontsize=11)
            axes[1].axis("off")

            overlay = mark_boundaries(temp_img / 255.0, lime_mask)
            axes[2].imshow(overlay)
            axes[2].set_title("LIME Overlay", fontsize=11)
            axes[2].axis("off")

            class_label = class_names[true_label].replace(" ", "_")
            fname = f"{model_name}_{class_label}_{class_counts[true_label]+1:02d}.png"
            path  = os.path.join(save_dir, fname)
            plt.suptitle(
                f"Model: {model_name}  |  BreakHis\n"
                f"SLIC superpixels={num_superpixels}  LIME samples={num_samples}",
                fontsize=11
            )
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

            class_counts[true_label] += 1
            saved += 1
            print(f"  [{saved}/{total_target}] Saved: {fname}")

        if saved >= total_target:
            break

    print(f"\n[LIME] Done — {saved} images saved to: {save_dir}")


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
    SAVE_DIR     = os.path.join(BASE_RESULTS, "xai_results", args.task, args.model, "lime")

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

    run_lime(model, test_loader, args.model, class_names, SAVE_DIR, device, samples_per_class)
