"""
Grad-CAM++ — BreakHis Dataset
Applies Grad-CAM++ to CNN+ViT and EfficientNet+ViT models.

Grad-CAM++ applied to:
  CNN+ViT          → last Conv2d in CNN branch (block 4)
  EfficientNet+ViT → conv_head of EfficientNet-B4

Output: 4-panel figure per sample
  Original | Grad-CAM++ Heatmap | Grad-CAM++ Overlay | ViT Attention Rollout

Saves to:
  D:/breast cancer project/breakhis_results/xai_results/{task}/{model}/gradcam/

Usage:
  python xai/gradcam_breakhis.py --task binary --model cnn_vit
  python xai/gradcam_breakhis.py --task binary --model efficientnet_vit
  python xai/gradcam_breakhis.py --task multiclass --model cnn_vit
  python xai/gradcam_breakhis.py --task multiclass --model efficientnet_vit
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

BINARY_CLASS_NAMES     = ["Benign", "Malignant"]
MULTICLASS_CLASS_NAMES = [
    "Adenosis", "Ductal Carcinoma", "Fibroadenoma", "Lobular Carcinoma",
    "Mucinous Carcinoma", "Papillary Carcinoma", "Phyllodes Tumor", "Tubular Adenoma"
]

# Samples per class for research paper figures
SAMPLES_PER_CLASS_BINARY     = 5   # 5 × 2 = 10 total
SAMPLES_PER_CLASS_MULTICLASS = 3   # 3 × 8 = 24 total


# ─────────────────────────────────────────────
# GRAD-CAM++
# ─────────────────────────────────────────────
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        for m in model.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, image, class_idx=None):
        self.model.eval()
        image = image.requires_grad_(True)

        output = self.model(image)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        grads = self.gradients   # [1, C, H, W]
        acts  = self.activations

        alpha_num   = grads ** 2
        alpha_denom = 2 * grads ** 2 + acts * (grads ** 3).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha       = alpha_num / alpha_denom
        weights     = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam         = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam         = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam         = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx


# ─────────────────────────────────────────────
# ATTENTION ROLLOUT (ViT branch)
# ─────────────────────────────────────────────
class AttentionRollout:
    def __init__(self, vit_model, discard_ratio=0.9):
        self.vit           = vit_model
        self.discard_ratio = discard_ratio
        self.attentions    = []
        self._register_hooks()

    def _register_hooks(self):
        def get_attention(module, input, output):
            self.attentions.append(output.detach())
        for block in self.vit.blocks:
            block.attn.attn_drop.register_forward_hook(get_attention)

    def __call__(self, image):
        self.attentions = []
        self.vit.eval()
        with torch.no_grad():
            self.vit(image)

        result = torch.eye(197)
        for attn in self.attentions:
            attn_mean = attn.mean(dim=1).squeeze(0)   # [197, 197]
            flat      = attn_mean.flatten()
            threshold = flat.quantile(self.discard_ratio)
            attn_mean[attn_mean < threshold] = 0
            attn_mean = attn_mean + torch.eye(197, device=attn_mean.device)
            attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
            result    = torch.matmul(attn_mean, result)

        mask = result[0, 1:].reshape(14, 14).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (224, 224), Image.BILINEAR)) / 255.0
        return mask


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def denormalize(tensor):
    img = tensor.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    img = IMAGENET_STD * img + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def overlay_heatmap(image_np, heatmap, alpha=0.5):
    cmap      = plt.get_cmap("jet")
    heatmap_c = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * heatmap_c + (1 - alpha) * image_np).astype(np.uint8)


def get_target_layer(model):
    if hasattr(model, "cnn_branch"):
        return model.cnn_branch.features[-3], model.vit_branch.vit
    elif hasattr(model, "effnet_branch"):
        return model.effnet_branch.effnet.conv_head, model.vit_branch.vit
    else:
        raise ValueError("Unknown model structure")


# ─────────────────────────────────────────────
# MAIN RUN FUNCTION
# ─────────────────────────────────────────────
def run_gradcam(model, test_loader, model_name, class_names, save_dir, device,
                samples_per_class):
    os.makedirs(save_dir, exist_ok=True)
    num_classes   = len(class_names)
    class_counts  = {c: 0 for c in range(num_classes)}
    total_target  = samples_per_class * num_classes

    target_layer, vit_model = get_target_layer(model)
    cam_extractor  = GradCAMPlusPlus(model, target_layer)
    attn_extractor = AttentionRollout(vit_model)

    model.eval()
    model = model.to(device)
    saved = 0

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
            cam, pred_class = cam_extractor(img_tensor)
            rollout = attn_extractor(img_tensor)
            img_np  = denormalize(img_tensor)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(img_np)
            axes[0].set_title(f"Original\nTrue: {class_names[true_label]}", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title(f"Grad-CAM++\nPred: {class_names[pred_class]}", fontsize=11)
            axes[1].axis("off")

            axes[2].imshow(overlay_heatmap(img_np, cam))
            axes[2].set_title("Grad-CAM++ Overlay", fontsize=11)
            axes[2].axis("off")

            axes[3].imshow(overlay_heatmap(img_np, rollout))
            axes[3].set_title("ViT Attention Rollout", fontsize=11)
            axes[3].axis("off")

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

    print(f"\n[Grad-CAM++] Done — {saved} images saved to: {save_dir}")


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
    SAVE_DIR     = os.path.join(BASE_RESULTS, "xai_results", args.task, args.model, "gradcam")

    if args.task == "binary":
        from data.breakhis_dataset import get_breakhis_dataloaders
        from models.cnn_vit import CNNViT
        from models.efficientnet_vit import EfficientNetViT

        train_loader, val_loader, test_loader, _, _ = get_breakhis_dataloaders()
        class_names      = BINARY_CLASS_NAMES
        num_classes      = 2
        samples_per_class = SAMPLES_PER_CLASS_BINARY
        ckpt_dir         = os.path.join(BASE_RESULTS, "binary")

    else:  # multiclass
        from data.breakhis_multiclass_dataset import get_multiclass_dataloaders
        from models.cnn_vit import CNNViT
        from models.efficientnet_vit import EfficientNetViT

        _, _, test_loader, _, _ = get_multiclass_dataloaders()
        class_names      = MULTICLASS_CLASS_NAMES
        num_classes      = 8
        samples_per_class = SAMPLES_PER_CLASS_MULTICLASS
        ckpt_dir         = os.path.join(BASE_RESULTS, "multiclass")

    # Load model
    if args.model == "cnn_vit":
        model = CNNViT(num_classes=num_classes)
    else:
        model = EfficientNetViT(num_classes=num_classes)

    ckpt_path = os.path.join(ckpt_dir, f"{args.model}_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    run_gradcam(model, test_loader, args.model, class_names, SAVE_DIR, device, samples_per_class)
