"""
Deletion Score + Insertion Score — BreakHis Dataset
Quantitative XAI evaluation for CNN+ViT and EfficientNet+ViT models.

Metrics (no GT mask needed — image-level labels only):
  Deletion Score : Progressively remove top salient pixels → measure accuracy drop
                   Lower AUC = better (model relies on highlighted regions)
  Insertion Score: Progressively reveal top salient pixels from black background
                   Higher AUC = better (highlighted regions sufficient for correct pred)

Saliency map source: Grad-CAM++ (from CNN/EfficientNet branch)

Saves:
  - Per-sample curves as PNG
  - Summary JSON with mean scores per model
  D:/breast cancer project/breakhis_results/xai_results/{task}/{model}/deletion_insertion/

Usage:
  python xai/deletion_insertion_breakhis.py --task binary --model cnn_vit
  python xai/deletion_insertion_breakhis.py --task binary --model efficientnet_vit
  python xai/deletion_insertion_breakhis.py --task multiclass --model cnn_vit
  python xai/deletion_insertion_breakhis.py --task multiclass --model efficientnet_vit
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
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

# Number of test samples to evaluate per class
SAMPLES_PER_CLASS_BINARY     = 5
SAMPLES_PER_CLASS_MULTICLASS = 3

# Number of deletion/insertion steps
STEPS = 10


# ─────────────────────────────────────────────
# GRAD-CAM++ (saliency source)
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
        def fwd(module, input, output):
            self.activations = output.detach()
        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def __call__(self, image, class_idx=None):
        self.model.eval()
        image = image.requires_grad_(True)
        output = self.model(image)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        grads = self.gradients
        acts  = self.activations
        alpha_num   = grads ** 2
        alpha_denom = 2 * grads ** 2 + acts * (grads ** 3).sum(dim=(2,3), keepdim=True) + 1e-8
        alpha   = alpha_num / alpha_denom
        weights = (alpha * F.relu(grads)).sum(dim=(2,3), keepdim=True)
        cam     = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx


def get_target_layer(model):
    if hasattr(model, "cnn_branch"):
        return model.cnn_branch.features[-3]
    elif hasattr(model, "effnet_branch"):
        return model.effnet_branch.effnet.conv_head
    else:
        raise ValueError("Unknown model structure")


# ─────────────────────────────────────────────
# DELETION SCORE
# Mask out top-k salient pixels → measure confidence on predicted class
# Lower AUC = model truly relies on salient region
# ─────────────────────────────────────────────
@torch.no_grad()
def deletion_score(model, image, heatmap, pred_class, device, steps=STEPS):
    """
    image   : [1, 3, 224, 224] tensor (on device)
    heatmap : [224, 224] numpy array [0,1]
    Returns : list of confidences at each step, AUC
    """
    flat_idx  = np.argsort(heatmap.flatten())[::-1]   # most salient first
    h, w      = heatmap.shape
    step_size = len(flat_idx) // steps
    confs     = []

    for step in range(steps + 1):
        img_mod = image.clone()
        if step > 0:
            n_masked  = step * step_size
            mask_idx  = flat_idx[:n_masked]
            row_idx   = mask_idx // w
            col_idx   = mask_idx % w
            img_mod[0, :, row_idx, col_idx] = 0.0   # replace with black

        probs = F.softmax(model(img_mod), dim=1)
        confs.append(probs[0, pred_class].item())

    auc = float(np.trapezoid(confs) / steps)
    return confs, auc


# ─────────────────────────────────────────────
# INSERTION SCORE
# Reveal top-k salient pixels from black background → measure confidence rise
# Higher AUC = salient region is sufficient for correct prediction
# ─────────────────────────────────────────────
@torch.no_grad()
def insertion_score(model, image, heatmap, pred_class, device, steps=STEPS):
    """
    image   : [1, 3, 224, 224] tensor (on device)
    heatmap : [224, 224] numpy array [0,1]
    Returns : list of confidences at each step, AUC
    """
    flat_idx  = np.argsort(heatmap.flatten())[::-1]
    h, w      = heatmap.shape
    step_size = len(flat_idx) // steps
    confs     = []

    baseline  = torch.zeros_like(image)   # black background

    for step in range(steps + 1):
        img_mod = baseline.clone()
        if step > 0:
            n_revealed = step * step_size
            rev_idx    = flat_idx[:n_revealed]
            row_idx    = rev_idx // w
            col_idx    = rev_idx % w
            img_mod[0, :, row_idx, col_idx] = image[0, :, row_idx, col_idx]

        probs = F.softmax(model(img_mod), dim=1)
        confs.append(probs[0, pred_class].item())

    auc = float(np.trapezoid(confs) / steps)
    return confs, auc


# ─────────────────────────────────────────────
# SAVE CURVE FIGURE
# ─────────────────────────────────────────────
def save_curve(del_confs, ins_confs, del_auc, ins_auc,
               class_name, sample_idx, model_name, save_dir):
    x = np.linspace(0, 100, len(del_confs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x, del_confs, 'r-o', markersize=4)
    axes[0].set_title(f"Deletion Score\nAUC = {del_auc:.4f}", fontsize=12)
    axes[0].set_xlabel("% Pixels Removed")
    axes[0].set_ylabel("Confidence")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, ins_confs, 'b-o', markersize=4)
    axes[1].set_title(f"Insertion Score\nAUC = {ins_auc:.4f}", fontsize=12)
    axes[1].set_xlabel("% Pixels Revealed")
    axes[1].set_ylabel("Confidence")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Model: {model_name}  |  Class: {class_name}  |  Sample #{sample_idx}",
                 fontsize=12)
    plt.tight_layout()

    fname = f"{model_name}_{class_name.replace(' ','_')}_{sample_idx:02d}.png"
    path  = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────
# MAIN RUN FUNCTION
# ─────────────────────────────────────────────
def run_deletion_insertion(model, test_loader, model_name, class_names,
                           save_dir, device, samples_per_class):
    os.makedirs(save_dir, exist_ok=True)
    num_classes  = len(class_names)
    class_counts = {c: 0 for c in range(num_classes)}
    total_target = samples_per_class * num_classes

    target_layer = get_target_layer(model)
    cam_extractor = GradCAMPlusPlus(model, target_layer)

    model.eval()
    model = model.to(device)

    all_del_aucs = []
    all_ins_aucs = []
    per_class_del = {c: [] for c in range(num_classes)}
    per_class_ins = {c: [] for c in range(num_classes)}
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

            # Get saliency map via Grad-CAM++
            cam, pred_class = cam_extractor(img_tensor)

            # Deletion Score
            del_confs, del_auc = deletion_score(
                model, img_tensor, cam, pred_class, device
            )

            # Insertion Score
            ins_confs, ins_auc = insertion_score(
                model, img_tensor, cam, pred_class, device
            )

            # Save curve figure
            save_curve(
                del_confs, ins_confs, del_auc, ins_auc,
                class_names[true_label],
                class_counts[true_label] + 1,
                model_name, save_dir
            )

            all_del_aucs.append(del_auc)
            all_ins_aucs.append(ins_auc)
            per_class_del[true_label].append(del_auc)
            per_class_ins[true_label].append(ins_auc)

            class_counts[true_label] += 1
            saved += 1
            print(f"  [{saved}/{total_target}] {class_names[true_label]} "
                  f"| Del AUC: {del_auc:.4f}  Ins AUC: {ins_auc:.4f}")

        if saved >= total_target:
            break

    # ── Summary ──
    summary = {
        "model"              : model_name,
        "deletion_auc_mean"  : float(np.mean(all_del_aucs)),
        "deletion_auc_std"   : float(np.std(all_del_aucs)),
        "insertion_auc_mean" : float(np.mean(all_ins_aucs)),
        "insertion_auc_std"  : float(np.std(all_ins_aucs)),
        "per_class"          : {}
    }
    for c in range(num_classes):
        if per_class_del[c]:
            summary["per_class"][class_names[c]] = {
                "deletion_auc_mean"  : float(np.mean(per_class_del[c])),
                "insertion_auc_mean" : float(np.mean(per_class_ins[c])),
            }

    json_path = os.path.join(save_dir, f"{model_name}_del_ins_scores.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Model: {model_name}")
    print(f"  Deletion  AUC (mean): {summary['deletion_auc_mean']:.4f} "
          f"± {summary['deletion_auc_std']:.4f}")
    print(f"  Insertion AUC (mean): {summary['insertion_auc_mean']:.4f} "
          f"± {summary['insertion_auc_std']:.4f}")
    print(f"  Saved: {json_path}")
    print(f"{'='*55}")

    return summary


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  type=str, required=True,
                        choices=["binary", "multiclass"])
    parser.add_argument("--model", type=str, required=True,
                        choices=["cnn_vit", "efficientnet_vit"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Task: {args.task}  |  Model: {args.model}")

    BASE_RESULTS = r"D:/breast cancer project/breakhis_results"
    SAVE_DIR     = os.path.join(BASE_RESULTS, "xai_results", args.task,
                                args.model, "deletion_insertion")

    if args.task == "binary":
        from data.breakhis_dataset import get_breakhis_dataloaders
        from models.cnn_vit import CNNViT
        from models.efficientnet_vit import EfficientNetViT

        train_loader, val_loader, test_loader, _, _ = get_breakhis_dataloaders()
        class_names       = BINARY_CLASS_NAMES
        num_classes       = 2
        samples_per_class = SAMPLES_PER_CLASS_BINARY
        ckpt_dir          = os.path.join(BASE_RESULTS, "binary")

    else:
        from data.breakhis_multiclass_dataset import get_multiclass_dataloaders
        from models.cnn_vit import CNNViT
        from models.efficientnet_vit import EfficientNetViT

        train_loader, val_loader, test_loader, _, _ = get_multiclass_dataloaders()
        class_names       = MULTICLASS_CLASS_NAMES
        num_classes       = 8
        samples_per_class = SAMPLES_PER_CLASS_MULTICLASS
        ckpt_dir          = os.path.join(BASE_RESULTS, "multiclass")

    if args.model == "cnn_vit":
        model = CNNViT(num_classes=num_classes)
    else:
        model = EfficientNetViT(num_classes=num_classes)

    ckpt_path  = os.path.join(ckpt_dir, f"{args.model}_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    run_deletion_insertion(model, test_loader, args.model, class_names,
                           SAVE_DIR, device, samples_per_class)
