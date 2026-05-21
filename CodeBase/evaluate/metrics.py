"""
Evaluation Metrics — BreakHis Combined Model
=============================================
Evaluates one trained model and reports:
  1. Overall metrics  (all 4 mags combined)
  2. Per-magnification metrics (40X / 100X / 200X / 400X)

Usage:
  python evaluate/metrics.py --model cnn_vit
  python evaluate/metrics.py --model efficientnet_vit
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    r"D:/breast cancer project/DataSet 2/results/breakhis/evaluation"
)
CLASS_NAMES = ["benign", "malignant"]


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels, _ in loader:
        images  = images.to(device)
        outputs = model(images)
        probs   = F.softmax(outputs, dim=1).cpu().numpy()
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs, axis=0))


def compute_metrics(preds, labels, probs):
    acc       = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)
    f1        = f1_score(labels, preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    else:
        specificity = 0.0
        sensitivity = float(recall)

    report = classification_report(labels, preds,
                                   target_names=CLASS_NAMES,
                                   zero_division=0, output_dict=True)
    return {
        "accuracy"        : float(acc),
        "precision_macro" : float(precision),
        "recall_macro"    : float(recall),
        "sensitivity"     : sensitivity,
        "specificity"     : specificity,
        "f1_macro"        : float(f1),
        "auc_roc"         : float(auc),
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
    }


def plot_confusion_matrix(cm_arr, model_name, label):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}\n{label}", fontsize=13)
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("+", "").replace("/", "_")
    path = os.path.join(RESULTS_DIR, f"{model_name}_cm_{safe_label}.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


def print_metrics(metrics, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision (mac) : {metrics['precision_macro']*100:.2f}%")
    print(f"  Recall (mac)    : {metrics['recall_macro']*100:.2f}%")
    print(f"  Sensitivity     : {metrics['sensitivity']*100:.2f}%")
    print(f"  Specificity     : {metrics['specificity']*100:.2f}%")
    print(f"  F1-Score (mac)  : {metrics['f1_macro']*100:.2f}%")
    print(f"  AUC-ROC         : {metrics['auc_roc']*100:.2f}%")


def evaluate_model(model, test_loader, per_mag_loaders, model_name, device):
    full_results = {}

    preds, labels, probs = collect_predictions(model, test_loader, device)
    overall = compute_metrics(preds, labels, probs)
    cm = np.array(overall["confusion_matrix"])
    plot_confusion_matrix(cm, model_name, "Overall Combined")
    print_metrics(overall, f"{model_name} | Overall (40X+100X+200X+400X)")
    full_results["overall"] = overall

    per_mag = {}
    from data.breakhis_dataset import MAGNIFICATIONS
    for mag in MAGNIFICATIONS:
        if mag not in per_mag_loaders:
            continue
        preds_m, labels_m, probs_m = collect_predictions(
            model, per_mag_loaders[mag], device)
        mag_metrics = compute_metrics(preds_m, labels_m, probs_m)
        cm_m = np.array(mag_metrics["confusion_matrix"])
        plot_confusion_matrix(cm_m, model_name, f"BreakHis {mag}")
        print_metrics(mag_metrics, f"{model_name} | {mag}")
        per_mag[mag] = mag_metrics

    full_results["per_magnification"] = per_mag

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.json")
    with open(save_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  ✓ Metrics saved: {save_path}")

    return full_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn_vit",
                        choices=["cnn_vit", "efficientnet_vit"])
    args = parser.parse_args()

    from data.breakhis_dataset import get_breakhis_dataloaders
    from models.cnn_vit import CNNViT
    from models.efficientnet_vit import EfficientNetViT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, per_mag_loaders, _ = get_breakhis_dataloaders()

    checkpoint_path = os.path.join(RESULTS_DIR, f"{args.model}_best.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model = CNNViT(num_classes=2) if args.model == "cnn_vit" \
            else EfficientNetViT(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    evaluate_model(model, test_loader, per_mag_loaders, args.model, device)
