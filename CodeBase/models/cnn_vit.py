"""
CNN + ViT Hybrid Model — BreakHis Dataset (binary classification)
Architecture:
  - CNN Branch  : Custom 4-block CNN → extracts local texture features
  - ViT Branch  : ViT-B/16 (ImageNet-21k pretrained) → extracts global context
  - Fusion      : Concatenate → FC → Classification head
  - Output      : 2 classes (benign=0, malignant=1)
  - Trained separately per magnification: 40X, 100X, 200X, 400X
"""

import torch
import torch.nn as nn
import timm


# ─────────────────────────────────────────────
# CNN BRANCH — 4-block custom CNN
# ─────────────────────────────────────────────
class CNNBranch(nn.Module):
    """
    4-block CNN: Conv → BN → ReLU → MaxPool (×4)
    Input:  [B, 3, 224, 224]
    Output: [B, 512]  (after global average pool)
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 → 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 224 → 112

            # Block 2: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 112 → 56

            # Block 3: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 56 → 28

            # Block 4: 256 → 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28 → 14
        )

        self.pool = nn.AdaptiveAvgPool2d(1)   # [B, 512, 14, 14] → [B, 512, 1, 1]

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)              # [B, 512]


# ─────────────────────────────────────────────
# ViT BRANCH — ViT-B/16 pretrained
# ─────────────────────────────────────────────
class ViTBranch(nn.Module):
    """
    ViT-B/16 loaded from timm (ImageNet-21k pretrained).
    Classification head removed → outputs [CLS] token embedding.
    Output: [B, 768]
    """

    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0        # remove classification head → returns [CLS] token
        )

    def forward(self, x):
        return self.vit(x)       # [B, 768]


# ─────────────────────────────────────────────
# FUSION + CLASSIFICATION HEAD
# ─────────────────────────────────────────────
class CNNViT(nn.Module):
    """
    CNN + ViT Hybrid for BreakHis (binary classification).

    Pipeline:
      image → CNN branch → [B, 512]
      image → ViT branch → [B, 768]
      concat → [B, 1280]
      FC (1280 → 256) → ReLU → Dropout(0.3)
      FC (256 → num_classes)
    """

    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        self.cnn_branch = CNNBranch()
        self.vit_branch = ViTBranch()

        # CNN: 512, ViT: 768 → concat: 1280
        self.fusion = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        cnn_feat = self.cnn_branch(x)    # [B, 512]
        vit_feat = self.vit_branch(x)    # [B, 768]
        combined = torch.cat([cnn_feat, vit_feat], dim=1)  # [B, 1280]
        return self.fusion(combined)     # [B, num_classes]

    def get_cnn_features(self, x):
        """Returns intermediate CNN feature map for Grad-CAM."""
        return self.cnn_branch.features(x)   # [B, 512, 14, 14]

    def get_vit_attention(self, x):
        """Returns ViT attention weights for Attention Rollout."""
        return self.vit_branch.vit.get_intermediate_layers(x, n=1)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = CNNViT(num_classes=2)
    print(model)

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"\nOutput shape: {out.shape}")   # [2, 2]

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable:,}")
