"""
EfficientNet-B4 + ViT Hybrid Model — BreakHis Dataset (binary classification)
Architecture:
  - EfficientNet Branch : EfficientNet-B4 (ImageNet pretrained) → local features
  - ViT Branch          : ViT-B/16 (ImageNet-21k pretrained) → global context
  - Fusion              : Concatenate → FC → Classification head
  - Output              : 2 classes (benign=0, malignant=1)
  - Trained separately per magnification: 40X, 100X, 200X, 400X
"""

import torch
import torch.nn as nn
import timm


# ─────────────────────────────────────────────
# EFFICIENTNET BRANCH
# ─────────────────────────────────────────────
class EfficientNetBranch(nn.Module):
    """
    EfficientNet-B4 feature extractor (classifier head removed).
    Input:  [B, 3, 224, 224]
    Output: [B, 1792]  (EfficientNet-B4 feature dimension)
    """

    def __init__(self):
        super().__init__()
        self.effnet = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=0        # remove classifier head → returns pooled features
        )
        self.out_features = self.effnet.num_features   # 1792 for B4

    def forward(self, x):
        return self.effnet(x)    # [B, 1792]


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
            num_classes=0        # remove classification head
        )

    def forward(self, x):
        return self.vit(x)       # [B, 768]


# ─────────────────────────────────────────────
# FUSION + CLASSIFICATION HEAD
# ─────────────────────────────────────────────
class EfficientNetViT(nn.Module):
    """
    EfficientNet-B4 + ViT Hybrid for BreakHis (binary classification).

    Pipeline:
      image → EfficientNet-B4 → [B, 1792]
      image → ViT-B/16        → [B, 768]
      concat → [B, 2560]
      FC (2560 → 512) → ReLU → Dropout(0.3)
      FC (512 → 256)  → ReLU → Dropout(0.3)
      FC (256 → num_classes)
    """

    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        self.effnet_branch = EfficientNetBranch()
        self.vit_branch    = ViTBranch()

        effnet_dim = self.effnet_branch.out_features   # 1792
        vit_dim    = 768
        fused_dim  = effnet_dim + vit_dim              # 2560

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        effnet_feat = self.effnet_branch(x)               # [B, 1792]
        vit_feat    = self.vit_branch(x)                  # [B, 768]
        combined    = torch.cat([effnet_feat, vit_feat], dim=1)  # [B, 2560]
        return self.fusion(combined)                      # [B, num_classes]

    def get_effnet_features(self, x):
        """
        Returns EfficientNet feature maps BEFORE global pooling — for Grad-CAM.
        Shape: [B, 1792, H, W]  (~7x7 for 224 input)
        """
        return self.effnet_branch.effnet.forward_features(x)

    def get_vit_attention(self, x):
        """Returns ViT intermediate layer outputs for Attention Rollout."""
        return self.vit_branch.vit.get_intermediate_layers(x, n=1)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = EfficientNetViT(num_classes=2)
    print(model)

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"\nOutput shape: {out.shape}")   # [2, 2]

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable:,}")
