# Explainable AI Driven Deep Learning Framework for Breast Cancer Detection
Project on deep learning using the BreakHis dataset for binary and multiclass breast cancer detection compared with current studies.

**Published in:** IOSR Journal of Computer Engineering | Volume 28, Issue 3 | May–June 2026
**DOI:** https://doi.org/10.9790/0661-2803012345
**Paper:** https://iosrjournals.org/iosr-jce/papers/Vol28-issue3/Ser-1/C2803012345.pdf

---

## About the Project

Breast cancer is the most common cancer in women worldwide. Doctors currently diagnose it by manually examining microscopic tissue images — a slow and error-prone process.

We built an AI system that:
- Detects whether tissue is cancerous or not with **99.49% accuracy**
- Identifies the specific type of cancer from 8 possible types with **94.06% accuracy**
- Explains its decisions visually so doctors can trust and verify the results

---

## Results

| Model | Task | Accuracy | AUC-ROC |
|-------|------|----------|---------|
| EfficientNet-B4 + ViT | Binary | 99.49% | 99.95% |
| CNN + ViT | Binary | 98.86% | 99.92% |
| CNN + ViT | 8-Class | 94.06% | 99.68% |
| EfficientNet-B4 + ViT | 8-Class | 93.55% | 99.60% |

---

## Tech Stack

- Python 3.10+
- PyTorch 2.x
- EfficientNet-B4, Vision Transformer (ViT-B/16)
- Grad-CAM++, GradientSHAP, LIME
- Google Colab (NVIDIA Tesla T4 GPU)

---

## Team

- Atharv Dinkar Zend
- Hariom Vidyanand Yadav
- Siddhi Narayan Pawar
- Tejas Navnath Giram

**Guide:** Dr. Rahul Chakre
**Institution:** JSPM University Pune
