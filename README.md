# Human-Guided Augmentation Filtering for Data-Scarce Image Classifiers

**Gains and Robustness Limits**

This repository contains the research code and paper for a study on whether human-reviewed data augmentation improves image classifier performance and adversarial robustness compared to unfiltered augmentation and automated (confidence-based) filtering.

Accepted at IEEE DSC (Dependable, Secure, and Trustworthy Computing).

---

## Overview

Image classifiers can latch onto spurious background cues instead of the features that actually define a class; this is a known failure mode called shortcut learning, and it's a separate problem from adversarial vulnerability, though both stem from reliance on brittle features. This project asks a simple question: if you augment training data and then filter out the augmentations that no longer look like their assigned class, does it matter whether a human or a model does the filtering?

The framework has four stages:

1. **Foreground-focused preprocessing** – reduce background influence before augmenting (intensity-based masking for MNIST, DINO ViT-S/8 attention maps for CIFAR-10 and STL-10).
2. **Candidate augmentation** – generate multiple augmented views per source image (crops, flips, rotation, color jitter, erasing, etc.).
3. **Quality filtering** – three parallel conditions built from the same candidate pool:
   - **Baseline** – no filtering, all candidates kept.
   - **Auto** – a confidence-based classifier auto-approves candidates (cross-fitted to avoid circularity).
   - **Human** – a human reviewer approves/rejects each candidate through an interactive widget.
4. **Dual-objective training** – a shared encoder feeds both a classification head and a reconstruction decoder, so the model is pushed to preserve source-image identity across augmented views.

All three conditions are evaluated on clean test accuracy and under white-box FGSM and PGD adversarial attacks, on MNIST, CIFAR-10, and STL-10, plus four pretrained transfer-learning backbones (MobileNetV2, ResNet50, VGG16, VGG19) on CIFAR-10 and STL-10.

**Headline finding:** human filtering wins on clean accuracy everywhere, but its adversarial robustness advantage is not consistent; automated filtering matches or beats it under FGSM on the natural-image datasets. Human curation is a data quality improvement, not a substitute for adversarial training.

---

## Repository Structure

```
.
├── MNIST_FINAL.ipynb        # Baseline vs. Human-in-the-Loop on MNIST
├── CIFAR-10_FINAL.ipynb     # Baseline vs. Auto vs. Human on CIFAR-10 + transfer learning
├── STL-10_FINAL.ipynb       # Baseline vs. Auto vs. Human on STL-10 + transfer learning
├── paper/                   # IEEE DSC paper (LaTeX/PDF)
└── README.md
```

Each notebook is self-contained: run top-to-bottom for a full fresh run, or use the built-in shortcut cells to load cached segmentation arrays, candidates, or trained models from Google Drive and skip the slow steps.

---

## Method Details

### Foreground preprocessing
- **MNIST**: deterministic intensity-based masking (a natural-image segmenter would be a domain mismatch for handwritten digits).
- **CIFAR-10 / STL-10**: a frozen, pretrained DINO ViT-S/8 encoder's attention maps are used to localize the foreground. Rather than hard-cutting the background, it's blurred (σ = 2 for CIFAR-10, σ = 3 for STL-10) so the model still sees the whole image but with reduced background detail.

### Candidate augmentation
- MNIST: 2 augmented views per source image (crop + pad, brightness jitter, random erasing). No flipping, since it isn't label-preserving for digits.
- CIFAR-10 / STL-10: 3 augmented views per source image (flip, rotation ±30°, aggressive crop, brightness/color jitter, grayscale conversion, random erasing).

### Filtering conditions
- **Auto**: an independently cross-fitted classifier accepts a candidate if it predicts the correct class with confidence ≥ 0.70.
- **Human**: a reviewer approves a candidate only if (1) it's still consistent with its assigned class, (2) the class-defining foreground structure is still visible, and (3) the transformation hasn't introduced severe distortion or artifacts.
- All three conditions are compared on **class- and sample-matched subsets** so differences in outcome come from candidate quality, not candidate count.

### Model architecture
Dual-head CNN: shared convolutional encoder → 128-dim latent vector → (a) classification head, (b) reconstruction decoder that reconstructs the foreground-focused source image. The reconstruction loss (weighted by β = 0.5) encourages different augmented views of the same source to stay identifiable as that source.

| Dataset | Input size | Encoder blocks | Params |
|---|---|---|---|
| MNIST | 28×28×1 | 2 | 939,275 |
| CIFAR-10 | 32×32×3 | 2 | 1,188,109 |
| STL-10 | 96×96×3 | 3 | 5,293,069 |

Training: 40 epochs, batch size 64, 300 steps/epoch, Adam optimizer. MNIST uses a fixed LR of 1e-3; CIFAR-10 and STL-10 use a 5-epoch linear warmup to 1e-3 followed by cosine decay.

### Adversarial evaluation
White-box, untargeted **FGSM** (single-step) and **PGD** (40 iterations, step size ε/10, random init) attacks in the classifier-input space, swept across a range of ε values. MNIST: ε ∈ {0, 0.05, …, 0.50}. CIFAR-10 / STL-10: ε ∈ {0, 0.005, …, 0.30}.

---

## Results

### Clean test accuracy

| Dataset | Baseline | Auto | Human | Human − Baseline |
|---|---|---|---|---|
| MNIST | 98.60% | 98.35% | 98.80% | +0.20 pp |
| CIFAR-10 | 75.80% | 82.30% | 82.80% | +7.00 pp |
| STL-10 | 80.50% | 81.80% | 84.13% | +3.63 pp |

Human filtering wins clean accuracy on all three datasets with the custom architecture, though the improved classification doesn't correspond to a lower reconstruction error; reconstruction MSE is actually highest for Human in every case.

### Adversarial robustness (custom architecture)
- **MNIST**: Auto is stronger at low FGSM perturbations; Human overtakes it at moderate-to-high ε (e.g. 32.05% vs. 15.25% at ε = 0.20). Under PGD, all conditions degrade sharply.
- **CIFAR-10**: Auto generally leads under both FGSM and PGD at low-to-moderate ε (e.g. 41.85% vs. 38.90% for Human at ε = 0.010). All conditions collapse under PGD by ε ≥ 0.020.
- **STL-10**: Auto leads under FGSM at low ε (28.80% vs. 24.60% for Human at ε = 0.005). PGD differences are small at low ε and all models collapse at higher perturbations.

### Transfer learning (clean accuracy, Baseline vs. Human)

| Architecture | CIFAR-10 Baseline | CIFAR-10 Human | STL-10 Baseline | STL-10 Human |
|---|---|---|---|---|
| MobileNetV2 | 51.70% | 59.90% | 72.10% | 77.85% |
| ResNet50 | 52.10% | 52.90% | 83.65% | 86.00% |
| VGG16 | 65.35% | 56.55% | 72.05% | 78.95% |
| VGG19 | 63.05% | 52.85% | 74.50% | 78.95% |

Human filtering helps consistently on STL-10 across all four backbones, but *hurts* VGG16/VGG19 on CIFAR-10 — likely because those architectures lack residual connections and are more sensitive to augmentation diversity at low (32×32) resolution.

---

## Requirements

These notebooks are designed to run on **Google Colab** with a GPU runtime (T4 or better) and access to Google Drive for caching intermediate artifacts (segmented arrays, candidate pools, trained models).

Key dependencies (installed within the notebooks):
- `tensorflow` 2.20 / `keras` 3.x
- `torch`, `torchvision` (for DeepLabV3 in the MNIST notebook, DINO in CIFAR-10/STL-10)
- `numpy`, `scipy`, `matplotlib`, `Pillow`
- `ipywidgets` (for the interactive human-review widget)
- `scikit-learn` (classification reports)
- `tensorflow-datasets` (STL-10 notebook)

## How to Run

Each notebook follows the same general flow:

1. **Shared setup** – imports, hyperparameters, seed fixing.
2. **Segmentation** – run DINO/DeepLabV3 over the dataset, or load pre-cached segmented arrays from Drive via the shortcut cell.
3. **Part 1: Baseline model** – trained on unfiltered on-the-fly augmentation.
4. **Part 2: Human-in-the-Loop** – generate candidates, review them via the interactive widget (or load a previously-approved set), then train.
5. **Part 3: Auto-Filtered model** (CIFAR-10 / STL-10 only) – the baseline model's own confidence scores auto-approve candidates.
6. **Part 4: Adversarial evaluation** – FGSM and PGD sweeps across all trained models.
7. **Transfer learning** – repeat Baseline vs. Human (vs. Auto for the primary comparison) with MobileNetV2, ResNet50, VGG16, and VGG19 backbones (CIFAR-10 and STL-10 only).
8. **Checkpointing** – save/load trained models and result metrics to/from Drive so a fresh session can restore everything without retraining.

Every notebook includes shortcut cells for skipping the slow steps (segmentation, candidate review, training) by loading cached results from Drive instead.

---

## Authors

- **Miles Ehrlich** 
- **Mohamed Rahouti**
- **Kamil Rampatsingh**
- **Kaiqi Xiong**


## Limitations

As noted in the paper: results are based on ten-class datasets, a single human reviewer, and white-box attacks only. Future work includes larger and fine-grained datasets, multiple reviewers, black-box/transfer attacks, and vision-language models for scalable quality control.
