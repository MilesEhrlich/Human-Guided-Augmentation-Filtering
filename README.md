# Human-Guided Augmentation Filtering for Data-Scarce Image Classifiers: Gains and Robustness Limits

Research code for a study on whether human-reviewed data augmentation improves image classifier performance and adversarial robustness compared to unfiltered augmentation and automated, confidence-based filtering.

Accepted at the **IEEE Conference on Dependable and Secure Computing (DSC 2026)**, New York, NY, October 9 to 11, 2026.

## Overview

Image classifiers can latch onto spurious background cues instead of the features that actually define a class. This is known as shortcut learning. It is a separate problem from adversarial vulnerability, though both involve reliance on brittle features. This project asks a simple question: if you augment training data and then filter out the augmentations that no longer look like their assigned class, does it matter whether a human or a model does the filtering?

The framework has four stages:

1. **Foreground-focused preprocessing:** reduce background influence before augmenting (intensity-based masking for MNIST; DINO ViT-S/8 attention maps for CIFAR-10 and STL-10).
2. **Candidate augmentation:** generate multiple augmented views per source image (crops, flips, rotation, color jitter, erasing, and so on).
3. **Quality filtering:** three conditions built from the same candidate pool:
   - **Baseline:** no filtering; all candidates kept.
   - **Auto:** an independent, cross-fitted classifier accepts candidates it predicts as the correct class with confidence of at least 0.70.
   - **Human:** a human reviewer approves or rejects each candidate through an interactive widget.
4. **Dual-objective training:** a shared encoder feeds both a classification head and a reconstruction decoder, so the model is pushed to preserve source-image identity across augmented views.

All three conditions are evaluated on clean test accuracy and under white-box FGSM and PGD attacks on MNIST, CIFAR-10, and STL-10. We also compare Baseline and Human with four pretrained transfer-learning encoders (MobileNetV2, ResNet50, VGG16, VGG19) on CIFAR-10 and STL-10.

**Headline finding:** with the custom architecture, human filtering gives the highest clean accuracy on all three datasets, but its adversarial robustness advantage is not consistent. Automated filtering matches or beats it under FGSM on CIFAR-10 and STL-10, and all conditions collapse under stronger PGD attacks. Human curation improves data quality; it is not a substitute for adversarial training.

## Repository Structure

```
.
├── 01_mnist.ipynb       # Baseline vs. Auto vs. Human on MNIST
├── 02_cifar10.ipynb     # Baseline vs. Auto vs. Human on CIFAR-10, plus transfer learning
├── 03_stl10.ipynb       # Baseline vs. Auto vs. Human on STL-10, plus transfer learning
├── paper/               # Accepted manuscript (see note below)
├── LICENSE
└── README.md
```

Each notebook is self-contained. Run it top to bottom for a full fresh run, or use the built-in shortcut cells to load cached foreground arrays, candidate pools, or trained models from Google Drive and skip the slow steps.

## Method Details

### Foreground preprocessing

- **MNIST:** deterministic intensity-based masking. A natural-image segmenter would be a domain mismatch for handwritten digits.
- **CIFAR-10 / STL-10:** attention maps from a frozen, pretrained DINO ViT-S/8 encoder localize the foreground (attention averaged across heads, thresholded at 0.1). Rather than removing the background, we blur it (σ = 2 for CIFAR-10, σ = 3 for STL-10), so the model still sees the whole image with reduced background detail.

### Candidate augmentation

- **MNIST:** 2 augmented views per source image (crop with reflection padding, brightness variation, random erasing). No flipping, since it isn't label-preserving for digits.
- **CIFAR-10 / STL-10:** 3 augmented views per source image (horizontal flip, rotation within ±30°, stronger cropping, per-channel color jitter, grayscale conversion, random erasing).

Augmentation parameters are sampled once to build a single shared candidate pool, so differences between conditions come from candidate selection, not from different random transformations.

### Filtering conditions

- **Auto:** an auxiliary classifier accepts a candidate if it predicts the assigned class with confidence ≥ 0.70. Five-fold cross-fitting ensures each candidate is scored by a model that never trained on it. The auxiliary models are discarded after filtering and are not used to initialize the final models.
- **Human:** a reviewer approves a candidate only if (1) it is still consistent with its assigned class, (2) the class-defining foreground structure is still visible, and (3) the transformation hasn't introduced severe cropping, erasing, color distortion, or artifacts.

The primary comparison uses class- and sample-matched subsets across all three conditions, so differences in outcome reflect candidate quality, not candidate count.

### Model architecture

Dual-head CNN: shared convolutional encoder → 128-dim latent vector → (a) classification head and (b) reconstruction decoder that reconstructs the foreground-focused source image. The total loss is L = L_cls + β·L_rec with β = 0.5, selected on a held-out validation set.

| Dataset  | Input size | Encoder blocks | Trainable params |
| -------- | ---------- | -------------- | ---------------- |
| MNIST    | 28×28×1    | 2              | 939,275          |
| CIFAR-10 | 32×32×3    | 2              | 1,188,109        |
| STL-10   | 96×96×3    | 3              | 5,293,069        |

**Training:** 40 epochs, batch size 64, 300 steps per epoch, Adam optimizer. MNIST uses a fixed learning rate of 1e-3; CIFAR-10 and STL-10 use a 5-epoch linear warmup to 1e-3 followed by cosine decay. All experiments use five random seeds (0 to 4), reported as mean ± standard deviation.

### Adversarial evaluation

White-box, untargeted FGSM and PGD (40 iterations, step size ε/10, random initialization, 10 restarts) in the classifier-input space.

- MNIST: ε ∈ {0, 0.05, …, 0.50}
- CIFAR-10 / STL-10: ε ∈ {0, 0.005, …, 0.30}

Attacks perturb the foreground-focused classifier inputs. The DINO masking is a cached, offline preprocessing step, so these results do not claim end-to-end robustness against perturbations of the raw images.

## Results

### Clean test accuracy (custom architecture)

| Dataset  | Baseline | Auto   | Human  | Human − Baseline |
| -------- | -------- | ------ | ------ | ---------------- |
| MNIST    | 98.60%   | 98.35% | 98.80% | +0.20 pp         |
| CIFAR-10 | 75.80%   | 82.30% | 82.80% | +7.00 pp         |
| STL-10   | 80.50%   | 81.80% | 84.13% | +3.63 pp         |

Better classification does not necessarily mean better reconstruction: Human has the highest reconstruction MSE on MNIST and STL-10.

### Adversarial robustness (custom architecture)

- **MNIST:** Auto is stronger at low FGSM perturbations; Human overtakes it at moderate-to-high ε (32.05% vs. 15.25% at ε = 0.20). Under PGD, all conditions degrade sharply.
- **CIFAR-10:** Auto generally leads at low-to-moderate ε (41.85% vs. 38.90% for Human and 25.30% for Baseline under FGSM at ε = 0.010). All conditions collapse under PGD at ε ≥ 0.020.
- **STL-10:** Auto leads under FGSM at low ε (28.80% vs. 24.60% for Human at ε = 0.005). PGD differences are small at low ε, and all models collapse at larger perturbations.

### Transfer learning (clean accuracy, Baseline vs. Human)

| Encoder     | CIFAR-10 Baseline | CIFAR-10 Human | STL-10 Baseline | STL-10 Human |
| ----------- | ----------------- | -------------- | --------------- | ------------ |
| MobileNetV2 | 51.70%            | 59.90%         | 72.10%          | 77.85%       |
| ResNet50    | 52.10%            | 52.90%         | 83.65%          | 86.00%       |
| VGG16       | 65.35%            | 56.55%         | 72.05%          | 78.95%       |
| VGG19       | 63.05%            | 52.85%         | 74.50%          | 78.95%       |

Human filtering helps all four encoders on STL-10 but hurts VGG16 and VGG19 on CIFAR-10. One likely reason is that VGG lacks residual connections, making it more sensitive to augmentation-driven distribution shift at CIFAR-10's 32×32 resolution.

## Requirements

The notebooks are designed for Google Colab with a GPU runtime (T4 or better) and Google Drive access for caching intermediate artifacts (foreground arrays, candidate pools, trained models).

Key dependencies (installed within the notebooks):

- tensorflow 2.20 / keras 3.x
- torch, torchvision (DINO ViT-S/8 for CIFAR-10 and STL-10)
- numpy, scipy, matplotlib, Pillow
- ipywidgets (interactive human-review widget)
- scikit-learn
- tensorflow-datasets (STL-10)

## How to Run

Each notebook follows the same general flow:

1. **Setup:** imports, hyperparameters, seed fixing.
2. **Foreground preprocessing:** run masking (MNIST) or DINO attention (CIFAR-10 / STL-10), or load cached arrays from Drive.
3. **Candidate generation:** build the shared candidate pool.
4. **Filtering:** run the cross-fitted Auto filter and the interactive Human review widget, or load previously approved sets.
5. **Training:** train Baseline, Auto, and Human models on class-matched subsets.
6. **Adversarial evaluation:** FGSM and PGD sweeps across all trained models.
7. **Transfer learning (CIFAR-10 / STL-10):** Baseline vs. Human with MobileNetV2, ResNet50, VGG16, and VGG19.
8. **Checkpointing:** save and load models and metrics from Drive so a fresh session can restore results without retraining.

## Paper

The `paper/` folder contains the accepted manuscript. [Add the IEEE copyright notice here and a link to the IEEE Xplore version once published.]

## Citation

```bibtex
@inproceedings{ehrlich2026humanguided,
  title     = {Human-Guided Augmentation Filtering for Data-Scarce Image Classifiers: Gains and Robustness Limits},
  author    = {Ehrlich, Miles and Rampatsingh, Kamil and Rahouti, Mohamed and Xiong, Kaiqi},
  booktitle = {Proceedings of the IEEE Conference on Dependable and Secure Computing (DSC)},
  year      = {2026}
}
```

## Authors

- Miles Ehrlich, Fordham University
- Kamil Rampatsingh, Fordham University
- Mohamed Rahouti, Fordham University
- Kaiqi Xiong, University of South Florida

## Limitations

Results are based on ten-class datasets, a single human reviewer, and white-box attacks only. Future work includes larger and fine-grained datasets, multiple reviewers, black-box and transfer attacks, and vision-language models for scalable quality control.

## License

See [LICENSE](LICENSE).
