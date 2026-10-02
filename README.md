# Domain Adaptation for Mammography Classification

Exploring simple methods to reduce performance degradation caused by domain shift between two mammography datasets.

## Overview

Deep learning models for medical imaging can achieve strong performance when training and test data come from similar distributions. However, their performance may deteriorate substantially when deployed on images acquired under different conditions.

This phenomenon is known as **domain shift**.

In mammography, domain shift may arise from differences in:

- imaging devices,
- acquisition protocols,
- image reconstruction,
- preprocessing pipelines,
- patient populations,
- image contrast and intensity distributions.

The objective of this project is to investigate simple methods for reducing the loss of performance of a mammography classifier when moving from one image distribution to another.

We consider:

- **Mini-DDSM** as the source domain;
- **VinDr-Mammo** as the target domain.

A binary classifier is first trained on Mini-DDSM and then evaluated directly on VinDr-Mammo. Several adaptation strategies are subsequently explored.

## Problem formulation

The task is formulated as a binary classification problem:

- **Class 0:** normal mammogram;
- **Class 1:** mammogram containing a potentially abnormal mass.

The classifier is trained on Mini-DDSM and evaluated on VinDr-Mammo without initially modifying its parameters.

The purpose of this setup is to measure how strongly the model is affected by the distribution shift between the two datasets.

## Datasets

### Mini-DDSM

Mini-DDSM is used as the **source domain**.

The dataset contains digitized film mammograms and exhibits substantial variability, including noise and acquisition artifacts.

Only **MLO (Mediolateral Oblique)** views are retained in our experiments.

Images are resized to:

```text
700 x 700 pixels
```

### VinDr-Mammo

VinDr-Mammo is used as the **target domain**.

It contains digital mammograms acquired using more modern imaging systems and presents noticeably different image statistics compared with Mini-DDSM.

After filtering, the dataset used in our experiments contains:

```text
9,999 images
6,703 normal
3,296 abnormal
```

## Model

We use a pretrained **ResNet-18** convolutional neural network and fine-tune it for binary mammography classification.

### Training setup

The main training configuration includes:

- pretrained ResNet-18;
- Binary Cross-Entropy loss;
- Adam optimizer;
- learning-rate scheduler;
- weighted random sampling to address class imbalance;
- dropout;
- L2 regularization;
- data augmentation:
  - contrast changes,
  - brightness changes,
  - rotations,
  - horizontal flips.

The Mini-DDSM dataset is divided into:

```text
70% training
15% validation
15% test
```

## Evaluation

Performance is mainly assessed using the **ROC AUC**, complemented by confusion matrices.

Confidence intervals for the AUC are estimated using a **non-parametric bootstrap with 1,000 resamples**.

This choice is particularly useful because the datasets are class-imbalanced and because AUC does not depend on a fixed classification threshold.

## Baseline results

The ResNet-18 classifier performs reasonably well on the source domain:

| Dataset | ROC AUC |
|---|---:|
| Mini-DDSM | **0.77** |
| VinDr-Mammo | **0.53** |

The 95% confidence interval on Mini-DDSM is approximately:

```text
[0.72, 0.81]
```

while on VinDr-Mammo it is approximately:

```text
[0.52, 0.54]
```

The large performance drop when moving from Mini-DDSM to VinDr-Mammo illustrates a strong **domain shift**.

On VinDr-Mammo, the model performs only slightly better than a random classifier.

# Domain adaptation experiments

## 1. Image-level adaptation

The first approach attempts to modify the target images so that their pixel distributions become closer to those observed in the source domain.

### CLAHE

We apply **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to VinDr-Mammo images.

CLAHE operates locally by:

1. dividing an image into small regions;
2. applying histogram equalization within each region;
3. limiting excessive contrast amplification;
4. recombining the transformed regions.

The objective is to make pathological structures more visible while reducing differences in contrast between VinDr-Mammo and Mini-DDSM.

However, despite modifying the pixel-intensity distribution, CLAHE does not significantly improve classification performance.

The ROC AUC remains approximately:

```text
AUC ≈ 0.53
```

## 2. Optimal transport / Quantile matching

We then explore a one-dimensional optimal transport transformation between the pixel-intensity distributions of the two datasets.

For a target-domain pixel intensity $x$, the transformation is:

$$
T(x) = F_s^{-1}(F_t(x))
$$

where:

- $F_t$ is the cumulative distribution function of the target domain;
- $F_s^{-1}$ is the inverse cumulative distribution function of the source domain.

In practice, this corresponds to **quantile matching**.

For example, a value located at the 70th percentile of the VinDr distribution is replaced by the value located at the 70th percentile of the Mini-DDSM distribution.

This transformation successfully aligns the global pixel-intensity histograms.

However, classification performance remains essentially unchanged:

```text
AUC ≈ 0.53
```

This suggests that matching low-level image statistics is not sufficient to align the internal representations learned by the neural network.

## 3. Elastic Weight Consolidation

The most successful approach investigated in this project is **Elastic Weight Consolidation (EWC)**.

Rather than transforming the images, EWC directly adapts the parameters of the neural network to the target domain while attempting to preserve parameters that were important for the source task.

Starting from the model trained on Mini-DDSM, the objective becomes:

```math
\mathcal{L}(\theta)
=
\mathcal{L}_{\mathrm{VinDr}}(\theta)
+
\frac{\lambda}{2}
\sum_i F_i(\theta_i - \theta_i^*)^2

where:

- $\theta_i^*$ are the parameters learned on Mini-DDSM;
- $F_i$ represents the estimated importance of parameter $i$, obtained from the diagonal of the Fisher information matrix;
- $\lambda$ controls the strength of the constraint.

A large value of `lambda` strongly preserves the source-domain model, while a small value allows more adaptation to VinDr-Mammo.

We tested several values of:

$$
\lambda \in [10^2, 10^9]
$$

with three epochs of adaptation.

## EWC results

A value around:

$$
\lambda \approx 5 \times 10^6
$$

provides an interesting compromise between source-domain retention and target-domain adaptation.

| Model | Mini-DDSM AUC | VinDr-Mammo AUC |
|---|---:|---:|
| Before adaptation | **0.77** | **0.53** |
| EWC adaptation | **≈ 0.60** | **≈ 0.60** |

EWC therefore substantially improves performance on the target domain while retaining part of the performance learned on the source domain.

The experiment also illustrates the trade-off inherent in continual/domain adaptation: improving the target-domain performance may come at the cost of some degradation on the original task.

## Main findings

The experiments highlight three main results:

1. **A strong domain shift exists between Mini-DDSM and VinDr-Mammo.**

   A ResNet-18 trained on Mini-DDSM achieves an AUC of approximately 0.77 on its source test set but only approximately 0.53 on VinDr-Mammo.

2. **Pixel-level transformations are not sufficient.**

   CLAHE and one-dimensional optimal transport successfully modify the intensity distributions of the images but do not significantly improve classification performance.

3. **Model-level adaptation is more effective.**

   Elastic Weight Consolidation substantially improves performance on the target domain while constraining catastrophic forgetting on the source domain.

These results suggest that the domain shift between the two mammography datasets cannot be explained solely by differences in pixel intensities. Part of the shift occurs in the higher-level representations learned by the neural network.

## Repository structure

```text
.
├── README.md
├── dev/
│   ├── get_data/
│   │   ├── download_data_VinDr.ipynb
│   │   ├── download_data_ddsm.ipynb
│   │   ├── importing_miniddsm.ipynb
│   │   └── importing_vinDr.ipynb
│   ├── structuring/
│   │   ├── restructuration_VinDr.ipynb
│   │   └── restructuration_miniddsm.ipynb
│   ├── training_on_miniddsm.ipynb
│   ├── evaluation_VinDr.ipynb
│   ├── evaluation_VinDr_clahe.ipynb
│   ├── evaluation_VinDr_transformed.ipynb
│   ├── transformation_VinDr.ipynb
│   ├── fine-tuning.ipynb
│   └── test/
│       ├── training_two_classes.ipynb
│       └── training_three_classes.ipynb
└── ...
```

The repository includes code for:

- dataset preparation;
- image preprocessing;
- ResNet-18 training;
- evaluation on Mini-DDSM and VinDr-Mammo;
- CLAHE preprocessing;
- one-dimensional optimal transport;
- fine-tuning;
- Elastic Weight Consolidation.

## Limitations and future work

### Higher-dimensional optimal transport

Our optimal transport experiment operates only on one-dimensional pixel-intensity distributions.

It therefore ignores:

- spatial information;
- local image structures;
- textures;
- learned feature representations.

Higher-dimensional optimal transport methods could potentially align the domains more effectively.

### Feature-level domain adaptation

Rather than aligning raw pixels, future work could align representations extracted by the neural network.

Possible directions include:

- feature-space optimal transport;
- domain-adversarial neural networks;
- distribution matching in latent space;
- self-supervised adaptation.

### Limited labeled target data

EWC requires labeled data from the target domain.

In real clinical settings, obtaining such labels can be expensive, which motivates the investigation of unsupervised or semi-supervised domain adaptation techniques.

### Larger-scale validation

The robustness of the conclusions should also be evaluated using:

- additional mammography datasets;
- different acquisition devices;
- different hospitals;
- larger patient cohorts.

## Conclusion

This project illustrates the difficulty of deploying medical-image classifiers outside the distribution on which they were trained.

Simple normalization techniques can reduce visible differences between datasets without necessarily improving the representations used by a neural network.

In our experiments, **Elastic Weight Consolidation provides the strongest adaptation among the methods investigated**, increasing performance on VinDr-Mammo while partially preserving the knowledge acquired on Mini-DDSM.

The results emphasize that robust medical AI requires methods capable of adapting not only image statistics, but also the internal representations learned by deep neural networks.

## References

The project builds on work related to:

- domain adaptation for medical image analysis;
- image normalization and radiomics;
- ResNet-18 fine-tuning;
- Elastic Weight Consolidation;
- optimal transport.

In particular:

Kirkpatrick, J. et al.  
*Overcoming catastrophic forgetting in neural networks.*  
Proceedings of the National Academy of Sciences, 2017.