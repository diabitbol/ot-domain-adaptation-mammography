# 🩻 OT Domain Adaptation for Mammography

**Domain adaptation with Optimal Transport (OT)** for robust breast cancer detection across heterogeneous mammography devices.  
This project explores OT-based methods to correct domain shifts between datasets acquired from different imaging systems — a key challenge in clinical AI deployment.

---

## 🌍 Overview

In mammography, **domain shifts** arise when hospitals replace or mix imaging machines (e.g., film vs full-field digital), causing strong **degradation of CNN classifiers** trained on previous data.  
Re-training on each new device is costly and data-hungry, motivating **unsupervised domain adaptation**.

We investigate **Optimal Transport (OT)** as a mathematically grounded way to align distributions between a **source** and **target** domain.  
OT provides an explicit metric (e.g., Wasserstein distance) and a transport plan between domains, offering a lightweight alternative to adversarial adaptation.

---

## 🎯 Objective

> **Goal:** improve cross-domain mammography classification performance (cancer vs non-cancer) using OT-based domain adaptation.

- **Source domain:** [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)  
- **Target domain:** [VinDr-Mammo](https://physionet.org/content/vindr-mammo/1.0.0/)

We train a CNN on CBIS-DDSM and adapt it to VinDr-Mammo using OT between either:
1. Pixel distributions (histogram-level), or  
2. Deep feature spaces (latent-level, Sinkhorn or barycentric mapping).

---

## 🧩 Methods

| Method | Description | Reference |
|--------|--------------|------------|
| **Source-Only** | CNN trained only on CBIS-DDSM, evaluated on VinDr-Mammo | baseline |
| **DANN** | Adversarial domain adaptation (discriminator-based) | Ganin et al. 2015 |
| **OT-F (Sinkhorn)** | Feature-level Sinkhorn divergence minimization | Genevay et al. 2019 |
| **OT-BJ** | Barycentric mapping between feature spaces | Courty et al. 2017 |
| **DeepJDOT** | Joint distribution OT between features and pseudo-labels | Damodaran et al. 2018 |

---

## 🧠 Architecture

A lightweight **ResNet-18** or **ConvNeXt-Tiny** backbone is used as a feature extractor.  
Adaptation losses (OT, adversarial) are attached to its intermediate representation.

