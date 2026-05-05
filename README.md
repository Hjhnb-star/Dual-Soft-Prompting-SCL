***
```markdown
# Dual-Soft-Prompting SCL: Parameter-Efficient Malicious Content Detection in Extreme Few-Shot Scenarios

> **🔒 Anonymous Repository for Double-Blind Peer Review**
> This repository contains the official PyTorch implementation of the paper *"Dual-Soft-Prompting SCL"*. All author identities and institutional affiliations have been anonymized.

## 📖 Overview

The exponential proliferation of user-generated content necessitates robust malicious content detection. While Parameter-Efficient Fine-Tuning (PEFT) methods reduce computational overhead, they frequently suffer from catastrophic overfitting, gradient sparsity, and representation collapse in **extreme few-shot scenarios (e.g., 8-16 samples per class)**.

To address the **"over-parameterization trap"**, we propose **Dual-Soft-Prompting Supervised Contrastive Learning (Dual-Soft-Prompting SCL)**. This minimalist framework integrates an input-level soft prompt with global learnable class prototypes in the final latent space. By optimizing a prototype-centered supervised alignment objective, it explicitly enforces intra-class compactness and inter-class separability within the frozen PLM's manifold without relying on large batch sizes.

## ✨ Key Features

*   **Extreme Parameter Efficiency:** Achieves state-of-the-art results by updating only **22.5K parameters** (less than 0.006% of the frozen RoBERTa-large backbone).
*   **Batch-Independent Alignment:** Introduces a Prototype-Centered Supervised Alignment (PC-SA) loss, eliminating the dependency on large mini-batches common in traditional SCL.
*   **Overcoming the Over-parameterization Trap:** Demonstrates that minimalist random initialization with dual-space alignment yields optimal generalization compared to heavily parameterized architectures.


## 📂 Repository Structure

To ensure modularity and readability, the codebase is structured as follows:

```text
Dual-Soft-Prompting-SCL/
│
├── main.py               # Main execution script with argparse for hyperparameter tuning
├── data_utils.py         # Handles extreme few-shot sampling and HuggingFace dataset processing
├── trainer_utils.py      # Contains the custom DualPromptSCLTrainer and PC-SA loss implementation
├── requirements.txt      # Python package dependencies
└── README.md             # This document
```

---

## 🚀 How to Run (Reproducing Main Results)

Our entry point (`main.py`) uses `argparse` for flexible execution. You can easily switch between datasets (`jigsaw`, `olid`), methods, and few-shot settings (`num_shots`).

### 1. Run Our Proposed Method (Dual-Soft-Prompting SCL)
To reproduce the **8-shot** results on the **Jigsaw** dataset (Table 2 in the paper):

```bash
python main.py \
    --dataset jigsaw \
    --method Ours-DualPromptSCL \
    --num_shots 8 \
    --epochs 80 \
    --learning_rate 3e-3 \
    --scl_alpha 0.2 \
    --seed 42
```
*Expected Output: Macro-F1 around 0.8702.*

### 2. Run Baselines
To evaluate standard PEFT baselines (e.g., LoRA) under the same few-shot constraints:

```bash
python main.py \
    --dataset olid \
    --method LoRA \
    --num_shots 16 \
    --epochs 30 \
    --learning_rate 3e-4 \
    --seed 42
```
*Supported Baseline Methods: `LoRA`, `IA3-Adapter`, `Prefix-Tuning`, `Vanilla-PT`.*

### 3. Ablation Study: The "Over-parameterization Trap"
To observe the catastrophic degradation caused by excessive trainable capacity in extreme few-shot regimes (Ours-DeepDualPromptSCL with ~1.48M parameters):

```bash
python main.py \
    --dataset jigsaw \
    --method Ours-DeepDualPromptSCL \
    --num_shots 8 \
    --epochs 50 \
    --learning_rate 2e-3
```

---

## 📊 Datasets

The scripts utilize HuggingFace's `datasets` library to automatically download and process the following benchmarks:
1.  **Jigsaw Toxic Comment Classification**: Converted to a binary classification task (toxic vs. non-toxic).
2.  **OLID (OffensEval 2019)**: Task A (offensive language detection).

Extreme few-shot subsets (8/16 samples per class) are sampled dynamically during runtime using fixed random seeds (`--seed 42`) to guarantee **deterministic reproducibility**.

---
*Disclaimer: This repository is intended solely for peer review. Upon acceptance, the codebase will be de-anonymized, integrated with proper citation details, and published under an open-source license.*
```

***

