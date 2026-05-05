# Dual-Soft-Prompting SCL 

**Anonymous Repository for Peer Review**

This repository contains the official implementation of the paper: *"Dual-Soft-Prompting SCL: Parameter-Efficient Malicious Content Detection in Extreme Few-Shot Scenarios"*.

We propose a minimalist framework that explicitly enforces intra-class compactness and inter-class separability within a frozen PLM's representation manifold via **Dual-Space Alignment**, eliminating the dependency on large batch sizes in extreme few-shot regimes (8-16 shots).

## 🚀 Key Features
- **Extreme Parameter Efficiency**: Requires only 22.5K trainable parameters (<0.006% of RoBERTa-large).
- **Batch-Independent Alignment**: Replaces conventional batch-dependent Supervised Contrastive Learning (SCL) with Global Learnable Class Prototypes.
- **Robust in Extreme Few-Shot**: Achieves state-of-the-art results on Jigsaw and OLID under 8-shot and 16-shot scenarios.

## ⚙️ Setup & Requirements

```bash
git clone <anonymous_github_url>
cd dual-soft-prompting-scl
pip install -r requirements.txt
