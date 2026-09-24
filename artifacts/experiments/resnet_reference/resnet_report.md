# 1D ResNet Reference Model Experiment Report

## Executive Summary

- **Experiment:** `resnet1d_reference`  
- **Appliance:** `kettle`  
- **Seed:** `42`  
- **Best Epoch:** `4` (out of 9 trained)  
- **Train Standardization:** $\mu = 523.94\text{ W},\; \sigma = 764.91\text{ W}$ (fitted strictly on TRAIN)  

## Validation Performance vs Trivial Reference Baseline

| Metric | 1D ResNet Reference (Threshold = 0.5) | Trivial Majority Reference (Predict Inactive) |
|---|---|---|
| **Precision** | **21.36%** | 0.00% |
| **Recall** | **94.97%** | 0.00% |
| **F1-Score** | **34.88%** | 0.00% |
| **Balanced Accuracy** | **53.99%** | 50.00% |
| **Overall Accuracy** | **29.35%** | 80.08% |
| **True Positives (Active)** | **3,512** | 0 |
| **False Positives** | **12,928** | 0 |
| **True Negatives (Inactive)** | **1,935** | 14,863 |
| **False Negatives** | **186** | 3,698 |

## Methodological Invariants & Architectural Scope

1. **Binary Head Adaptation:** The published CamAL paper describes a 2-class softmax classification head. Our project intentionally implements a single output logit with `BCEWithLogitsLoss` and sigmoid-equivalent binary decision at threshold 0.5 as defined by our project model-training specification. It is an intentional binary-classification implementation adaptation, not byte-for-byte or mathematically identical to the paper's 2-class softmax head.
2. **Experiment Scope:** This experiment evaluates a **single ResNet reference model**. It is NOT yet the multi-scale CamAL ensemble, and Class Activation Map (CAM) localization is NOT yet implemented in this experiment. The eventual ensemble and localization pipeline will be a separate, downstream experiment.
3. **Test Set Isolation:** Test households H2 and H13 were strictly excluded from training, normalization, validation, and early stopping.
4. **Weak Label Supervision:** Model was trained exclusively against binary household-level metadata presence labels $y \in \{0, 1\}$.
5. **Zero Strong-Target Leakage:** Strong sub-meter timestamp annotations and window targets were completely absent from model inputs and loss functions.
6. **Threshold Policy:** Classification probability threshold was fixed at $0.5$ without validation-based post-hoc tuning.

## Checkpoints & Artifacts

- **Best Model Checkpoint:** `artifacts/experiments/resnet_reference/model_best.pt`  
- **Final Model Checkpoint:** `artifacts/experiments/resnet_reference/model_final.pt`  
