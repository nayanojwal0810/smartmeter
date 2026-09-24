# Simple 1D CNN Baseline Experiment Report (Decision D-008)

## Executive Summary

- **Experiment:** `simple_1d_cnn_baseline`  
- **Appliance:** `kettle`  
- **Seed:** `42`  
- **Best Epoch:** `1` (out of 6 trained)  
- **Train Standardization:** $\mu = 523.94\text{ W},\; \sigma = 764.91\text{ W}$ (fitted strictly on TRAIN)  

## Validation Performance vs Trivial Reference Baseline

| Metric | Simple 1D CNN Baseline (Threshold = 0.5) | Trivial Majority Reference (Predict Inactive) |
|---|---|---|
| **Precision** | **19.92%** | 0.00% |
| **Recall** | **100.00%** | 0.00% |
| **F1-Score** | **33.23%** | 0.00% |
| **Balanced Accuracy** | **50.00%** | 50.00% |
| **Overall Accuracy** | **19.92%** | 80.08% |
| **True Positives (Active)** | **3,698** | 0 |
| **False Positives** | **14,863** | 0 |
| **True Negatives (Inactive)** | **0** | 14,863 |
| **False Negatives** | **0** | 3,698 |

## Methodological Invariants Verified

1. **Test Set Isolation:** Test households H2 and H13 were strictly excluded from training, normalization, validation, and early stopping.
2. **Weak Label Supervision:** Model was trained exclusively against binary household-level metadata presence labels $y \in \{0, 1\}$.
3. **Zero Strong-Target Leakage:** Strong sub-meter timestamp annotations and window targets were completely absent from model inputs and loss functions.
4. **Threshold Policy:** Classification probability threshold was fixed at $0.5$ without validation-based post-hoc tuning.

## Checkpoints & Artifacts

- **Best Model Checkpoint:** `artifacts/experiments/cnn_baseline/model_best.pt`  
- **Final Model Checkpoint:** `artifacts/experiments/cnn_baseline/model_final.pt`  
