# SmartMeter Appliance Intelligence — Current Project State

This file records the current factual operational state of the repository.

---

## 1. Project Mission & Identity

- **Project Name:** SmartMeter Appliance Intelligence
- **Public Headline:** *An end-to-end MLOps system for weakly supervised appliance detection and temporal localization from smart-meter active power.*
- **Core Methodology:** Multiple-instance weak supervision on aggregate active power windows ($T=510$ points / 68 min @ 8s resolution) to extract temporal appliance activation masks.

---

## 2. Current Authorized Work

- **Gate A (Core ML Viability):** Authorized to execute the controlled 5-experiment ML campaign.
- **Immediate Task:** Implement **Experiment 3: Multi-Scale Temporal ResNet Ensemble + CAM-Based Localization** (ensemble of $\text{ResNet}(k)$ with $k \in \{5, 7, 9, 15, 25\}$ + CAM temporal timeline extraction).
- **Hold Policy:** Full MLOps infrastructure (Gate B) and holdout test evaluation (H2, H13) remain on hold until Gate A criteria (validation Localization F1 $\ge 0.50 - 0.60$) are satisfied.

---

## 3. Dataset & Preprocessing Status

- **Dataset:** *REFIT: Electrical Load Measurements (Cleaned)* (DOI: 10.5281/zenodo.5063428). 20 households, 119,495,879 rows, 6.47 GB uncompressed.
- **Appliance Ground Truth (Decision D-005):** Kettle ($\text{Power} \ge 1500\text{ W}$, continuity gap $\le 20\text{ s}$, duration ceiling $600\text{ s}$).
- **Timebase & Windowing (Decision D-006):** 8.0-second regular timebase, $\le 16\text{ s}$ zero-order hold bridge, $> 16\text{ s}$ segment break, 510-point strictly contiguous windows.
- **Frozen Split (Decision D-007):**
  - **TRAIN (16 households / 141,003 windows):** H1, H3, H5, H6, H7, H8, H9, H10, H11, H12, H15, H16, H18, H19, H20, H21.
  - **VALIDATION (2 households / 18,561 windows):** H4, H17.
  - **TEST (2 households / 14,508 windows — Sealed):** H2, H13.
- **Sealed Data Snapshot (Decision D-011):** `artifacts/data_snapshot/kettle_train_val_snapshot_2026-09-24.zip` (18 `.npy` arrays, 159,564 windows, 18,561 validation target records, SHA256: `dd41f144ceafa516d7b2ba2c0bb42959f7a6e7e7da3cf6a48d96b78da3b8e36a`).

---

## 4. Completed Experiments (Classification Diagnostics)

### Experiment 1 — Simple 1D CNN Baseline (Decision D-008)
- **Architecture:** 2 Conv1d layers + GAP + Linear head (1 logit).
- **Runtime:** Local CPU (6 epochs, early stopped at epoch 6).
- **Result:** Negative sanity baseline (Precision 19.92%, Recall 100.0%, F1 33.23%, Balanced Accuracy 50.0%). Shows near all-positive window classification behavior.
- **Artifacts:** `artifacts/experiments/cnn_baseline/`

### Experiment 2 — 1D ResNet Reference Model (Decision D-009)
- **Architecture:** 3 ResNet blocks (`[64, 128, 128]` channels, `{8, 5, 3}` kernels) + GAP + Linear head (1 logit).
- **Runtime:** Google Colab Tesla T4 GPU (9 epochs, early stopped at epoch 9).
- **Result:** Best Epoch 4 — Validation Window F1: **0.3488** (34.88%), Balanced Accuracy: **0.5399** (53.99%), Precision: **21.36%**, Recall: **94.97%** (TP: 3,512, FP: 12,928, TN: 1,935, FN: 186).
- **Artifacts:** `artifacts/experiments/resnet_reference/` (Weights, metrics, curves, provenance JSON).

---

## 5. Source Control & Environment Status

- **Canonical Source Tree:** `D:\SmartMeter` (local working repository).
- **GitHub Repository:** `https://github.com/nayanojwal0810/smartmeter.git` (`main` branch).
- **Canonical Git Commit:** [`1b4590f2278ee401681dab54892f32c38554d514`](file:///d:/SmartMeter/.git/refs/heads/main) (`Record reproducible Kettle data snapshot`).
- **Binary Storage:** Large `.npy` arrays, large `.jsonl` manifests, and `*.pt` weights reside locally and on Google Drive.
- **Compute Execution:** Google Colab operates strictly as an ephemeral GPU compute engine.

---

## 6. Technical Invariants & Constraints

1. **Test Set Isolation:** H2 and H13 remain completely sealed until Gate A is passed.
2. **Zero Strong-Target Leakage:** Strong sub-meter measurements are strictly evaluation-only.
3. **Train-Derived Normalization:** $\mu_{\text{train}} = 523.94\text{ W}$, $\sigma_{\text{train}} = 764.91\text{ W}$ fitted strictly on the 16 TRAIN households.
4. **Metric Definition:** Primary Gate A acceptance metric is **temporal Localization F1**, not window classification F1.
5. **No Forced Novelty Claims:** Position research literature accurately as technical foundation.

---

## 7. Immediate Next Action

Implement **Experiment 3: Multi-Scale Temporal ResNet Ensemble + CAM-Based Localization** (`src/models/` and evaluation pipeline), testing the 5-model ensemble ($k \in \{5, 7, 9, 15, 25\}$) and CAM activation extraction against validation households H4 and H17.
