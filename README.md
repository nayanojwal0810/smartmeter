# SmartMeter NILM: Weakly Supervised Kettle Detection (1D ResNet Reference)

This repository contains the end-to-end pipeline for weakly supervised Non-Intrusive Load Monitoring (NILM) on the REFIT dataset, specifically focused on detecting Kettle activations using aggregate power measurements.

## Project Overview

- **Task:** Weakly supervised Kettle detection from aggregate active power windows (510 points @ 8s interval = 68 minutes).
- **Model:** 1D ResNet Reference Model (3 residual blocks with [64, 128, 128] channels, kernel sizes [8, 5, 3], GAP + Linear head).
- **Supervision:** Household-level weak Kettle label (1 = monitored Kettle, 0 = unmonitored proxy).
- **Evaluation:** Strong Kettle ground truth activations on validation households (H4, H17) evaluated via strict Precision, Recall, F1, and Balanced Accuracy at fixed threshold 0.5.
- **Sealed Test Set:** H2 and H13 are strictly sealed.

## Quick Start (Google Colab)

Refer to [colab/README.md](colab/README.md) for step-by-step instructions to train the model on Google Colab with GPU acceleration.

### Quick Commands:
```bash
# 1. Setup environment
bash colab/setup_colab.sh

# 2. Run test suite
python -m pytest -v

# 3. ResNet Smoke Test
python scripts/train_resnet.py --smoke-test

# 4. Production Preflight
python scripts/train_resnet.py --config configs/resnet_reference.yaml --preflight

# 5. Run Real 1D ResNet Training
python scripts/train_resnet.py --config configs/resnet_reference.yaml
```
