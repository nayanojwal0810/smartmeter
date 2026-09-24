# Google Colab Execution Guide: 1D ResNet Reference Model

This guide provides the exact step-by-step instructions to run the 1D ResNet Reference Model experiment in Google Colab with GPU acceleration.

---

## 1. Prerequisites & Colab Runtime Setup

1. Open [Google Colab](https://colab.research.google.com/).
2. In the Colab menu, go to **Runtime** > **Change runtime type**.
3. Select **GPU** (e.g. T4 or A100 GPU hardware accelerator) and click **Save**.

---

## 2. Step-by-Step Execution Workflow

### Step A: Upload & Extract the Bundle
Upload `smartmeter_resnet_colab.zip` to Colab (via the Files sidebar or Google Drive), then run:

```bash
!unzip -q smartmeter_resnet_colab.zip -d smartmeter_resnet_colab
```

### Step B: Navigate into Project Directory
```bash
%cd smartmeter_resnet_colab
```

### Step C & D: Environment Setup & GPU Verification
Run the automated setup script to install dependencies and verify the environment:

```bash
!bash colab/setup_colab.sh
```

Or manually install and check:
```bash
!pip install -r colab/requirements.txt
!python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step E: Run Full Test Suite
Validate project integrity and all 62 unit/integration tests:

```bash
!python -m pytest -v
```

### Step F: Run ResNet Smoke Test
Verify the ResNet model forward/backward/checkpoint pipeline on synthetic data:

```bash
!python scripts/train_resnet.py --smoke-test
```

### Step G: Run Production Preflight
Verify dataset shapes, splits, test isolation (H2/H13 sealed), and evaluation target mappings without starting training:

```bash
!python scripts/train_resnet.py --config configs/resnet_reference.yaml --preflight
```

### Step H: Run the Real 1D ResNet Training Command
Execute the full multi-epoch 1D ResNet training run on GPU:

```bash
!python scripts/train_resnet.py --config configs/resnet_reference.yaml
```

*Note: CUDA GPU is automatically detected and selected by default (`--device auto`).*

---

## 3. Locating Generated Artifacts

Upon completion of training, all authoritative experiment artifacts are saved under `artifacts/experiments/resnet_reference/`:

- `best_resnet1d_reference_model.pt`: Checkpoint weights of the best validation model.
- `resnet_reference_metrics.json`: Full epoch-by-epoch loss, validation strong-target metrics (F1, Precision, Recall, Balanced Accuracy), and baseline comparisons.
- `resnet_report.md`: Formatted Markdown summary report of the experiment.
- `normalization_stats.json`: Train-only mean and standard deviation used for standardization.
- `trivial_baseline.json`: Majority/all-positive reference performance metrics.

To download all generated artifacts from Colab:

```python
from google.colab import files
import shutil

shutil.make_archive("resnet_reference_results", "zip", "artifacts/experiments/resnet_reference")
files.download("resnet_reference_results.zip")
```

---

## 4. Locked Experiment Policies Preserved

- **Target Appliance:** Kettle
- **Frozen Split:**
  - TRAIN: H1, H3, H5, H6, H7, H8, H9, H10, H11, H12, H15, H16, H18, H19, H20, H21 (16 households, 141,003 windows)
  - VALIDATION: H4, H17 (2 households, 18,561 windows)
  - TEST: H2, H13 (Strictly sealed, excluded from processed data)
- **Input:** Aggregate household power only (510 points @ 8s resolution)
- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam (lr=1e-3, weight_decay=0.0)
- **Batch Size:** 256
- **Max Epochs:** 30 (Early stopping patience = 5)
- **Model Selection:** Validation strong-target F1 score at threshold 0.5
