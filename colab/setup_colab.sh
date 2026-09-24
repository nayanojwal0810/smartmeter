#!/usr/bin/env bash
# setup_colab.sh - Environment verification and setup for Google Colab
set -e

echo "=================================================="
echo "SmartMeter 1D ResNet Reference Setup (Google Colab)"
echo "=================================================="

# 1. Install dependencies
echo ""
echo "[Step 1/4] Installing / verifying Python dependencies..."
pip install -r colab/requirements.txt

# 2. Verify Python, PyTorch, and CUDA
echo ""
echo "[Step 2/4] Verifying Python and PyTorch / CUDA runtime..."
python -c "
import sys, torch, numpy, yaml, pytest
print('  Python version :', sys.version.split()[0])
print('  PyTorch version:', torch.__version__)
print('  NumPy version  :', numpy.__version__)
print('  PyYAML version :', yaml.__version__)
print('  Pytest version :', pytest.__version__)
cuda_avail = torch.cuda.is_available()
print('  CUDA Available :', cuda_avail)
if cuda_avail:
    print('  GPU Device Name:', torch.cuda.get_device_name(0))
    print('  Device Count   :', torch.cuda.device_count())
    print('  VRAM Total (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
else:
    print('  NOTE: CUDA is not available. Execution will fall back to CPU.')
"

# 3. Verify directory structure and required data files
echo ""
echo "[Step 3/4] Verifying dataset and manifest presence..."
python -c "
from pathlib import Path

train_households = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21]
val_households = [4, 17]
test_households = [2, 13]

proc_dir = Path('data/processed/kettle')
missing_train = [h for h in train_households if not (proc_dir / f'house_{h}_aggregate.npy').exists()]
missing_val = [h for h in val_households if not (proc_dir / f'house_{h}_aggregate.npy').exists()]
leaked_test = [h for h in test_households if (proc_dir / f'house_{h}_aggregate.npy').exists()]

assert not missing_train, f'Missing train household arrays: {missing_train}'
assert not missing_val, f'Missing val household arrays: {missing_val}'
assert not leaked_test, f'Test household arrays present (leakage violation!): {leaked_test}'

manifests = [
    Path('artifacts/manifests/kettle_household_split.json'),
    Path('artifacts/manifests/timebase_window_index.json'),
    Path('artifacts/manifests/kettle_evaluation_targets.jsonl'),
]
for m in manifests:
    assert m.exists(), f'Missing required manifest: {m}'

print('  Processed Train Households (16): OK')
print('  Processed Val Households (2)   : OK')
print('  Test Households Isolated (H2,H13): OK (Sealed)')
print('  Required Manifests              : OK')
"

# 4. Ready summary
echo ""
echo "[Step 4/4] Colab environment setup complete!"
echo "Next commands to run in Colab:"
echo "  1. python -m pytest -v"
echo "  2. python scripts/train_resnet.py --smoke-test"
echo "  3. python scripts/train_resnet.py --config configs/resnet_reference.yaml --preflight"
echo "  4. python scripts/train_resnet.py --config configs/resnet_reference.yaml"
echo "=================================================="
