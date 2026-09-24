"""Unit and Smoke Tests for Simple 1D CNN Baseline (Decision D-008).

Verifies:
1. Model input/output tensor shapes (both 2D and 3D inputs).
2. Raw logits and sigmoid probabilities.
3. BCEWithLogitsLoss computation.
4. Normalization statistics calculation strictly uses training data.
5. Isolation of test households (H2, H13) from train/val loaders.
6. Exclusion of strong targets from model training tensors.
7. Deterministic configuration loading and reproducibility.
8. Trivial majority baseline reference logic.
9. Lightweight smoke training loop on synthetic data.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.data.dataset import KettleWeakDataset, NormalizationTransform
from src.data.split import HouseholdSplitManager
from src.models.cnn_baseline import Simple1DCNN
from src.models.train import (
    compute_binary_classification_metrics,
    compute_train_normalization_stats,
    evaluate_trivial_majority_baseline,
    evaluate_validation_strong_targets,
    run_training_loop,
    set_seed,
)


def test_model_architecture_shapes():
    """Verify exact 1D CNN baseline layer output shapes and forward pass."""
    model = Simple1DCNN(
        in_channels=1,
        conv1_channels=32,
        conv1_kernel=9,
        pool_kernel=2,
        conv2_channels=64,
        conv2_kernel=9,
        out_features=1,
    )
    model.eval()

    # 3D input: (B, 1, 510)
    x_3d = torch.randn(8, 1, 510)
    out_3d = model(x_3d)
    assert out_3d.shape == (8, 1), f"Expected shape (8, 1), got {out_3d.shape}"

    # 2D input: (B, 510)
    x_2d = torch.randn(16, 510)
    out_2d = model(x_2d)
    assert out_2d.shape == (16, 1), f"Expected shape (16, 1), got {out_2d.shape}"

    # Verify predict_proba is in [0, 1]
    probs = model.predict_proba(x_3d)
    assert probs.shape == (8, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_model_bce_loss_computation():
    """Verify BCEWithLogitsLoss computes valid scalar loss."""
    model = Simple1DCNN()
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randn(4, 1, 510)
    y_weak = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

    logits = model(x)
    loss = criterion(logits, y_weak)

    assert loss.dim() == 0, "Loss must be scalar"
    assert loss.item() >= 0.0, "Loss must be non-negative"
    assert not torch.isnan(loss), "Loss must not be NaN"


def test_train_normalization_uses_only_train_data():
    """Verify normalization statistics are computed exclusively on training data."""
    # Synthetic train dataset with known mean=100, std=20
    x_train = np.random.normal(loc=100.0, scale=20.0, size=(200, 510)).astype(np.float32)
    y_train = np.ones(200, dtype=np.int64)

    # Synthetic val dataset with different mean=500, std=50
    x_val = np.random.normal(loc=500.0, scale=50.0, size=(50, 510)).astype(np.float32)
    y_val = np.ones(50, dtype=np.int64)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    mean, std = compute_train_normalization_stats(train_ds, batch_size=64)

    assert abs(mean - 100.0) < 5.0, f"Expected train mean ~100.0, got {mean}"
    assert abs(std - 20.0) < 3.0, f"Expected train std ~20.0, got {std}"

    # Verify transform applies train statistics
    transform = NormalizationTransform(mean=mean, std=std, method="standardize")
    sample_raw = np.array([100.0] * 510, dtype=np.float32)
    sample_norm = transform(sample_raw)
    assert abs(sample_norm.mean()) < 0.1, "Normalized sample should be near 0"


def test_test_household_isolation():
    """Verify test households (H2, H13) are strictly forbidden from train and validation datasets."""
    split_config = HouseholdSplitManager.load_from_yaml(Path("configs/kettle_household_split.yaml"))

    train_houses = set(split_config.train_households)
    val_houses = set(split_config.validation_households)
    test_houses = set(split_config.test_households)

    assert test_houses == {2, 13}, "Test households must be exactly H2 and H13"
    assert len(train_houses.intersection(test_houses)) == 0, "Train split must not overlap test split"
    assert len(val_houses.intersection(test_houses)) == 0, "Validation split must not overlap test split"

    # Verify dataset loader instantiation
    train_dataset = KettleWeakDataset(split="train")
    for s in train_dataset._samples:
        assert s.household_id not in test_houses, f"Test household {s.household_id} found in train dataset!"

    val_dataset = KettleWeakDataset(split="validation")
    for s in val_dataset._samples:
        assert s.household_id not in test_houses, f"Test household {s.household_id} found in val dataset!"


def test_training_dataset_does_not_expose_strong_targets():
    """Verify KettleWeakDataset returns strictly (aggregate_w[510], weak_label)."""
    dataset = KettleWeakDataset(split="train")
    assert len(dataset) > 0, "Train dataset should contain samples"

    x_tensor, weak_label = dataset[0]
    assert isinstance(x_tensor, np.ndarray), "Window must be numpy array"
    assert x_tensor.shape == (510,), f"Expected shape (510,), got {x_tensor.shape}"
    assert x_tensor.dtype == np.float32, f"Expected float32 dtype, got {x_tensor.dtype}"
    assert weak_label in {0, 1}, f"Weak label must be 0 or 1, got {weak_label}"


def test_deterministic_config_loading():
    """Verify config loading and deterministic seed setting."""
    config_path = Path("configs/cnn_baseline.yaml")
    assert config_path.exists(), "configs/cnn_baseline.yaml must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"]["name"] == "simple_1d_cnn"
    assert cfg["training"]["appliance"] == "kettle"
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["fixed_threshold"] == 0.5

    set_seed(cfg["training"]["seed"])
    t1 = torch.randn(10)
    set_seed(cfg["training"]["seed"])
    t2 = torch.randn(10)
    assert torch.equal(t1, t2), "Deterministic seeding must produce identical tensors"


def test_trivial_majority_baseline_logic():
    """Verify majority-class predictor logic on known imbalanced labels."""
    # 80 negative, 20 positive -> majority = 0
    targets_map = {i: (1 if i < 20 else 0) for i in range(100)}
    metrics = evaluate_trivial_majority_baseline(targets_map)

    assert metrics["majority_class"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["accuracy"] == 0.8
    assert metrics["tn"] == 80
    assert metrics["fn"] == 20


def test_smoke_training_loop(tmp_path):
    """Run a tiny 2-epoch training loop on synthetic data to verify end-to-end execution."""
    train_dataset = KettleWeakDataset(split="train")
    val_dataset = KettleWeakDataset(split="validation")

    dummy_train = np.random.uniform(50.0, 3000.0, size=(20, 510)).astype(np.float32)
    dummy_val = np.random.uniform(50.0, 3000.0, size=(10, 510)).astype(np.float32)

    train_dataset.register_household_array(1, dummy_train)
    val_dataset.register_household_array(4, dummy_val)

    train_dataset._samples = train_dataset._samples[:20]
    val_dataset._samples = val_dataset._samples[:10]

    targets_map = {val_dataset.get_metadata(i)["window_id"]: (1 if i % 2 == 0 else 0) for i in range(10)}

    config = {
        "model": {
            "in_channels": 1,
            "conv1_channels": 8,
            "conv1_kernel": 9,
            "pool_kernel": 2,
            "conv2_channels": 16,
            "conv2_kernel": 9,
            "out_features": 1,
        },
        "training": {
            "seed": 42,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "fixed_threshold": 0.5,
            "normalization": "standardize",
        },
    }

    metrics = run_training_loop(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        targets_map=targets_map,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert (tmp_path / "model_best.pt").exists()
    assert (tmp_path / "model_final.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "normalization_stats.json").exists()
    assert (tmp_path / "trivial_baseline.json").exists()
    assert (tmp_path / "training_curves.json").exists()
    assert metrics["total_epochs_trained"] == 2


def test_production_cli_data_preparation_contract(tmp_path):
    """Verify production CLI data-preparation function signature and dataset instantiation contract."""
    from scripts.train_cnn_baseline import ensure_mmap_arrays_exist

    split_cfg_path = Path("configs/kettle_household_split.yaml")
    split_mgr = HouseholdSplitManager.from_yaml(split_cfg_path)
    assert isinstance(split_mgr, HouseholdSplitManager)
    assert hasattr(split_mgr, "config")

    # Verify ensure_mmap_arrays_exist accepts split_manager keyword argument
    mock_raw_dir = tmp_path / "raw"
    mock_raw_dir.mkdir(parents=True, exist_ok=True)
    mock_proc_dir = tmp_path / "proc"
    mock_proc_dir.mkdir(parents=True, exist_ok=True)

    # Call with split_manager keyword arg (matching CLI caller contract)
    ensure_mmap_arrays_exist(
        processed_dir=mock_proc_dir,
        raw_data_dir=mock_raw_dir,
        split_manager=split_mgr,
        splits=["train", "validation"],
    )

    # Verify test households are never extracted or included in target splits
    train_h = set(split_mgr.config.train_households)
    val_h = set(split_mgr.config.validation_households)
    test_h = set(split_mgr.config.test_households)

    assert test_h == {2, 13}
    assert not train_h.intersection(test_h)
    assert not val_h.intersection(test_h)

    # Verify dataset loader instantiation and split manager access
    train_ds = KettleWeakDataset(
        split="train",
        data_dir=mock_proc_dir,
        split_config_path=split_cfg_path,
        window_manifest_path=Path("artifacts/manifests/kettle_household_split.json"),
    )
    assert hasattr(train_ds.split_manager, "config")
    assert train_ds.split_manager.config.train_households == split_mgr.config.train_households

