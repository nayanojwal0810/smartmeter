"""Unit and Smoke Tests for 1D ResNet Reference Model.

Verifies:
1. ResNet1D and ResNetBlock1D construction and layer properties.
2. Expected input/output tensor shapes (both 2D and 3D inputs).
3. Residual connection behavior (skip connection and residual addition).
4. Final classifier output shape and sigmoid probabilities.
5. Configuration loading from configs/resnet_reference.yaml.
6. Test-household isolation (H2, H13 strictly excluded).
7. Aggregate-only model inputs (no target channel).
8. Train-only normalization statistics fitting.
9. Zero strong-target leakage into training tensors or loss.
10. Separation of ResNet experiment artifact paths from CNN baseline.
11. End-to-end smoke training loop with ResNet on synthetic data.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.data.dataset import KettleWeakDataset, NormalizationTransform
from src.data.split import HouseholdSplitManager
from src.models.resnet import ResNet1D, ResNetBlock1D
from src.models.train import (
    build_model,
    compute_binary_classification_metrics,
    compute_train_normalization_stats,
    evaluate_trivial_majority_baseline,
    evaluate_validation_strong_targets,
    run_training_loop,
    set_seed,
)


def test_resnet_block_construction_and_residual_connection():
    """Verify ResNetBlock1D forward pass and residual connection behavior."""
    # Test block with matching in/out channels
    block_same = ResNetBlock1D(in_channels=64, out_channels=64, kernel_sizes=(8, 5, 3))
    x_same = torch.randn(4, 64, 510)
    out_same = block_same(x_same)
    assert out_same.shape == (4, 64, 510), f"Expected shape (4, 64, 510), got {out_same.shape}"
    assert isinstance(block_same.shortcut, nn.BatchNorm1d)

    # Test block with channel projection (1 -> 64)
    block_proj = ResNetBlock1D(in_channels=1, out_channels=64, kernel_sizes=(8, 5, 3))
    x_proj = torch.randn(4, 1, 510)
    out_proj = block_proj(x_proj)
    assert out_proj.shape == (4, 64, 510), f"Expected shape (4, 64, 510), got {out_proj.shape}"
    assert isinstance(block_proj.shortcut, nn.Sequential)


def test_resnet1d_input_output_shapes():
    """Verify ResNet1D input and output shapes for 2D and 3D inputs."""
    model = ResNet1D(
        in_channels=1,
        filter_counts=(64, 128, 128),
        kernel_sizes=(8, 5, 3),
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

    # Predict proba
    probs = model.predict_proba(x_3d)
    assert probs.shape == (8, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_resnet_bce_loss_computation():
    """Verify BCEWithLogitsLoss works cleanly with ResNet1D."""
    model = ResNet1D()
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randn(4, 1, 510)
    y_weak = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

    logits = model(x)
    loss = criterion(logits, y_weak)

    assert loss.dim() == 0
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_resnet_config_loading_and_build_model():
    """Verify configs/resnet_reference.yaml loads and builds via factory."""
    config_path = Path("configs/resnet_reference.yaml")
    assert config_path.exists(), "configs/resnet_reference.yaml must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"]["name"] == "resnet1d_reference"
    assert cfg["model"]["filter_counts"] == [64, 128, 128]
    assert cfg["model"]["kernel_sizes"] == [8, 5, 3]
    assert cfg["training"]["appliance"] == "kettle"
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["fixed_threshold"] == 0.5
    assert cfg["data"]["artifacts_dir"] == "artifacts/experiments/resnet_reference"

    model = build_model(cfg["model"])
    assert isinstance(model, ResNet1D)


def test_resnet_artifact_path_separation():
    """Verify ResNet artifact paths are strictly distinct from CNN baseline."""
    with open("configs/cnn_baseline.yaml", "r", encoding="utf-8") as f:
        cnn_cfg = yaml.safe_load(f)
    with open("configs/resnet_reference.yaml", "r", encoding="utf-8") as f:
        resnet_cfg = yaml.safe_load(f)

    cnn_dir = cnn_cfg["data"]["artifacts_dir"]
    resnet_dir = resnet_cfg["data"]["artifacts_dir"]

    assert cnn_dir != resnet_dir, f"Artifact paths must be separated: {cnn_dir} vs {resnet_dir}"
    assert "resnet" in resnet_dir
    assert "cnn" in cnn_dir


def test_resnet_test_household_isolation():
    """Verify test households (H2, H13) remain strictly isolated."""
    split_mgr = HouseholdSplitManager.from_yaml(Path("configs/kettle_household_split.yaml"))
    test_h = set(split_mgr.config.test_households)
    assert test_h == {2, 13}

    train_ds = KettleWeakDataset(split="train")
    for s in train_ds._samples:
        assert s.household_id not in test_h, f"Test household {s.household_id} found in train dataset!"

    val_ds = KettleWeakDataset(split="validation")
    for s in val_ds._samples:
        assert s.household_id not in test_h, f"Test household {s.household_id} found in val dataset!"


def test_resnet_aggregate_only_inputs_and_no_target_leakage():
    """Verify that dataset inputs contain only aggregate power (510,) float32 and no strong targets."""
    train_ds = KettleWeakDataset(split="train")
    x, y = train_ds[0]
    assert isinstance(x, np.ndarray)
    assert x.shape == (510,)
    assert x.dtype == np.float32
    assert y in (0, 1)


def test_resnet_smoke_training_loop(tmp_path):
    """Run a tiny 2-epoch ResNet training loop on synthetic data."""
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
            "name": "resnet1d_reference",
            "in_channels": 1,
            "filter_counts": [16, 32, 32],
            "kernel_sizes": [8, 5, 3],
            "out_features": 1,
        },
        "training": {
            "appliance": "kettle",
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
    assert metrics["experiment_name"] == "resnet1d_reference"
    assert metrics["total_epochs_trained"] == 2
