"""Tests for Experiment 3 Multi-Scale ResNet Ensemble Training and Pipeline Wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
import yaml

from scripts.train_e3_ensemble import (
    build_resnet_model,
    train_single_resnet_branch,
    evaluate_ensemble_localization_on_validation,
    run_smoke_test,
)
from src.data.dataset import KettleWeakDataset, NormalizationTransform
from src.models.localization import (
    localize_ensemble_window,
    threshold_activation,
    apply_attention_sigmoid,
)
from src.models.resnet import ResNet1D


def test_build_all_five_resnet_branches() -> None:
    """Verify all 5 ensemble branches {k in 5, 7, 9, 15, 25} can be constructed."""
    kernel_sizes = [5, 7, 9, 15, 25]
    for k in kernel_sizes:
        model = build_resnet_model(
            kernel_size=k,
            filter_counts=(64, 128, 128),
            in_channels=1,
            out_features=1,
        )
        assert isinstance(model, ResNet1D)
        # Verify first conv kernel size in block1
        assert model.block1.conv1.kernel_size == (k,)


def test_ensemble_forward_and_cam_extraction_shapes() -> None:
    """Verify batch forward pass and CAM extraction across all 5 branches."""
    batch_size = 4
    x_dummy = torch.randn(batch_size, 510)
    kernel_sizes = [5, 7, 9, 15, 25]

    for k in kernel_sizes:
        model = build_resnet_model(kernel_size=k)
        model.eval()
        with torch.no_grad():
            logit = model(x_dummy)
            assert logit.shape == (batch_size, 1)

            cam = model.extract_cam(x_dummy)
            assert cam.shape == (batch_size, 510)


def test_e3_config_loading_and_structure() -> None:
    """Verify configs/e3_ensemble.yaml loads cleanly and matches locked specifications."""
    config_path = Path("configs/e3_ensemble.yaml")
    assert config_path.exists(), "configs/e3_ensemble.yaml does not exist"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Check key configuration attributes
    assert config["models"]["ensemble_kernels"] == [5, 7, 9, 15, 25]
    assert config["training"]["checkpoint_policy"] == "fixed_epoch"
    assert config["training"]["max_epochs"] == 10
    assert config["training"]["learning_rate"] == 0.001
    assert config["training"]["optimizer"] == "Adam"
    assert config["training"]["batch_size"] == 256
    assert config["training"]["fixed_threshold"] == 0.50
    assert config["localization"]["detection_threshold"] == 0.50
    assert config["localization"]["loc_threshold"] == 0.50


def test_train_single_branch_fixed_epoch(tmp_path: Path) -> None:
    """Verify single branch trains for exactly the requested fixed epochs."""
    train_dataset = KettleWeakDataset(split="train")
    dummy_train = np.random.uniform(50.0, 3000.0, size=(32, 510)).astype(np.float32)
    train_dataset.register_household_array(1, dummy_train)
    train_dataset._samples = train_dataset._samples[:32]

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    model, hist = train_single_resnet_branch(
        kernel_size=5,
        train_loader=loader,
        max_epochs=2,
        learning_rate=0.001,
        weight_decay=0.0,
        device=torch.device("cpu"),
        seed=42,
        filter_counts=(16, 32, 32),
    )

    assert len(hist) == 2
    assert hist[0]["epoch"] == 1
    assert hist[1]["epoch"] == 2
    assert "train_loss" in hist[0]
    assert hist[0]["train_loss"] > 0.0


def test_e3_smoke_test_end_to_end(tmp_path: Path) -> None:
    """Verify complete smoke test execution producing 5 checkpoints and report."""
    config_path = Path("configs/e3_ensemble.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_smoke_test(config, tmp_path)

    # Check 5 checkpoints exist
    for k in [5, 7, 9, 15, 25]:
        ckpt_p = tmp_path / f"checkpoint_k{k}.pt"
        assert ckpt_p.exists(), f"Missing checkpoint: {ckpt_p}"
        ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        assert ckpt["kernel_size"] == k
        assert "model_state_dict" in ckpt
        assert "normalization" in ckpt

    # Check Markdown report generated
    report_p = tmp_path / "e3_report.md"
    assert report_p.exists()
    report_text = report_p.read_text(encoding="utf-8")
    assert "Experiment 3: Multi-Scale ResNet Ensemble" in report_text
    assert "Zero-Power Deterministic Suppression" in report_text
