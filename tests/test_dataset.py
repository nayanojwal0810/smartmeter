"""Unit tests for KettleWeakDataset and model-ready dataset layer."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.data.dataset import KettleWeakDataset, NormalizationTransform
from src.data.split import HouseholdSplitManager


def test_dataset_sample_counts_by_split() -> None:
    """Verify exact sample counts matching the locked primary household split (Decision D-007)."""
    train_ds = KettleWeakDataset(split="train")
    val_ds = KettleWeakDataset(split="validation")
    test_ds = KettleWeakDataset(split="test")

    assert len(train_ds) == 141003
    assert len(val_ds) == 18561
    assert len(test_ds) == 14508
    assert len(train_ds) + len(val_ds) + len(test_ds) == 174072


def test_input_shape_and_dtype() -> None:
    """Verify model input tensor shape is strictly (510,) and dtype is float32."""
    train_ds = KettleWeakDataset(split="train")

    # Register mock array for House 1
    mock_h1 = np.ones((10388, 510), dtype=np.float32) * 500.0
    train_ds.register_household_array(household_id=1, array=mock_h1)

    x, y = train_ds[0]

    assert isinstance(x, np.ndarray)
    assert x.shape == (510,)
    assert x.dtype == np.float32
    assert x[0] == 500.0
    # House 1 is unmonitored -> weak label 0
    assert y == 0


def test_weak_label_attachment_and_h12_unmonitored_handling() -> None:
    """Verify weak household label attachment for positive, unmonitored, and H12 houses."""
    train_ds = KettleWeakDataset(split="train")

    # Check unmonitored house (H1) -> weak label 0
    meta_h1 = train_ds.get_metadata(0)
    assert meta_h1["household_id"] == 1
    assert meta_h1["weak_label"] == 0

    # Check positive presence house (H3)
    # H1 has 10,388 windows, so index 10388 is first window of H3
    meta_h3 = train_ds.get_metadata(10388)
    assert meta_h3["household_id"] == 3
    assert meta_h3["weak_label"] == 1

    # Check H12 (positive presence weak label = 1)
    # Find index where household_id == 12
    h12_idx = next(i for i in range(len(train_ds)) if train_ds.get_metadata(i)["household_id"] == 12)
    meta_h12 = train_ds.get_metadata(h12_idx)
    assert meta_h12["household_id"] == 12
    assert meta_h12["weak_label"] == 1


def test_target_separation_in_model_inputs() -> None:
    """Verify that model input tensor x contains ONLY aggregate power and zero target fields."""
    val_ds = KettleWeakDataset(split="validation")
    mock_h4 = np.full((10208, 510), 1200.0, dtype=np.float32)
    val_ds.register_household_array(household_id=4, array=mock_h4)

    x, y = val_ds[0]

    # Verify input tensor is 1D array of length 510
    assert x.shape == (510,)
    # Verify tuple contains only (x, weak_label)
    assert isinstance(y, int)
    assert y in (0, 1)


def test_test_split_isolation() -> None:
    """Verify holdout test households (H2, H13) cannot enter train or validation datasets."""
    train_ds = KettleWeakDataset(split="train")
    val_ds = KettleWeakDataset(split="validation")
    test_ds = KettleWeakDataset(split="test")

    train_houses = {train_ds.get_metadata(i)["household_id"] for i in range(0, len(train_ds), 5000)}
    val_houses = {val_ds.get_metadata(i)["household_id"] for i in range(0, len(val_ds), 2000)}
    test_houses = {test_ds.get_metadata(i)["household_id"] for i in range(0, len(test_ds), 2000)}

    assert 2 not in train_houses
    assert 13 not in train_houses
    assert 2 not in val_houses
    assert 13 not in val_houses

    assert 2 in test_houses
    assert 13 in test_houses


def test_normalization_transform_hook() -> None:
    """Verify train-derived normalization transform applies cleanly."""
    train_mean = 500.0
    train_std = 250.0

    transform = NormalizationTransform(mean=train_mean, std=train_std, method="standardize")
    train_ds = KettleWeakDataset(split="train", transform=transform)

    mock_data = np.full((10, 510), 750.0, dtype=np.float32)
    train_ds.register_household_array(household_id=1, array=mock_data)

    x, y = train_ds[0]

    # (750 - 500) / 250 = 1.0
    np.testing.assert_allclose(x, np.ones(510, dtype=np.float32), rtol=1e-5)


def test_validation_sanity_check_single_class() -> None:
    """Verify validation dataset weak-label is single-class and manifest documents this."""
    spec_path = Path("artifacts/manifests/kettle_model_dataset.json")
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        val_summary = spec["split_summary"]["validation"]
        assert val_summary["weak_validation_is_single_class"] is True
        assert val_summary["weak_positive_samples"] == 18561
        assert val_summary["weak_negative_samples"] == 0

        val_cov = val_summary["strong_evaluation_target_coverage"]
        assert val_cov["strong_active_target_windows"] == 3698
        assert val_cov["strong_inactive_target_windows"] == 14863
        assert val_cov["both_target_classes_present_for_evaluation"] is True


def test_mmap_array_writable_slice_regression(tmp_path: Path) -> None:
    """Regression test: verify that slicing read-only memory-mapped arrays yields writable tensors without warnings."""
    import warnings
    import torch

    # Create dummy read-only memory-mapped npy file
    npy_file = tmp_path / "house_1_aggregate.npy"
    data = np.full((10, 510), 1234.0, dtype=np.float32)
    np.save(npy_file, data)

    # Load in read-only mmap mode
    mmap_arr = np.load(npy_file, mmap_mode="r")
    assert mmap_arr.flags.writeable is False, "mmap_mode='r' array must be non-writable"

    train_ds = KettleWeakDataset(split="train")
    train_ds.register_household_array(household_id=1, array=mmap_arr)

    # Fetch sample
    x, y = train_ds[0]

    # Assert single slice returned is writable
    assert x.flags.writeable is True, "Dataset sample must be writable"

    # Assert torch conversion emits zero UserWarnings about non-writable arrays
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        tensor = torch.from_numpy(x)
        assert tensor.shape == (510,)
        for w in recorded_warnings:
            assert "not writable" not in str(w.message), f"Unexpected non-writable warning: {w.message}"

