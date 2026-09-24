"""Unit tests for household split and weak-supervision label module (Decision D-007)."""

from __future__ import annotations

import pytest

from src.data.split import (
    ALL_REFIT_HOUSEHOLDS,
    HouseholdSplitConfig,
    HouseholdSplitManager,
)


def test_primary_household_split_membership() -> None:
    """Verify exact household allocation matching locked Decision D-007."""
    mgr = HouseholdSplitManager.load_default()
    cfg = mgr.config

    expected_train = (1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21)
    expected_val = (4, 17)
    expected_test = (2, 13)

    assert cfg.train_households == expected_train
    assert cfg.validation_households == expected_val
    assert cfg.test_households == expected_test

    assert len(cfg.train_households) == 16
    assert len(cfg.validation_households) == 2
    assert len(cfg.test_households) == 2


def test_split_disjointness_and_completeness() -> None:
    """Verify mutual disjointness and complete 20-household coverage."""
    mgr = HouseholdSplitManager.load_default()
    cfg = mgr.config

    train_s = set(cfg.train_households)
    val_s = set(cfg.validation_households)
    test_s = set(cfg.test_households)

    assert len(train_s & val_s) == 0
    assert len(train_s & test_s) == 0
    assert len(val_s & test_s) == 0

    assert (train_s | val_s | test_s) == ALL_REFIT_HOUSEHOLDS
    assert len(ALL_REFIT_HOUSEHOLDS) == 20


def test_household_to_split_mapping() -> None:
    """Verify get_split accurately maps each household."""
    mgr = HouseholdSplitManager()

    # Train houses
    for h in [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21]:
        assert mgr.get_split(h) == "train"
        assert mgr.is_test_household(h) is False

    # Validation houses
    for h in [4, 17]:
        assert mgr.get_split(h) == "validation"
        assert mgr.is_test_household(h) is False

    # Test houses
    for h in [2, 13]:
        assert mgr.get_split(h) == "test"
        assert mgr.is_test_household(h) is True

    # Unknown houses raise KeyError
    with pytest.raises(KeyError):
        mgr.get_split(14)
    with pytest.raises(KeyError):
        mgr.get_split(22)


def test_weak_label_vs_strong_target_separation() -> None:
    """Verify weak presence label is derived from metadata and separate from evaluation targets."""
    mgr = HouseholdSplitManager()

    # 15 positive presence households -> weak label 1
    for h in [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 17, 19, 20, 21]:
        assert mgr.get_weak_label(h) == 1

    # 5 unmonitored households -> weak label 0
    for h in [1, 10, 15, 16, 18]:
        assert mgr.get_weak_label(h) == 0


def test_h12_handling() -> None:
    """Verify H12 provides weak presence label (1) while remaining in train split."""
    mgr = HouseholdSplitManager()
    assert mgr.get_split(12) == "train"
    assert mgr.get_weak_label(12) == 1
    assert mgr.is_test_household(12) is False


def test_unmonitored_households_handling() -> None:
    """Verify all 5 unmonitored houses reside in train split with weak label 0."""
    mgr = HouseholdSplitManager()
    for h in [1, 10, 15, 16, 18]:
        assert mgr.get_split(h) == "train"
        assert mgr.get_weak_label(h) == 0


def test_invalid_split_configurations_rejected() -> None:
    """Verify configuration validation rejects overlaps, missing houses, or duplicates."""
    # 1. Overlapping train and val
    with pytest.raises(ValueError, match="Overlap"):
        HouseholdSplitConfig(
            appliance="kettle",
            split_version="v1",
            split_name="bad_split",
            train_households=(1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21),
            validation_households=(2, 4, 17),  # 2 in both
            test_households=(13,),
            positive_households=(2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 17, 19, 20, 21),
            unmonitored_households=(1, 10, 15, 16, 18),
            rationale="",
        )

    # 2. Missing household
    with pytest.raises(ValueError, match="Missing households"):
        HouseholdSplitConfig(
            appliance="kettle",
            split_version="v1",
            split_name="missing_split",
            train_households=(1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20),  # Missing 21
            validation_households=(4, 17),
            test_households=(2, 13),
            positive_households=(2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 17, 19, 20, 21),
            unmonitored_households=(1, 10, 15, 16, 18),
            rationale="",
        )
