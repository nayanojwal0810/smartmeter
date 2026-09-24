"""Household Split and Weak-Supervision Label Management Module.

Enforces strict household-level isolation under Decision D-007:
- Primary split: 16 Train / 2 Validation / 2 Test.
- All windows from a household inherit the household's split assignment.
- Zero random window splitting, zero segment splitting, zero timestamp splitting.
- Weak household labels (1 = indicated presence, 0 = unmonitored) are separate from
  strong timestamp-level evaluation targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import yaml
except ImportError:
    yaml = None  # Fallback manual parser if yaml not installed


ALL_REFIT_HOUSEHOLDS: Set[int] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 15, 16, 17, 18, 19, 20, 21
}


@dataclass(frozen=True)
class HouseholdSplitConfig:
    """Immutable representation of the primary household train/val/test split."""

    appliance: str
    split_version: str
    split_name: str
    train_households: tuple[int, ...]
    validation_households: tuple[int, ...]
    test_households: tuple[int, ...]
    positive_households: tuple[int, ...]
    unmonitored_households: tuple[int, ...]
    rationale: str

    def __post_init__(self) -> None:
        """Validate disjointness, completeness, and integrity upon instantiation."""
        train_set = set(self.train_households)
        val_set = set(self.validation_households)
        test_set = set(self.test_households)

        # 1. Verify exact counts
        if len(train_set) != len(self.train_households):
            raise ValueError("Duplicate household IDs found in train split.")
        if len(val_set) != len(self.validation_households):
            raise ValueError("Duplicate household IDs found in validation split.")
        if len(test_set) != len(self.test_households):
            raise ValueError("Duplicate household IDs found in test split.")

        # 2. Verify mutual disjointness
        if train_set & val_set:
            raise ValueError(f"Overlap between train and validation splits: {train_set & val_set}")
        if train_set & test_set:
            raise ValueError(f"Overlap between train and test splits: {train_set & test_set}")
        if val_set & test_set:
            raise ValueError(f"Overlap between validation and test splits: {val_set & test_set}")

        # 3. Verify total coverage of all 20 REFIT households
        all_assigned = train_set | val_set | test_set
        if all_assigned != ALL_REFIT_HOUSEHOLDS:
            missing = ALL_REFIT_HOUSEHOLDS - all_assigned
            unknown = all_assigned - ALL_REFIT_HOUSEHOLDS
            err_msg = []
            if missing:
                err_msg.append(f"Missing households: {sorted(missing)}")
            if unknown:
                err_msg.append(f"Unknown households: {sorted(unknown)}")
            raise ValueError(f"Household split coverage error: {'; '.join(err_msg)}")

        # 4. Verify weak label sets
        pos_set = set(self.positive_households)
        unmon_set = set(self.unmonitored_households)
        if pos_set & unmon_set:
            raise ValueError(f"Overlap between positive and unmonitored sets: {pos_set & unmon_set}")
        if pos_set | unmon_set != ALL_REFIT_HOUSEHOLDS:
            raise ValueError("Weak label sets do not cover all 20 REFIT households.")


class HouseholdSplitManager:
    """Manages household-to-split mappings and weak presence labels."""

    def __init__(self, config: Optional[HouseholdSplitConfig] = None) -> None:
        self.config = config or self.load_default_config()

        # Build fast lookup tables
        self._household_to_split: Dict[int, str] = {}
        for h in self.config.train_households:
            self._household_to_split[h] = "train"
        for h in self.config.validation_households:
            self._household_to_split[h] = "validation"
        for h in self.config.test_households:
            self._household_to_split[h] = "test"

        self._household_to_weak_label: Dict[int, int] = {}
        pos_set = set(self.config.positive_households)
        for h in ALL_REFIT_HOUSEHOLDS:
            self._household_to_weak_label[h] = 1 if h in pos_set else 0

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> HouseholdSplitConfig:
        """Load and parse HouseholdSplitConfig from YAML file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Split config not found at: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            if yaml is not None:
                data = yaml.safe_load(f)
            else:
                # Basic line parser fallback if PyYAML is not installed
                import json
                text = f.read()
                # If PyYAML missing, convert simple YAML structure
                data = cls._parse_simple_yaml(text)

        return HouseholdSplitConfig(
            appliance=data["appliance"],
            split_version=data["split_version"],
            split_name=data["split_name"],
            train_households=tuple(int(x) for x in data["train_households"]),
            validation_households=tuple(int(x) for x in data["validation_households"]),
            test_households=tuple(int(x) for x in data["test_households"]),
            positive_households=tuple(int(x) for x in data["positive_households"]),
            unmonitored_households=tuple(int(x) for x in data["unmonitored_households"]),
            rationale=data.get("rationale", "").strip(),
        )

    @classmethod
    def _parse_simple_yaml(cls, text: str) -> Dict[str, Any]:
        """Minimal YAML parser fallback for simple key-value and list configs."""
        res: Dict[str, Any] = {}
        current_list_key = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("- ") and current_list_key:
                val = line[2:].strip().strip("\"'")
                res[current_list_key].append(val)
            elif ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().strip("\"'")
                if not val or val == ">":
                    res[key] = []
                    current_list_key = key
                else:
                    res[key] = val
                    current_list_key = None
        return res

    @classmethod
    def load_default_config(cls) -> HouseholdSplitConfig:
        """Load default frozen 16/2/2 split configuration."""
        default_yaml = Path("configs/kettle_household_split.yaml")
        if default_yaml.exists():
            return cls.load_from_yaml(default_yaml)

        # Fallback hardcoded definition matching locked Decision D-007
        return HouseholdSplitConfig(
            appliance="kettle",
            split_version="v1.0.0",
            split_name="primary_household_split_16_2_2",
            train_households=(1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21),
            validation_households=(4, 17),
            test_households=(2, 13),
            positive_households=(2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 17, 19, 20, 21),
            unmonitored_households=(1, 10, 15, 16, 18),
            rationale="16 Train / 2 Validation / 2 Test primary household isolation split.",
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> HouseholdSplitManager:
        """Instantiate HouseholdSplitManager directly from a YAML split configuration file."""
        config = cls.load_from_yaml(yaml_path)
        return cls(config=config)

    @classmethod
    def load_default(cls) -> HouseholdSplitManager:
        """Create HouseholdSplitManager with default frozen 16/2/2 configuration."""
        cfg = cls.load_default_config()
        return cls(config=cfg)

    def get_split(self, household_id: int) -> str:
        """Return split name ('train', 'validation', 'test') for a given household ID."""
        if household_id not in self._household_to_split:
            raise KeyError(f"Unknown household ID: {household_id}. Valid: {sorted(ALL_REFIT_HOUSEHOLDS)}")
        return self._household_to_split[household_id]

    def get_weak_label(self, household_id: int) -> int:
        """Return binary weak presence label (1 = indicated presence, 0 = unmonitored)."""
        if household_id not in self._household_to_weak_label:
            raise KeyError(f"Unknown household ID: {household_id}. Valid: {sorted(ALL_REFIT_HOUSEHOLDS)}")
        return self._household_to_weak_label[household_id]

    def is_test_household(self, household_id: int) -> bool:
        """Check if household belongs to frozen holdout test split."""
        return self.get_split(household_id) == "test"

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of split metadata."""
        return {
            "appliance": self.config.appliance,
            "split_version": self.config.split_version,
            "split_name": self.config.split_name,
            "train_households": list(self.config.train_households),
            "validation_households": list(self.config.validation_households),
            "test_households": list(self.config.test_households),
            "positive_households": list(self.config.positive_households),
            "unmonitored_households": list(self.config.unmonitored_households),
            "rationale": self.config.rationale,
        }
