"""Model-Ready Weakly Supervised Dataset Layer for REFIT Aggregate Power.

Provides memory-mapped, random-accessible, and split-aware dataset loading for
weakly supervised 1D CNN / ResNet training under Decision D-007:
- Model input: aggregate_w[510], shape (510,), dtype float32.
- Training target: weak_kettle_presence ∈ {0, 1} (household metadata label).
- Strict target separation: strong sub-meter Kettle measurements and evaluation targets
  are strictly excluded from training tensors.
- Leakage protection: test split (H2, H13) is isolated and forbidden during training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.data.split import HouseholdSplitConfig, HouseholdSplitManager


@dataclass
class DatasetSample:
    """Metadata container for a single dataset sample (bookkeeping only)."""

    window_id: int
    household_id: int
    segment_id: int
    split: str
    weak_label: int  # 1 = indicated presence, 0 = unmonitored


class NormalizationTransform:
    """Train-derived normalization transform hook for aggregate power windows."""

    def __init__(
        self,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        method: str = "none",
    ) -> None:
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.method = method

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply train-fitted normalization to input window tensor."""
        x_out = np.asarray(x, dtype=np.float32)
        if self.method == "standardize" and self.mean is not None and self.std is not None:
            eps = 1e-8
            return (x_out - self.mean) / (self.std + eps)
        elif self.method == "minmax" and self.min_val is not None and self.max_val is not None:
            eps = 1e-8
            return (x_out - self.min_val) / (self.max_val - self.min_val + eps)
        return x_out


class KettleWeakDataset:
    """Memory-mapped, split-aware weakly supervised dataset for 510-point windows.

    Compatible with standard PyTorch DataLoader and NumPy batching pipelines without
    loading the full dataset into RAM.
    """

    def __init__(
        self,
        split: str,
        data_dir: Path = Path("data/processed/kettle"),
        split_config_path: Path = Path("configs/kettle_household_split.yaml"),
        window_manifest_path: Path = Path("artifacts/manifests/kettle_household_split.json"),
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'validation', or 'test'.")

        self.split = split
        self.data_dir = Path(data_dir)
        self.transform = transform

        self.split_manager = (
            HouseholdSplitManager.from_yaml(split_config_path)
            if Path(split_config_path).exists()
            else HouseholdSplitManager()
        )

        # Build index of samples belonging to this split
        self._samples: List[DatasetSample] = []
        self._house_arrays: Dict[int, np.ndarray] = {}
        self._house_window_offsets: Dict[int, int] = {}

        self._load_split_index(window_manifest_path)

    def _load_split_index(self, manifest_path: Path) -> None:
        """Load structural sample index for this split from split manifest or memory map."""
        if not manifest_path.exists():
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        hh_list = manifest_data.get("households", [])
        global_win_id = 1

        for h in hh_list:
            h_id = h["household_id"]
            h_split = h["split"]
            weak_lbl = h["weak_label"]
            n_win = h["total_510_windows"]

            if h_split == self.split:
                self._house_window_offsets[h_id] = len(self._samples)
                for w_idx in range(n_win):
                    self._samples.append(
                        DatasetSample(
                            window_id=global_win_id + w_idx,
                            household_id=h_id,
                            segment_id=0,  # Bookkeeping
                            split=h_split,
                            weak_label=weak_lbl,
                        )
                    )

            global_win_id += n_win

    def register_household_array(self, household_id: int, array: np.ndarray) -> None:
        """Register in-memory or memory-mapped numpy array (N, 510) for a household."""
        if array.ndim != 2 or array.shape[1] != 510:
            raise ValueError(f"Array for House {household_id} must have shape (N, 510), got {array.shape}")
        self._house_arrays[household_id] = array.astype(np.float32, copy=False)

    def load_mmap_arrays(self) -> None:
        """Attempt to memory-map binary .npy files for all households in this split."""
        target_houses = (
            self.split_manager.config.train_households
            if self.split == "train"
            else self.split_manager.config.validation_households
            if self.split == "validation"
            else self.split_manager.config.test_households
        )

        for h_id in target_houses:
            npy_path = self.data_dir / f"house_{h_id}_aggregate.npy"
            if npy_path.exists():
                mmap_arr = np.load(npy_path, mmap_mode="r")
                self.register_household_array(h_id, mmap_arr)

    def __len__(self) -> int:
        """Return total number of windows in this split."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get (aggregate_tensor[510], weak_label) for model input.

        Guarantees:
        - Input shape is strictly (510,) dtype float32.
        - Contains ONLY aggregate power.
        - Zero sub-meter measurements or strong target metadata.
        """
        if idx < 0 or idx >= len(self._samples):
            raise IndexError(f"Index {idx} out of range for {self.split} dataset with {len(self._samples)} samples.")

        sample = self._samples[idx]
        h_id = sample.household_id
        weak_label = sample.weak_label

        if h_id in self._house_arrays:
            offset = self._house_window_offsets[h_id]
            local_idx = idx - offset
            raw_window = np.array(self._house_arrays[h_id][local_idx], copy=True, dtype=np.float32)
        else:
            # Placeholder/fallback when binary arrays are not yet materialized on disk
            raw_window = np.zeros(510, dtype=np.float32)

        # Apply optional normalization transform
        if self.transform is not None:
            x_tensor = self.transform(raw_window)
        else:
            x_tensor = raw_window

        return x_tensor, weak_label

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Return structural bookkeeping metadata for a sample (separated from model tensor)."""
        sample = self._samples[idx]
        return {
            "window_id": sample.window_id,
            "household_id": sample.household_id,
            "split": sample.split,
            "weak_label": sample.weak_label,
        }
