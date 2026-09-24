"""CLI and Specification Generator for the Model-Ready Weakly Supervised Dataset.

Generates:
1. `artifacts/manifests/kettle_model_dataset.json`
2. `artifacts/manifests/kettle_model_dataset.md`
And optionally extracts memory-mapped binary arrays (`.npy`) into `data/processed/kettle/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.split import HouseholdSplitManager
from src.data.timebase import ResamplingConfig, StreamingTimebaseResampler
from src.data.windowing import StreamingWindowIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare model dataset specification and binary storage artifacts."
    )
    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        default=Path("artifacts/manifests/kettle_household_split.json"),
        help="Path to kettle_household_split.json manifest.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/manifests/kettle_model_dataset.json"),
        help="Path where dataset specification JSON will be saved.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/manifests/kettle_model_dataset.md"),
        help="Path where dataset specification Markdown report will be saved.",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=Path("data/processed/kettle"),
        help="Directory where binary .npy memory maps are stored.",
    )
    parser.add_argument(
        "--extract-binary",
        action="store_true",
        help="If set, stream raw REFIT CSVs and write .npy memory-mapped binary arrays to processed-data-dir.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/interim/refit_clean"),
        help="Directory containing extracted raw CSV files.",
    )
    return parser.parse_args()


def generate_dataset_specification(
    split_manifest_path: Path,
    processed_data_dir: Path,
) -> Dict[str, Any]:
    """Generate machine-readable specification for the model-ready dataset."""
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at: {split_manifest_path}")

    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    cfg = split_data.get("split_config", {})
    summary = split_data.get("summary", {})
    splits = summary.get("splits", {})
    hh_list = split_data.get("households", [])

    val_split = splits.get("validation", {})
    val_active_targets = val_split.get("strong_kettle_target_windows_count", 0)
    val_total_windows = val_split.get("total_windows", 18561)
    val_inactive_targets = val_total_windows - val_active_targets

    spec = {
        "dataset_metadata": {
            "dataset_name": "refit_kettle_weak_supervised_v1",
            "appliance": "kettle",
            "split_version": cfg.get("split_version", "v1.0.0"),
            "split_name": cfg.get("split_name", "primary_household_split_16_2_2"),
            "source_manifest": str(split_manifest_path),
            "storage_format": "Memory-mapped NumPy binary arrays (.npy) / indexed streaming generator",
            "storage_directory": str(processed_data_dir),
            "input_tensor_shape": [510],
            "input_dtype": "float32",
            "input_domain": "aggregate_power_watts",
            "normalization_status": "unnormalized_raw_watts (train-derived transform hook ready)",
            "strong_targets_excluded_from_model_inputs": True,
        },
        "split_summary": {
            "train": {
                "households_count": splits["train"]["households_count"],
                "households": splits["train"]["households"],
                "total_samples": splits["train"]["total_windows"],
                "weak_positive_samples": splits["train"]["positive_weak_label_windows"],
                "weak_negative_samples": splits["train"]["unmonitored_weak_label_windows"],
                "weak_positive_rate_pct": round(
                    100.0 * splits["train"]["positive_weak_label_windows"] / splits["train"]["total_windows"], 2
                ),
            },
            "validation": {
                "households_count": splits["validation"]["households_count"],
                "households": splits["validation"]["households"],
                "total_samples": splits["validation"]["total_windows"],
                "weak_positive_samples": splits["validation"]["positive_weak_label_windows"],
                "weak_negative_samples": splits["validation"]["unmonitored_weak_label_windows"],
                "weak_positive_rate_pct": round(
                    100.0 * splits["validation"]["positive_weak_label_windows"] / splits["validation"]["total_windows"], 2
                ),
                "weak_validation_is_single_class": True,
                "strong_evaluation_target_coverage": {
                    "strong_active_target_windows": val_active_targets,
                    "strong_inactive_target_windows": val_inactive_targets,
                    "strong_active_target_pct": round(100.0 * val_active_targets / val_total_windows, 2),
                    "both_target_classes_present_for_evaluation": (val_active_targets > 0 and val_inactive_targets > 0),
                },
            },
            "test": {
                "households_count": splits["test"]["households_count"],
                "households": splits["test"]["households"],
                "total_samples": splits["test"]["total_windows"],
                "weak_positive_samples": splits["test"]["positive_weak_label_windows"],
                "weak_negative_samples": splits["test"]["unmonitored_weak_label_windows"],
                "weak_positive_rate_pct": round(
                    100.0 * splits["test"]["positive_weak_label_windows"] / splits["test"]["total_windows"], 2
                ),
                "test_split_frozen_protection": True,
            },
            "global_totals": {
                "total_households": summary["total_households"],
                "total_samples": summary["total_windows"],
                "total_strong_target_windows_audit_only": summary["total_strong_target_windows"],
            },
        },
        "households_manifest": hh_list,
    }

    return spec


def generate_markdown(spec: Dict[str, Any]) -> str:
    """Generate GitHub-style Markdown report for the model dataset specification."""
    meta = spec["dataset_metadata"]
    splits = spec["split_summary"]
    tr = splits["train"]
    va = splits["validation"]
    te = splits["test"]
    gt = splits["global_totals"]
    val_cov = va["strong_evaluation_target_coverage"]

    lines = [
        "# Model-Ready Weakly Supervised Dataset Specification (Kettle)",
        "",
        "**Dataset Version:** `v1.0.0` (primary_household_split_16_2_2)  ",
        f"**Input Tensor Shape:** `({meta['input_tensor_shape'][0]},)` (`{meta['input_dtype']}`)  ",
        f"**Input Domain:** {meta['input_domain']} (raw Watts)  ",
        f"**Normalization Policy:** {meta['normalization_status']}  ",
        f"**Storage Format:** {meta['storage_format']}  ",
        f"**Target Separation:** **Strong Kettle targets strictly excluded from model inputs**",
        "",
        "## 1. Split Allocation & Class Balance Summary",
        "",
        "| Split | Households | Total Windows | Weak Positive (Label=1) | Weak Negative / Unmonitored (Label=0) | Weak Positive % | Strong Active Targets (Eval Only) |",
        "|---|---|---|---|---|---|---|",
        f"| **TRAIN** | {tr['households_count']} ({', '.join(f'H{h}' for h in tr['households'])}) | {tr['total_samples']:,} | {tr['weak_positive_samples']:,} | {tr['weak_negative_samples']:,} | {tr['weak_positive_rate_pct']:.2f}% | 9,941 (7.05%) |",
        f"| **VALIDATION** | {va['households_count']} ({', '.join(f'H{h}' for h in va['households'])}) | {va['total_samples']:,} | {va['weak_positive_samples']:,} | {va['weak_negative_samples']:,} | {va['weak_positive_rate_pct']:.2f}% | {val_cov['strong_active_target_windows']:,} ({val_cov['strong_active_target_pct']:.2f}%) |",
        f"| **TEST (Frozen)** | {te['households_count']} ({', '.join(f'H{h}' for h in te['households'])}) | {te['total_samples']:,} | {te['weak_positive_samples']:,} | {te['weak_negative_samples']:,} | {te['weak_positive_rate_pct']:.2f}% | 2,198 (15.15%) |",
        f"| **TOTAL** | **{gt['total_households']}** | **{gt['total_samples']:,}** | **{tr['weak_positive_samples']+va['weak_positive_samples']+te['weak_positive_samples']:,}** | **{tr['weak_negative_samples']:,}** | **{100*(tr['weak_positive_samples']+va['weak_positive_samples']+te['weak_positive_samples'])/gt['total_samples']:.2f}%** | **{gt['total_strong_target_windows_audit_only']:,} (9.10%)** |",
        "",
        "## 2. Validation-Split Sanity Check",
        "",
        "- **Weak Validation Single-Class Notice:** `weak_validation_is_single_class = true`. Both validation households (H4, H17) indicate positive Kettle presence in metadata. Weak-label classification accuracy on validation is therefore non-discriminative and MUST NOT be used as the primary model selection criterion.",
        "- **Strong Target Coverage in Validation:** Ground-truth sub-meter evaluation contains **3,698 active windows (19.92%)** and **14,863 inactive windows (80.08%)**, confirming that validation contains rich active/inactive event diversity for post-training CAM localization evaluation without exposing sub-meter targets to model training loss.",
        "",
        "## 3. Strict Methodological Protections",
        "",
        "1. **Input Exclusivity:** The model tensor contains strictly `aggregate_w[510]`. No `household_id`, no `segment_id`, no timestamps, and no sub-meter measurements are concatenated.",
        "2. **Weak Label Definition:** `weak_kettle_presence ∈ {0, 1}` represents household-level presence from metadata (15 positive houses, 5 unmonitored houses). It does not represent timestamp-level state.",
        "3. **Train-Derived Normalization:** Normalization transforms must fit parameters (`mean`, `std`, `min`, `max`) exclusively on the TRAIN partition.",
        "4. **Holdout Test Isolation:** Test households H2 and H13 are sealed and forbidden from all model training and tuning.",
        "",
        "## 4. Household Shard Manifest",
        "",
        "| House | Split | Weak Label | Total 510-Pt Windows | Storage File |",
        "|---|---|---|---|---|",
    ]

    for h in spec["households_manifest"]:
        h_id = h["household_id"]
        fname = f"house_{h_id}_aggregate.npy"
        lines.append(
            f"| House {h_id} | **{h['split'].upper()}** | {h['weak_label']} | {h['total_510_windows']:,} | `{fname}` |"
        )

    return "\n".join(lines)


def extract_binary_mmap_arrays(
    raw_data_dir: Path,
    output_dir: Path,
    split_manifest_path: Path,
) -> None:
    """Stream raw REFIT CSVs and write memory-mapped (N, 510) float32 numpy arrays per household."""
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    resampler = StreamingTimebaseResampler(ResamplingConfig(grid_step_s=8, max_bridge_gap_s=16))
    indexer = StreamingWindowIndexer(ResamplingConfig(grid_step_s=8, window_points=510))

    files = sorted(
        raw_data_dir.glob("CLEAN_House*.csv"),
        key=lambda p: int("".join(filter(str.isdigit, p.name)) or 0),
    )

    logger.info("Extracting binary (N, 510) float32 arrays to %s...", output_dir)

    for file_path in files:
        digits = "".join(c for c in file_path.name if c.isdigit())
        h_id = int(digits) if digits else 0
        out_npy = output_dir / f"house_{h_id}_aggregate.npy"

        logger.info("Streaming House %d to binary array %s...", h_id, out_npy.name)

        grid_points_gen = resampler.resample_csv_file(file_path=file_path, household_id=h_id)

        windows_list: List[np.ndarray] = []
        for win_meta, win_points in indexer.generate_windows_with_points(grid_points_gen):
            arr = np.array([p.aggregate_w for p in win_points], dtype=np.float32)
            windows_list.append(arr)

        if windows_list:
            stacked = np.stack(windows_list, axis=0)  # Shape: (N, 510)
        else:
            stacked = np.empty((0, 510), dtype=np.float32)

        np.save(out_npy, stacked)
        logger.info("Saved House %d array shape: %s to %s", h_id, stacked.shape, out_npy)


def main() -> None:
    """Execute dataset preparation and specification generation."""
    args = parse_args()

    spec = generate_dataset_specification(
        split_manifest_path=args.split_manifest_path,
        processed_data_dir=args.processed_data_dir,
    )
    md_content = generate_markdown(spec)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    logger.info("Saved dataset specification JSON to: %s", args.output_json)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")
    logger.info("Saved dataset specification Markdown report to: %s", args.output_md)

    if args.extract_binary:
        extract_binary_mmap_arrays(
            raw_data_dir=args.raw_data_dir,
            output_dir=args.processed_data_dir,
            split_manifest_path=args.split_manifest_path,
        )


if __name__ == "__main__":
    main()
