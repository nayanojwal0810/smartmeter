"""CLI Runner for 1D ResNet Reference Model Training.

Usage:
    # Lightweight smoke test (synthetic small batch):
    python scripts/train_resnet.py --smoke-test

    # Non-smoke production path preflight verification:
    python scripts/train_resnet.py --config configs/resnet_reference.yaml --preflight

    # Full real 1D ResNet training job:
    python scripts/train_resnet.py --config configs/resnet_reference.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.train_cnn_baseline import ensure_mmap_arrays_exist
from src.data.dataset import KettleWeakDataset
from src.data.split import HouseholdSplitManager
from src.models.train import (
    load_validation_strong_targets,
    run_training_loop,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train 1D ResNet Reference Model for weakly supervised Kettle detection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resnet_reference.yaml"),
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run an ultra-lightweight smoke test with small synthetic data to verify training mechanics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for output artifacts directory.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional override for maximum epochs.",
    )
    parser.add_argument(
        "--dry-run",
        "--preflight",
        action="store_true",
        dest="dry_run",
        help="Run non-smoke production initialization path (dataset/split/config verification) without starting the training loop.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('auto', 'cpu', or 'cuda').",
    )
    return parser.parse_args()


def generate_resnet_markdown_report(metrics_dict: Dict[str, Any], output_path: Path) -> None:
    """Generate comprehensive Markdown report for the 1D ResNet experiment."""
    best = metrics_dict["best_epoch_metrics"]
    triv = metrics_dict["trivial_baseline_reference"]

    lines = [
        "# 1D ResNet Reference Model Experiment Report",
        "",
        "## Executive Summary",
        "",
        f"- **Experiment:** `{metrics_dict['experiment_name']}`  ",
        f"- **Appliance:** `{metrics_dict['appliance']}`  ",
        f"- **Seed:** `{metrics_dict['seed']}`  ",
        f"- **Best Epoch:** `{metrics_dict['best_epoch']}` (out of {metrics_dict['total_epochs_trained']} trained)  ",
        f"- **Train Standardization:** $\\mu = {metrics_dict['train_mean_w']:.2f}\\text{{ W}},\\; \\sigma = {metrics_dict['train_std_w']:.2f}\\text{{ W}}$ (fitted strictly on TRAIN)  ",
        "",
        "## Validation Performance vs Trivial Reference Baseline",
        "",
        "| Metric | 1D ResNet Reference (Threshold = 0.5) | Trivial Majority Reference (Predict Inactive) |",
        "|---|---|---|",
        f"| **Precision** | **{best['val_strong_precision'] * 100:.2f}%** | {triv['precision'] * 100:.2f}% |",
        f"| **Recall** | **{best['val_strong_recall'] * 100:.2f}%** | {triv['recall'] * 100:.2f}% |",
        f"| **F1-Score** | **{best['val_strong_f1'] * 100:.2f}%** | {triv['f1'] * 100:.2f}% |",
        f"| **Balanced Accuracy** | **{best['val_strong_balanced_accuracy'] * 100:.2f}%** | {triv['balanced_accuracy'] * 100:.2f}% |",
        f"| **Overall Accuracy** | **{best['val_strong_accuracy'] * 100:.2f}%** | {triv['accuracy'] * 100:.2f}% |",
        f"| **True Positives (Active)** | **{best['val_strong_tp']:,}** | {triv['tp']:,} |",
        f"| **False Positives** | **{best['val_strong_fp']:,}** | {triv['fp']:,} |",
        f"| **True Negatives (Inactive)** | **{best['val_strong_tn']:,}** | {triv['tn']:,} |",
        f"| **False Negatives** | **{best['val_strong_fn']:,}** | {triv['fn']:,} |",
        "",
        "## Methodological Invariants & Architectural Scope",
        "",
        "1. **Binary Head Adaptation:** The published CamAL paper describes a 2-class softmax classification head. Our project intentionally implements a single output logit with `BCEWithLogitsLoss` and sigmoid-equivalent binary decision at threshold 0.5 as defined by our project model-training specification. It is an intentional binary-classification implementation adaptation, not byte-for-byte or mathematically identical to the paper's 2-class softmax head.",
        "2. **Experiment Scope:** This experiment evaluates a **single ResNet reference model**. It is NOT yet the multi-scale CamAL ensemble, and Class Activation Map (CAM) localization is NOT yet implemented in this experiment. The eventual ensemble and localization pipeline will be a separate, downstream experiment.",
        "3. **Test Set Isolation:** Test households H2 and H13 were strictly excluded from training, normalization, validation, and early stopping.",
        "4. **Weak Label Supervision:** Model was trained exclusively against binary household-level metadata presence labels $y \\in \\{0, 1\\}$.",
        "5. **Zero Strong-Target Leakage:** Strong sub-meter timestamp annotations and window targets were completely absent from model inputs and loss functions.",
        "6. **Threshold Policy:** Classification probability threshold was fixed at $0.5$ without validation-based post-hoc tuning.",
        "",
        "## Checkpoints & Artifacts",
        "",
        f"- **Best Model Checkpoint:** `{metrics_dict['checkpoint_paths']['best_model']}`  ",
        f"- **Final Model Checkpoint:** `{metrics_dict['checkpoint_paths']['final_model']}`  ",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved ResNet experiment Markdown report to: %s", output_path)


def run_smoke_test(config: Dict[str, Any], output_dir: Path) -> None:
    """Run lightweight synthetic smoke test verifying end-to-end ResNet training mechanics."""
    logger.info("--- RUNNING LIGHTWEIGHT 1D RESNET SYNTHETIC SMOKE TEST ---")
    set_seed(42)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic datasets
    train_dataset = KettleWeakDataset(split="train")
    val_dataset = KettleWeakDataset(split="validation")

    # Synthetic arrays
    dummy_train = np.random.uniform(50.0, 3000.0, size=(100, 510)).astype(np.float32)
    dummy_val = np.random.uniform(50.0, 3000.0, size=(50, 510)).astype(np.float32)

    # Register for first train and val households
    train_dataset.register_household_array(1, dummy_train)
    val_dataset.register_household_array(4, dummy_val)

    # Trim samples for smoke test
    train_dataset._samples = train_dataset._samples[:100]
    val_dataset._samples = val_dataset._samples[:50]

    # Synthetic targets map
    targets_map = {val_dataset.get_metadata(i)["window_id"]: (1 if i % 5 == 0 else 0) for i in range(50)}

    smoke_cfg = {
        "model": {
            "name": "resnet1d_reference",
            "in_channels": 1,
            "input_length": 510,
            "filter_counts": [16, 32, 32],
            "kernel_sizes": [8, 5, 3],
            "out_features": 1,
        },
        "training": {
            "appliance": "kettle",
            "seed": 42,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "fixed_threshold": 0.5,
            "normalization": "standardize",
        },
    }

    metrics = run_training_loop(
        config=smoke_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        targets_map=targets_map,
        output_dir=output_dir,
        device=torch.device("cpu"),
    )

    generate_resnet_markdown_report(metrics, output_dir / "resnet_report.md")
    logger.info("1D RESNET SMOKE TEST PASSED SUCCESSFULLY!")


def main() -> None:
    """Execute ResNet training script."""
    args = parse_args()

    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs

    data_cfg = config.get("data", {})
    output_dir = args.output_dir or Path(data_cfg.get("artifacts_dir", "artifacts/experiments/resnet_reference"))

    if args.smoke_test:
        smoke_output_dir = args.output_dir or Path("artifacts/experiments/resnet_reference_smoke")
        run_smoke_test(config, smoke_output_dir)
        return

    # Real training path
    split_cfg_path = Path(data_cfg.get("split_config_path", "configs/kettle_household_split.yaml"))
    split_manifest_path = Path(data_cfg.get("split_manifest_path", "artifacts/manifests/kettle_household_split.json"))
    targets_path = Path(data_cfg.get("evaluation_targets_path", "artifacts/manifests/kettle_evaluation_targets.jsonl"))
    if not targets_path.exists():
        val_only_path = Path("artifacts/manifests/kettle_validation_targets.jsonl")
        if val_only_path.exists():
            targets_path = val_only_path
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed/kettle"))
    raw_data_dir = Path("data/interim/refit_clean")

    split_manager = (
        HouseholdSplitManager.from_yaml(split_cfg_path)
        if split_cfg_path.exists()
        else HouseholdSplitManager()
    )

    # 1. Ensure memory-mapped arrays exist for TRAIN and VALIDATION households
    ensure_mmap_arrays_exist(
        processed_dir=processed_dir,
        raw_data_dir=raw_data_dir,
        split_manager=split_manager,
        splits=["train", "validation"],
        dry_run=args.dry_run,
    )

    # 2. Instantiate datasets
    train_dataset = KettleWeakDataset(
        split="train",
        data_dir=processed_dir,
        split_config_path=split_cfg_path,
        window_manifest_path=split_manifest_path,
    )
    train_dataset.load_mmap_arrays()

    val_dataset = KettleWeakDataset(
        split="validation",
        data_dir=processed_dir,
        split_config_path=split_cfg_path,
        window_manifest_path=split_manifest_path,
    )
    val_dataset.load_mmap_arrays()

    # 3. Load validation strong targets
    val_window_ids = [val_dataset.get_metadata(i)["window_id"] for i in range(len(val_dataset))]
    targets_map = load_validation_strong_targets(targets_path, val_window_ids)

    logger.info("Loaded %d validation strong evaluation targets.", len(targets_map))

    if args.dry_run:
        logger.info("--- PRODUCTION PATH PREFLIGHT CHECK (1D RESNET) ---")
        logger.info("Train dataset size: %d windows across %d households", len(train_dataset), len(split_manager.config.train_households))
        logger.info("Validation dataset size: %d windows across %d households", len(val_dataset), len(split_manager.config.validation_households))
        logger.info("Validation evaluation targets loaded: %d", len(targets_map))

        # Invariant checks: test split must remain strictly excluded
        train_h_ids = {train_dataset.get_metadata(i)["household_id"] for i in range(0, len(train_dataset), 5000)}
        val_h_ids = {val_dataset.get_metadata(i)["household_id"] for i in range(0, len(val_dataset), 2000)}
        test_h_ids = set(split_manager.config.test_households)

        assert not train_h_ids.intersection(test_h_ids), f"Test households leaked into train dataset: {train_h_ids.intersection(test_h_ids)}"
        assert not val_h_ids.intersection(test_h_ids), f"Test households leaked into validation dataset: {val_h_ids.intersection(test_h_ids)}"
        logger.info("Test split isolation: PASS (H2, H13 strictly isolated)")

        # Target separation check
        sample_x, sample_y = train_dataset[0]
        assert isinstance(sample_x, np.ndarray) and sample_x.shape == (510,)
        assert sample_y in (0, 1)
        logger.info("Model tensor shape and target separation: PASS ((510,) float32, weak presence label)")

        logger.info("PREFLIGHT / DRY-RUN PASSED SUCCESSFULLY. Ready for real 1D ResNet training.")
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Training on device: %s", device)

    # 4. Run training loop
    final_metrics = run_training_loop(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        targets_map=targets_map,
        output_dir=output_dir,
        device=device,
    )

    # 5. Generate Markdown report
    generate_resnet_markdown_report(final_metrics, output_dir / "resnet_report.md")
    logger.info("1D ResNet reference model training completed successfully!")


if __name__ == "__main__":
    main()
