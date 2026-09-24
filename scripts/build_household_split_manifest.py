"""CLI script to generate household split validation manifests.

Cross-references the locked primary household split (Decision D-007) with the
persisted 174,072-window manifest and generates:
1. `artifacts/manifests/kettle_household_split.json`
2. `artifacts/manifests/kettle_household_split.md`
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate household split validation manifests."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/kettle_household_split.yaml"),
        help="Path to kettle_household_split.yaml config.",
    )
    parser.add_argument(
        "--window-index-json",
        type=Path,
        default=Path("artifacts/manifests/timebase_window_index.json"),
        help="Path to timebase_window_index.json summary manifest.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/manifests/kettle_household_split.json"),
        help="Path where split validation JSON will be saved.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/manifests/kettle_household_split.md"),
        help="Path where split validation Markdown report will be saved.",
    )
    return parser.parse_args()


def generate_split_manifest(
    config_path: Path,
    window_index_json_path: Path,
) -> Dict[str, Any]:
    """Generate detailed split manifest and summary metrics."""
    if config_path.exists():
        cfg = HouseholdSplitManager.load_from_yaml(config_path)
        split_mgr = HouseholdSplitManager(config=cfg)
    else:
        split_mgr = HouseholdSplitManager()

    if not window_index_json_path.exists():
        raise FileNotFoundError(f"Window index JSON not found at: {window_index_json_path}")

    with open(window_index_json_path, "r", encoding="utf-8") as f:
        window_index = json.load(f)

    hh_summaries = window_index.get("households", [])

    splits_data: Dict[str, Dict[str, Any]] = {
        "train": {
            "split_name": "train",
            "households_count": len(split_mgr.config.train_households),
            "households": list(split_mgr.config.train_households),
            "positive_weak_label_households": [],
            "unmonitored_weak_label_households": [],
            "total_windows": 0,
            "positive_weak_label_windows": 0,
            "unmonitored_weak_label_windows": 0,
            "strong_evaluation_eligible_households": [],
            "strong_evaluation_ineligible_households": [],
            "strong_kettle_target_windows_count": 0,
        },
        "validation": {
            "split_name": "validation",
            "households_count": len(split_mgr.config.validation_households),
            "households": list(split_mgr.config.validation_households),
            "positive_weak_label_households": [],
            "unmonitored_weak_label_households": [],
            "total_windows": 0,
            "positive_weak_label_windows": 0,
            "unmonitored_weak_label_windows": 0,
            "strong_evaluation_eligible_households": [],
            "strong_evaluation_ineligible_households": [],
            "strong_kettle_target_windows_count": 0,
        },
        "test": {
            "split_name": "test",
            "households_count": len(split_mgr.config.test_households),
            "households": list(split_mgr.config.test_households),
            "positive_weak_label_households": [],
            "unmonitored_weak_label_households": [],
            "total_windows": 0,
            "positive_weak_label_windows": 0,
            "unmonitored_weak_label_windows": 0,
            "strong_evaluation_eligible_households": [],
            "strong_evaluation_ineligible_households": [],
            "strong_kettle_target_windows_count": 0,
        },
    }

    household_rows: List[Dict[str, Any]] = []

    for h in hh_summaries:
        h_id = h["household_id"]
        split_name = split_mgr.get_split(h_id)
        weak_label = split_mgr.get_weak_label(h_id)
        n_windows = h["total_510_windows"]
        n_target_win = h["windows_with_kettle_target"]
        is_eval_elig = h["is_evaluation_eligible"]

        sp = splits_data[split_name]
        sp["total_windows"] += n_windows
        sp["strong_kettle_target_windows_count"] += n_target_win

        if weak_label == 1:
            sp["positive_weak_label_households"].append(h_id)
            sp["positive_weak_label_windows"] += n_windows
        else:
            sp["unmonitored_weak_label_households"].append(h_id)
            sp["unmonitored_weak_label_windows"] += n_windows

        if is_eval_elig:
            sp["strong_evaluation_eligible_households"].append(h_id)
        else:
            sp["strong_evaluation_ineligible_households"].append(h_id)

        household_rows.append({
            "household_id": h_id,
            "split": split_name,
            "weak_label": weak_label,
            "weak_label_name": "Positive (Presence)" if weak_label == 1 else "Negative (Unmonitored)",
            "kettle_channel": h["kettle_channel"],
            "total_510_windows": n_windows,
            "strong_target_windows": n_target_win,
            "is_evaluation_eligible": is_eval_elig,
        })

    # Sort household rows by household_id
    household_rows.sort(key=lambda r: r["household_id"])

    total_windows_all = sum(s["total_windows"] for s in splits_data.values())
    total_target_windows_all = sum(s["strong_kettle_target_windows_count"] for s in splits_data.values())

    manifest = {
        "split_config": split_mgr.to_dict(),
        "summary": {
            "total_households": len(household_rows),
            "total_windows": total_windows_all,
            "total_strong_target_windows": total_target_windows_all,
            "splits": splits_data,
        },
        "households": household_rows,
    }

    return manifest


def generate_markdown(manifest: Dict[str, Any]) -> str:
    """Generate GitHub-style Markdown report of the primary household split."""
    cfg = manifest["split_config"]
    sum_data = manifest["summary"]
    splits = sum_data["splits"]
    hh_rows = manifest["households"]

    tr = splits["train"]
    va = splits["validation"]
    te = splits["test"]

    lines = [
        "# Primary Household Split & Weak-Supervision Dataset Specification",
        "",
        "**Decision Reference:** D-007 (Frozen Primary Household Split)  ",
        f"**Split Version:** `{cfg['split_version']}` ({cfg['split_name']})  ",
        "**Split Architecture:** **16 Train / 2 Validation / 2 Test**  ",
        f"**Total Processed Households:** {sum_data['total_households']}  ",
        f"**Total Valid 510-Point Windows:** {sum_data['total_windows']:,}  ",
        "",
        "## 1. Split Allocation & Window Yield Summary",
        "",
        "| Split | Households | Household IDs | Total Windows | Window % | Positive Weak Windows (Label=1) | Unmonitored Weak Windows (Label=0) | Strong Active Target Windows (Eval Only) |",
        "|---|---|---|---|---|---|---|---|",
        f"| **TRAIN** | {tr['households_count']} | {', '.join(f'H{h}' for h in tr['households'])} | {tr['total_windows']:,} | {100*tr['total_windows']/sum_data['total_windows']:.2f}% | {tr['positive_weak_label_windows']:,} (11 houses) | {tr['unmonitored_weak_label_windows']:,} (5 houses) | {tr['strong_kettle_target_windows_count']:,} |",
        f"| **VALIDATION** | {va['households_count']} | {', '.join(f'H{h}' for h in va['households'])} | {va['total_windows']:,} | {100*va['total_windows']/sum_data['total_windows']:.2f}% | {va['positive_weak_label_windows']:,} (2 houses) | 0 (0 houses) | {va['strong_kettle_target_windows_count']:,} |",
        f"| **TEST** | {te['households_count']} | {', '.join(f'H{h}' for h in te['households'])} | {te['total_windows']:,} | {100*te['total_windows']/sum_data['total_windows']:.2f}% | {te['positive_weak_label_windows']:,} (2 houses) | 0 (0 houses) | {te['strong_kettle_target_windows_count']:,} |",
        f"| **TOTAL** | **20** | **All 20 REFIT Houses** | **{sum_data['total_windows']:,}** | **100.00%** | **{tr['positive_weak_label_windows']+va['positive_weak_label_windows']+te['positive_weak_label_windows']:,} (15 houses)** | **{tr['unmonitored_weak_label_windows']:,} (5 houses)** | **{sum_data['total_strong_target_windows']:,} (9.10%)** |",
        "",
        "## 2. Leakage-Safety and Weak Supervision Rules",
        "",
        "- **Zero Cross-Household Leakage:** Every household is assigned strictly to a single split. All 510-point windows inherit their household's split unconditionally.",
        "- **Weak-Label Independence:** The weak training label is derived solely from metadata-level Kettle presence indicator (1 = positive presence, 0 = unmonitored). Timestamp-level sub-meter measurements are strictly excluded from model training.",
        "- **Test Split Freeze:** Holdout test households (H2, H13) are strictly frozen. They must never be accessed for model training, hyperparameter tuning, or threshold selection.",
        "- **House 12 Handling:** H12 provides weak supervision presence label (1) during training, but remains strictly excluded from primary localization evaluation (`is_evaluation_eligible=False`).",
        "- **Unmonitored Households:** H1, H10, H15, H16, H18 are all placed in the TRAIN split to provide unmonitored negative weak-supervision examples.",
        "",
        "## 3. Household-by-Household Allocation Table",
        "",
        "| House | Split | Weak Label | Weak Label Meaning | Kettle Channel | 510-Pt Windows | Strong Target Windows (Audit Only) | Strong Eval Eligible |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in hh_rows:
        elig_str = "Yes" if r["is_evaluation_eligible"] else "No (Excluded/None)"
        lines.append(
            f"| House {r['household_id']} | **{r['split'].upper()}** | {r['weak_label']} | {r['weak_label_name']} | {r['kettle_channel'] or 'None'} | {r['total_510_windows']:,} | {r['strong_target_windows']:,} | {elig_str} |"
        )

    return "\n".join(lines)


def main() -> None:
    """Execute split manifest generation."""
    args = parse_args()

    manifest = generate_split_manifest(
        config_path=args.config_path,
        window_index_json_path=args.window_index_json,
    )
    md_content = generate_markdown(manifest)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved JSON split manifest to: %s", args.output_json)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")
    logger.info("Saved Markdown split report to: %s", args.output_md)


if __name__ == "__main__":
    main()
