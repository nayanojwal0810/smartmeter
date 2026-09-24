"""CLI entry point to execute Kettle target quality and activation profile audit."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.validation.kettle_target_audit import KettleTargetAuditor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Kettle target quality and activation audit across REFIT households."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/interim/refit_clean"),
        help="Directory containing extracted CLEAN_House*.csv files.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/raw/MetaData_Tables.xlsx"),
        help="Path to MetaData_Tables.xlsx metadata workbook.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/reports/kettle_target_audit.json"),
        help="Path where structured JSON Kettle audit report will be saved.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/reports/kettle_target_audit.md"),
        help="Path where Markdown summary report will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute Kettle target audit."""
    args = parse_args()

    if not args.data_dir.exists():
        logger.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(1)

    logger.info("Initializing Kettle Target Auditor with metadata: %s", args.metadata_path)
    auditor = KettleTargetAuditor(metadata_path=args.metadata_path if args.metadata_path.exists() else None)

    logger.info("Auditing Kettle channels across CSV files in: %s", args.data_dir)
    results = auditor.audit_all(args.data_dir)

    logger.info("Generating Kettle audit summary report...")
    summary = auditor.generate_summary(results)
    md_content = auditor.generate_markdown(summary)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved JSON Kettle audit report to: %s", args.output_json)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved Markdown Kettle audit report to: %s", args.output_md)

    meta = summary["metadata_summary"]
    ev = summary["aggregate_kettle_events"]
    logger.info(
        "Kettle audit complete. Positive: %d | Negative: %d | 500W Events: %d | 1500W Events: %d",
        meta["positive_households_count"],
        meta["negative_households_count"],
        ev["total_events_500w_descriptive_rule"],
        ev["total_events_1500w_reference"],
    )


if __name__ == "__main__":
    main()
