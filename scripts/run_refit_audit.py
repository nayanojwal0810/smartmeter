"""CLI entry point to run forensic data quality audit on cleaned REFIT CSV files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path if executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.validation.refit_audit import REFITAuditor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run streaming forensic quality audit on extracted REFIT CSV files."
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
        default=Path("artifacts/reports/refit_data_audit.json"),
        help="Path where structured JSON audit report will be saved.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/reports/refit_data_audit.md"),
        help="Path where Markdown summary report will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute streaming forensic audit across all households."""
    args = parse_args()

    if not args.data_dir.exists():
        logger.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(1)

    logger.info("Initializing REFIT Auditor with metadata: %s", args.metadata_path)
    auditor = REFITAuditor(metadata_path=args.metadata_path if args.metadata_path.exists() else None)

    logger.info("Starting streaming audit across CSV files in: %s", args.data_dir)
    results = auditor.audit_all_households(args.data_dir)

    logger.info("Generating consolidated summary report...")
    summary = auditor.generate_summary_report(results)
    md_content = auditor.generate_markdown_report(summary)

    # Ensure output directory exists
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved JSON audit report to: %s", args.output_json)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved Markdown audit report to: %s", args.output_md)

    ov = summary["overview"]
    logger.info(
        "Audit completed successfully. Total households: %d | Total rows: %s | Issues==1: %.2f%%",
        ov["total_households"],
        f"{ov['total_rows']:,}",
        ov["overall_issues_pct"],
    )


if __name__ == "__main__":
    main()
