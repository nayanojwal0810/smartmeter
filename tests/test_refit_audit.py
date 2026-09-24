"""Unit tests for REFIT forensic data quality auditor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.validation.refit_audit import (
    ColumnStats,
    GapStats,
    HouseholdAuditResult,
    REFITAuditor,
    load_refit_metadata,
)


def test_column_stats_numeric() -> None:
    """Verify streaming column statistical updates."""
    stats = ColumnStats()
    for val in ["100", "200", "0", "4500", "-5", "NaN", "invalid", ""]:
        stats.update(val)

    assert stats.valid_count == 5
    assert stats.zero_count == 1
    assert stats.negative_count == 1
    assert stats.above_4000_count == 1
    assert stats.nan_count == 1
    assert stats.non_numeric_count == 2
    assert stats.min_val == -5.0
    assert stats.max_val == 4500.0
    assert pytest.approx(stats.mean_val, 0.01) == (100 + 200 + 0 + 4500 - 5) / 5


def test_gap_stats() -> None:
    """Verify time gap classification categories."""
    gaps = GapStats()
    # Feed various gap intervals
    gaps.update(8.0)
    gaps.update(12.0)
    gaps.update(20.0)
    gaps.update(45.0)
    gaps.update(90.0)
    gaps.update(300.0)
    gaps.update(-1.0)

    assert gaps.total_intervals == 7
    assert gaps.le_8s == 1
    assert gaps.between_9_15s == 1
    assert gaps.gt_15s == 4
    assert gaps.gt_30s == 3
    assert gaps.gt_60s == 2
    assert gaps.gt_120s == 1
    assert gaps.zero_or_negative_gaps == 1
    assert gaps.max_gap_seconds == 300.0


def test_metadata_loader() -> None:
    """Verify loading real metadata tables workbook."""
    meta_path = Path("data/raw/MetaData_Tables.xlsx")
    if not meta_path.exists():
        pytest.skip("MetaData_Tables.xlsx not found")

    metadata = load_refit_metadata(meta_path)
    assert len(metadata) == 20
    assert 1 in metadata
    assert 21 in metadata
    assert 14 not in metadata

    # Check House 2 Kettle
    h2 = metadata[2]
    assert "Appliance8" in h2["iam_channels"]
    assert h2["iam_channels"]["Appliance8"]["name"] == "Kettle"


def test_auditor_on_mock_csv() -> None:
    """Test auditing a mock synthetic household CSV."""
    csv_content = (
        "Time,Unix,Aggregate,Appliance1,Appliance2,Appliance3,Appliance4,Appliance5,Appliance6,Appliance7,Appliance8,Appliance9,Issues\n"
        "2013-10-09 13:06:17,1381323977,500,50,0,0,0,0,0,0,0,0,0\n"
        "2013-10-09 13:06:25,1381323985,550,55,0,0,0,0,0,0,0,0,0\n"
        "2013-10-09 13:06:40,1381324000,1200,60,1000,0,0,0,0,0,0,0,1\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_file = Path(tmpdir) / "CLEAN_House1.csv"
        mock_file.write_text(csv_content, encoding="utf-8")

        auditor = REFITAuditor()
        result = auditor.audit_household_file(mock_file, household_id=1)

        assert result.household_id == 1
        assert result.total_rows == 3
        assert result.malformed_rows == 0
        assert result.header_valid is True
        assert result.min_unix == 1381323977
        assert result.max_unix == 1381324000
        assert result.is_monotonic is True
        assert result.aggregate_stats.valid_count == 3
        assert result.appliance_stats["Appliance1"].max_val == 60.0
        assert result.appliance_stats["Appliance2"].max_val == 1000.0
        assert result.issues_1_count == 1
        assert result.issues_0_count == 2

        summary = auditor.generate_summary_report([result])
        assert summary["overview"]["total_households"] == 1
        assert summary["overview"]["total_rows"] == 3

        md = auditor.generate_markdown_report(summary)
        assert "REFIT Dataset Forensic Data Quality Audit Report" in md
