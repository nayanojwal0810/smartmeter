"""CLI entry point to build regular 8s timebase segments and 510-point window index.

Streams through extracted REFIT CSVs and incrementally writes:
1. `artifacts/manifests/timebase_windows.jsonl`: Structural model-input window descriptors.
2. `artifacts/manifests/kettle_evaluation_targets.jsonl`: Offline Kettle evaluation targets (Decision D-005).
3. `artifacts/manifests/timebase_window_index.json`: Dataset summary manifest and household yield metrics.
4. `artifacts/manifests/timebase_window_index.md`: Human-readable summary report.

Operates in single-pass streaming mode with bounded O(1) memory complexity.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.target import RawKettleTargetExtractor
from src.data.timebase import ResamplingConfig, StreamingTimebaseResampler
from src.data.windowing import StreamingWindowIndexer
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
        description="Build 8-second timebase segments and 510-point model window index manifest."
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
        "--output-summary-json",
        type=Path,
        default=Path("artifacts/manifests/timebase_window_index.json"),
        help="Path where dataset summary JSON manifest will be saved.",
    )
    parser.add_argument(
        "--output-windows-jsonl",
        type=Path,
        default=Path("artifacts/manifests/timebase_windows.jsonl"),
        help="Path where streaming model-window descriptors JSONL will be saved.",
    )
    parser.add_argument(
        "--output-targets-jsonl",
        type=Path,
        default=Path("artifacts/manifests/kettle_evaluation_targets.jsonl"),
        help="Path where streaming Kettle evaluation targets JSONL will be saved.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/manifests/timebase_window_index.md"),
        help="Path where Markdown summary report will be saved.",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=8,
        help="Regular grid resolution in seconds (default: 8).",
    )
    parser.add_argument(
        "--max-bridge-gap",
        type=int,
        default=16,
        help="Maximum gap in seconds bridged via zero-order hold (default: 16).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=510,
        help="Window length in grid points (default: 510).",
    )
    return parser.parse_args()


def build_manifest_streaming(
    data_dir: Path,
    metadata_path: Path,
    output_windows_jsonl: Path,
    output_targets_jsonl: Path,
    config: ResamplingConfig,
) -> Dict[str, Any]:
    """Stream across household CSVs, append records to JSONL files, and return summary manifest."""
    auditor = KettleTargetAuditor(metadata_path=metadata_path if metadata_path.exists() else None)
    resampler = StreamingTimebaseResampler(config=config)
    indexer = StreamingWindowIndexer(config=config)
    target_extractor = RawKettleTargetExtractor()

    col_indices = {
        "Appliance1": 3,
        "Appliance2": 4,
        "Appliance3": 5,
        "Appliance4": 6,
        "Appliance5": 7,
        "Appliance6": 8,
        "Appliance7": 9,
        "Appliance8": 10,
        "Appliance9": 11,
    }

    files = sorted(
        data_dir.glob("CLEAN_House*.csv"),
        key=lambda p: int("".join(filter(str.isdigit, p.name)) or 0),
    )

    output_windows_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_targets_jsonl.parent.mkdir(parents=True, exist_ok=True)

    household_summaries: List[Dict[str, Any]] = []

    global_win_id = 1
    total_contiguous_segments_all = 0
    total_segments_with_windows_all = 0
    total_valid_windows_all = 0
    total_windows_with_kettle_target_all = 0

    with open(output_windows_jsonl, "w", encoding="utf-8") as f_win, open(
        output_targets_jsonl, "w", encoding="utf-8"
    ) as f_tgt:

        for file_path in files:
            digits = "".join(c for c in file_path.name if c.isdigit())
            h_id = int(digits) if digits else 0

            kettle_col, _, _ = auditor.discover_kettle_channel(h_id)
            has_kettle = kettle_col is not None
            is_eval_eligible = target_extractor.is_household_eligible(h_id, has_kettle)
            kettle_idx = col_indices.get(kettle_col) if kettle_col else None

            logger.info("Processing House %d (Kettle: %s)...", h_id, kettle_col or "None")

            def raw_stream():
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if len(row) != 13:
                            continue
                        try:
                            u = int(row[1])
                            dt_str = row[0].strip()
                            agg = float(row[2])
                            kettle = float(row[kettle_idx]) if kettle_idx else 0.0
                        except (ValueError, IndexError):
                            continue

                        # Feed raw target extractor preserving 20s continuity independently of 16s timebase breaks
                        target_extractor.process_raw_observation(
                            obs_unix=u,
                            kettle_w=kettle,
                            household_id=h_id,
                            has_kettle_channel=has_kettle,
                        )
                        yield u, dt_str, agg, kettle

            grid_points_gen = resampler.resample_observations(raw_stream(), household_id=h_id)

            h_windows = 0
            h_active_target_windows = 0
            h_segments_seen = 0
            h_segments_with_windows = 0
            last_segment_id_seen = 0

            def tracked_grid_points():
                nonlocal h_segments_seen
                for pt in grid_points_gen:
                    if pt.segment_id > h_segments_seen:
                        h_segments_seen = pt.segment_id
                    yield pt

            # Window generator yielding (WindowMetadata, List[GridPoint])
            for win_meta, win_points in indexer.generate_windows_with_points(
                tracked_grid_points(), start_window_id=global_win_id
            ):
                h_windows += 1

                # Track segment statistics
                if win_meta.segment_id != last_segment_id_seen:
                    h_segments_with_windows += 1
                last_segment_id_seen = win_meta.segment_id

                # Match separate evaluation target from raw target extractor (Decision D-005)
                win_target = target_extractor.match_window_target(
                    window_id=win_meta.window_id,
                    household_id=h_id,
                    segment_id=win_meta.segment_id,
                    window_grid_points=win_points,
                    has_kettle_channel=has_kettle,
                )

                if win_target.has_active_target:
                    h_active_target_windows += 1

                # Incrementally stream window descriptor and evaluation target to disk
                f_win.write(json.dumps(win_meta.to_dict()) + "\n")
                f_tgt.write(json.dumps(win_target.to_dict()) + "\n")

                global_win_id += 1

            target_extractor.close_stream(household_id=h_id, has_kettle_channel=has_kettle)

            h_segments_without_windows = max(0, h_segments_seen - h_segments_with_windows)

            total_contiguous_segments_all += h_segments_seen
            total_segments_with_windows_all += h_segments_with_windows
            total_valid_windows_all += h_windows
            total_windows_with_kettle_target_all += h_active_target_windows

            household_summaries.append({
                "household_id": h_id,
                "filename": file_path.name,
                "kettle_channel": kettle_col,
                "is_evaluation_eligible": is_eval_eligible,
                "total_contiguous_segments": h_segments_seen,
                "segments_with_windows_count": h_segments_with_windows,
                "segments_without_windows_count": h_segments_without_windows,
                "total_510_windows": h_windows,
                "windows_with_kettle_target": h_active_target_windows,
                "kettle_target_window_pct": round(
                    100.0 * h_active_target_windows / max(1, h_windows), 2
                )
                if is_eval_eligible
                else 0.0,
            })

    total_segments_without_windows_all = (
        total_contiguous_segments_all - total_segments_with_windows_all
    )

    return {
        "config": {
            "grid_step_s": config.grid_step_s,
            "max_bridge_gap_s": config.max_bridge_gap_s,
            "window_points": config.window_points,
            "window_duration_seconds": (config.window_points - 1) * config.grid_step_s,
            "kettle_power_threshold_w": target_extractor.power_threshold_w,
            "kettle_max_continuity_gap_s": target_extractor.max_continuity_gap_s,
            "kettle_max_duration_s": target_extractor.max_duration_s,
            "kettle_excluded_households": list(target_extractor.excluded_households),
        },
        "overview": {
            "total_households": len(household_summaries),
            "total_contiguous_segments": total_contiguous_segments_all,
            "segments_with_windows_count": total_segments_with_windows_all,
            "segments_without_windows_count": total_segments_without_windows_all,
            "total_valid_510_windows": total_valid_windows_all,
            "total_windows_with_kettle_target": total_windows_with_kettle_target_all,
            "overall_kettle_target_window_pct": round(
                100.0 * total_windows_with_kettle_target_all / max(1, total_valid_windows_all), 2
            ),
            "windows_manifest_file": output_windows_jsonl.name,
            "targets_manifest_file": output_targets_jsonl.name,
        },
        "households": household_summaries,
    }


def generate_markdown(manifest: Dict[str, Any]) -> str:
    """Generate human-readable summary of timebase and window manifest."""
    cfg = manifest["config"]
    ov = manifest["overview"]
    hh = manifest["households"]

    win_span_min = cfg["window_duration_seconds"] / 60.0

    lines = [
        "# Regular Timebase & 510-Point Model Window Index Manifest",
        "",
        "## 1. Locked Timebase & Evaluation Target Configuration",
        "",
        "| Parameter | Value | Description |",
        "|---|---|---|",
        f"| **Grid Resolution** | {cfg['grid_step_s']} seconds | Uniform 8-second sampling grid (Decision D-006) |",
        f"| **Max Bridge Gap** | <= {cfg['max_bridge_gap_s']} seconds | Zero-order hold (forward-fill) |",
        f"| **Segment Boundary** | > {cfg['max_bridge_gap_s']} seconds | Starts new contiguous segment |",
        f"| **Window Length** | {cfg['window_points']} points ({cfg['window_duration_seconds']}s / {win_span_min:.2f} min) | Non-overlapping reference windows strictly within single segment |",
        f"| **Kettle ON Threshold** | >= {cfg['kettle_power_threshold_w']} W | Evaluation target threshold (Decision D-005) |",
        f"| **Kettle Event Continuity** | <= {cfg['kettle_max_continuity_gap_s']} seconds | Max gap between active samples in single target event |",
        f"| **Kettle Max Event Duration** | <= {cfg['kettle_max_duration_s']} seconds (10 min) | Candidate events >600s excluded as target anomalies |",
        f"| **Excluded Households** | House {cfg['kettle_excluded_households']} | Excluded from primary localization evaluation |",
        "",
        "## 2. Global Dataset Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Total Households Indexed** | {ov['total_households']} |",
        f"| **Total Contiguous Segments** | {ov['total_contiguous_segments']:,} |",
        f"| **Segments with Valid Windows (>=510 pts)** | {ov['segments_with_windows_count']:,} |",
        f"| **Segments without Valid Windows (<510 pts)** | {ov['segments_without_windows_count']:,} |",
        f"| **Total Valid 510-Point Windows** | {ov['total_valid_510_windows']:,} windows |",
        f"| **Windows with Valid Kettle Target (>=1500W)** | {ov['total_windows_with_kettle_target']:,} ({ov['overall_kettle_target_window_pct']}%) |",
        f"| **Structural Windows Artifact** | `{ov['windows_manifest_file']}` |",
        f"| **Evaluation Targets Artifact** | `{ov['targets_manifest_file']}` |",
        "",
        "## 3. Household-by-Household Window Yield",
        "",
        "| House | Kettle Channel | Eval Eligible | Total Segments | Segs w/ Windows | 510-Pt Windows | Active Kettle Windows | Active Window % |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for h in hh:
        k_col = h["kettle_channel"] or "None"
        eligible_str = "Yes" if h["is_evaluation_eligible"] else "No (Excluded/None)"
        lines.append(
            f"| House {h['household_id']} | {k_col} | {eligible_str} | {h['total_contiguous_segments']:,} | {h['segments_with_windows_count']:,} | {h['total_510_windows']:,} | {h['windows_with_kettle_target']:,} | {h['kettle_target_window_pct']:.2f}% |"
        )

    return "\n".join(lines)


def main() -> None:
    """Execute streaming timebase and window indexing."""
    args = parse_args()

    if not args.data_dir.exists():
        logger.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(1)

    cfg = ResamplingConfig(
        grid_step_s=args.grid_step,
        max_bridge_gap_s=args.max_bridge_gap,
        window_points=args.window_size,
    )

    logger.info(
        "Starting streaming timebase and window index build (Grid: %ds, Max Bridge: %ds, Window: %d pts / %ds)...",
        cfg.grid_step_s,
        cfg.max_bridge_gap_s,
        cfg.window_points,
        (cfg.window_points - 1) * cfg.grid_step_s,
    )

    manifest = build_manifest_streaming(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        output_windows_jsonl=args.output_windows_jsonl,
        output_targets_jsonl=args.output_targets_jsonl,
        config=cfg,
    )
    md_content = generate_markdown(manifest)

    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved JSON summary manifest to: %s", args.output_summary_json)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved Markdown summary report to: %s", args.output_md)

    ov = manifest["overview"]
    logger.info(
        "Index build completed. Total windows: %d | Windows with Kettle target: %d (%.2f%%)",
        ov["total_valid_510_windows"],
        ov["total_windows_with_kettle_target"],
        ov["overall_kettle_target_window_pct"],
    )


if __name__ == "__main__":
    main()
