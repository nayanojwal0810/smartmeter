"""Streaming Regular Timebase Resampling Module.

Converts irregular REFIT observations (~6-8s sampling) into a deterministic
8-second regular time grid under locked Decision D-006:
- Gaps <= 16 seconds are bridged with zero-order hold (forward-fill).
- Gaps > 16 seconds start a new contiguous recording segment.
- Zero cross-segment interpolation or forward-filling.
- Single-pass streaming with O(1) memory overhead.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple


@dataclass(frozen=True)
class ResamplingConfig:
    """Configuration parameters for regular timebase resampling."""

    grid_step_s: int = 8
    max_bridge_gap_s: int = 16
    window_points: int = 510


@dataclass
class Segment:
    """Contiguous recording segment metadata."""

    segment_id: int
    household_id: int
    start_unix: int
    end_unix: int
    start_datetime: str
    end_datetime: str
    raw_row_count: int = 0
    grid_point_count: int = 0

    @property
    def duration_seconds(self) -> int:
        """Calculate total segment duration in seconds."""
        return self.end_unix - self.start_unix

    @property
    def eligible_510_windows_count(self) -> int:
        """Calculate number of valid non-overlapping 510-point windows in this segment."""
        return self.grid_point_count // 510

    def to_dict(self) -> dict:
        """Convert segment metadata to dictionary."""
        return {
            "segment_id": self.segment_id,
            "household_id": self.household_id,
            "start_unix": self.start_unix,
            "end_unix": self.end_unix,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "duration_seconds": self.duration_seconds,
            "raw_row_count": self.raw_row_count,
            "grid_point_count": self.grid_point_count,
            "eligible_510_windows_count": self.eligible_510_windows_count,
        }


@dataclass
class GridPoint:
    """Single resampled regular grid point."""

    household_id: int
    segment_id: int
    grid_unix: int
    aggregate_w: float
    kettle_w: float = 0.0
    is_imputed: bool = False  # True if generated via zero-order hold gap bridging


class StreamingTimebaseResampler:
    """Online streaming resampler for REFIT time series data."""

    def __init__(self, config: Optional[ResamplingConfig] = None) -> None:
        self.config = config or ResamplingConfig()

    def resample_observations(
        self,
        observations: Iterator[Tuple[int, str, float, float]],
        household_id: int,
    ) -> Generator[GridPoint, None, None]:
        """Stream raw observations and yield resampled regular grid points.

        Args:
            observations: Iterator of (unix_timestamp, datetime_str, aggregate_watts, kettle_watts)
            household_id: Integer household identifier.

        Yields:
            GridPoint objects aligned to 8-second regular grids within contiguous segments.
        """
        step = self.config.grid_step_s
        max_bridge = self.config.max_bridge_gap_s

        current_segment_id = 0
        prev_obs_unix: Optional[int] = None
        last_agg: float = 0.0
        last_kettle: float = 0.0
        next_grid_unix: Optional[int] = None

        for obs_unix, dt_str, agg_w, kettle_w in observations:
            if prev_obs_unix is None:
                # First observation in the stream: anchor first segment
                current_segment_id += 1
                prev_obs_unix = obs_unix
                last_agg = agg_w
                last_kettle = kettle_w
                next_grid_unix = obs_unix

                # Emit anchor grid point
                yield GridPoint(
                    household_id=household_id,
                    segment_id=current_segment_id,
                    grid_unix=obs_unix,
                    aggregate_w=agg_w,
                    kettle_w=kettle_w,
                    is_imputed=False,
                )
                next_grid_unix += step
                continue

            gap = obs_unix - prev_obs_unix

            if gap <= max_bridge:
                # Safe gap: bridge with zero-order hold up to current obs
                while next_grid_unix is not None and next_grid_unix < obs_unix:
                    yield GridPoint(
                        household_id=household_id,
                        segment_id=current_segment_id,
                        grid_unix=next_grid_unix,
                        aggregate_w=last_agg,
                        kettle_w=last_kettle,
                        is_imputed=True,
                    )
                    next_grid_unix += step

                if next_grid_unix is not None and next_grid_unix == obs_unix:
                    yield GridPoint(
                        household_id=household_id,
                        segment_id=current_segment_id,
                        grid_unix=obs_unix,
                        aggregate_w=agg_w,
                        kettle_w=kettle_w,
                        is_imputed=False,
                    )
                    next_grid_unix += step

                last_agg = agg_w
                last_kettle = kettle_w
                prev_obs_unix = obs_unix

            else:
                # Gap > max_bridge (e.g. > 16s): Start a new contiguous segment
                current_segment_id += 1
                prev_obs_unix = obs_unix
                last_agg = agg_w
                last_kettle = kettle_w
                next_grid_unix = obs_unix

                yield GridPoint(
                    household_id=household_id,
                    segment_id=current_segment_id,
                    grid_unix=obs_unix,
                    aggregate_w=agg_w,
                    kettle_w=kettle_w,
                    is_imputed=False,
                )
                next_grid_unix += step

    def resample_csv_file(
        self,
        file_path: Path,
        household_id: int,
        kettle_col_name: Optional[str] = None,
    ) -> Generator[GridPoint, None, None]:
        """Stream a raw household CSV file and yield resampled 8s grid points."""
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
        kettle_idx = col_indices.get(kettle_col_name) if kettle_col_name else None

        def raw_row_generator() -> Iterator[Tuple[int, str, float, float]]:
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
                    yield u, dt_str, agg, kettle

        yield from self.resample_observations(raw_row_generator(), household_id)
