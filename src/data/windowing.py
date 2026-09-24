"""Streaming Window Generation and Indexing Module.

Generates deterministic 510-point non-overlapping model windows from regular 8-second
grid points under locked Decision D-006:
- Windows must reside strictly within a single contiguous recording segment.
- Zero windows crossing segment boundaries.
- Leftover points (< 510) at segment ends are safely dropped.
- O(1) streaming memory complexity.
- Pure structural model-input metadata (strong sub-meter targets strictly excluded).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

from src.data.timebase import GridPoint, ResamplingConfig, Segment


@dataclass
class WindowMetadata:
    """Structural metadata descriptor for a single 510-point model window.

    Contains only aggregate power metrics and temporal boundaries. Strong sub-meter
    appliance target labels are strictly excluded to prevent training data leakage.
    """

    window_id: int
    household_id: int
    segment_id: int
    window_index_in_segment: int
    start_unix: int
    end_unix: int
    start_datetime: str
    end_datetime: str
    num_points: int = 510
    duration_seconds: int = 4072  # (510 - 1) * 8s = 4072 seconds (~67.87 min)
    mean_aggregate_w: float = 0.0
    max_aggregate_w: float = 0.0
    imputed_points_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert window metadata to serializable dictionary."""
        return {
            "window_id": self.window_id,
            "household_id": self.household_id,
            "segment_id": self.segment_id,
            "window_index_in_segment": self.window_index_in_segment,
            "start_unix": self.start_unix,
            "end_unix": self.end_unix,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "num_points": self.num_points,
            "duration_seconds": self.duration_seconds,
            "mean_aggregate_w": round(self.mean_aggregate_w, 2),
            "max_aggregate_w": round(self.max_aggregate_w, 2),
            "imputed_points_count": self.imputed_points_count,
        }


class StreamingWindowIndexer:
    """Streaming window indexer assembling 510-point windows within contiguous segments."""

    def __init__(self, config: Optional[ResamplingConfig] = None) -> None:
        self.config = config or ResamplingConfig()

    def generate_windows(
        self,
        grid_points: Iterator[GridPoint],
        start_window_id: int = 1,
    ) -> Generator[WindowMetadata, None, None]:
        """Stream regular grid points and yield non-overlapping 510-point WindowMetadata entries.

        Ensures that every window strictly belongs to a single segment. Residual points
        at segment terminations are discarded without crossing segment boundaries.
        """
        for win_meta, _ in self.generate_windows_with_points(grid_points, start_window_id):
            yield win_meta

    def generate_windows_with_points(
        self,
        grid_points: Iterator[GridPoint],
        start_window_id: int = 1,
    ) -> Generator[Tuple[WindowMetadata, List[GridPoint]], None, None]:
        """Stream regular grid points and yield (WindowMetadata, List[GridPoint]) pairs."""
        win_size = self.config.window_points
        step = self.config.grid_step_s
        current_window_id = start_window_id

        current_segment_id: Optional[int] = None
        current_window_in_seg = 0
        point_buffer: List[GridPoint] = []

        for pt in grid_points:
            if current_segment_id is None:
                current_segment_id = pt.segment_id

            if pt.segment_id != current_segment_id:
                # Segment boundary transition: discard any leftover points (< win_size)
                point_buffer.clear()
                current_segment_id = pt.segment_id
                current_window_in_seg = 0

            point_buffer.append(pt)

            if len(point_buffer) == win_size:
                # Complete 510-point window formed within current segment
                first_pt = point_buffer[0]
                last_pt = point_buffer[-1]

                agg_sum = sum(p.aggregate_w for p in point_buffer)
                agg_max = max(p.aggregate_w for p in point_buffer)
                imputed_count = sum(1 for p in point_buffer if p.is_imputed)

                start_dt = datetime.fromtimestamp(first_pt.grid_unix, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                end_dt = datetime.fromtimestamp(last_pt.grid_unix, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                win_meta = WindowMetadata(
                    window_id=current_window_id,
                    household_id=first_pt.household_id,
                    segment_id=first_pt.segment_id,
                    window_index_in_segment=current_window_in_seg,
                    start_unix=first_pt.grid_unix,
                    end_unix=last_pt.grid_unix,
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                    num_points=win_size,
                    duration_seconds=(win_size - 1) * step,
                    mean_aggregate_w=agg_sum / win_size,
                    max_aggregate_w=agg_max,
                    imputed_points_count=imputed_count,
                )

                yield win_meta, list(point_buffer)

                current_window_id += 1
                current_window_in_seg += 1
                point_buffer.clear()
