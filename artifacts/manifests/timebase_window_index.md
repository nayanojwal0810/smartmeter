# Regular Timebase & 510-Point Model Window Index Manifest

## 1. Locked Timebase & Evaluation Target Configuration

| Parameter | Value | Description |
|---|---|---|
| **Grid Resolution** | 8 seconds | Uniform 8-second sampling grid (Decision D-006) |
| **Max Bridge Gap** | <= 16 seconds | Zero-order hold (forward-fill) |
| **Segment Boundary** | > 16 seconds | Starts new contiguous segment |
| **Window Length** | 510 points (4072s / 67.87 min) | Non-overlapping reference windows strictly within single segment |
| **Kettle ON Threshold** | >= 1500.0 W | Evaluation target threshold (Decision D-005) |
| **Kettle Event Continuity** | <= 20 seconds | Max gap between active samples in single target event |
| **Kettle Max Event Duration** | <= 600 seconds (10 min) | Candidate events >600s excluded as target anomalies |
| **Excluded Households** | House [12] | Excluded from primary localization evaluation |

## 2. Global Dataset Summary

| Metric | Value |
|---|---|
| **Total Households Indexed** | 20 |
| **Total Contiguous Segments** | 386,864 |
| **Segments with Valid Windows (>=510 pts)** | 23,548 |
| **Segments without Valid Windows (<510 pts)** | 363,316 |
| **Total Valid 510-Point Windows** | 174,072 windows |
| **Windows with Valid Kettle Target (>=1500W)** | 15,837 (9.1%) |
| **Structural Windows Artifact** | `timebase_windows.jsonl` |
| **Evaluation Targets Artifact** | `kettle_evaluation_targets.jsonl` |

## 3. Household-by-Household Window Yield

| House | Kettle Channel | Eval Eligible | Total Segments | Segs w/ Windows | 510-Pt Windows | Active Kettle Windows | Active Window % |
|---|---|---|---|---|---|---|---|
| House 1 | None | No (Excluded/None) | 12,779 | 1,517 | 10,388 | 0 | 0.00% |
| House 2 | Appliance8 | Yes | 14,707 | 1,106 | 8,730 | 2,053 | 23.52% |
| House 3 | Appliance9 | Yes | 12,448 | 1,382 | 10,246 | 125 | 1.22% |
| House 4 | Appliance9 | Yes | 21,146 | 1,357 | 10,208 | 1,441 | 14.12% |
| House 5 | Appliance8 | Yes | 57,521 | 2,243 | 7,651 | 1,798 | 23.50% |
| House 6 | Appliance7 | Yes | 22,827 | 1,469 | 8,423 | 81 | 0.96% |
| House 7 | Appliance9 | Yes | 16,072 | 1,185 | 10,187 | 54 | 0.53% |
| House 8 | Appliance9 | Yes | 16,059 | 1,202 | 9,362 | 2,264 | 24.18% |
| House 9 | Appliance7 | Yes | 16,026 | 1,027 | 9,154 | 2,120 | 23.16% |
| House 10 | None | No (Excluded/None) | 27,333 | 1,430 | 9,120 | 0 | 0.00% |
| House 11 | Appliance7 | Yes | 11,209 | 538 | 6,807 | 1,838 | 27.00% |
| House 12 | Appliance6 | No (Excluded/None) | 7,515 | 1,062 | 8,758 | 0 | 0.00% |
| House 13 | Appliance9 | Yes | 50,161 | 1,400 | 5,778 | 145 | 2.51% |
| House 15 | None | No (Excluded/None) | 15,061 | 1,012 | 9,387 | 0 | 0.00% |
| House 16 | None | No (Excluded/None) | 16,734 | 1,097 | 8,316 | 0 | 0.00% |
| House 17 | Appliance8 | Yes | 15,333 | 909 | 8,353 | 2,257 | 27.02% |
| House 18 | None | No (Excluded/None) | 11,648 | 845 | 8,065 | 0 | 0.00% |
| House 19 | Appliance5 | Yes | 15,790 | 1,002 | 8,253 | 57 | 0.69% |
| House 20 | Appliance9 | Yes | 12,765 | 819 | 8,419 | 1,595 | 18.95% |
| House 21 | Appliance7 | Yes | 13,730 | 946 | 8,467 | 9 | 0.11% |