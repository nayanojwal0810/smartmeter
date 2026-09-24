# Post-Build Preprocessing & Index Artifact Integrity Audit Report

**Audit Status:** **PASS**  
**Recorded Execution Runtime:** **8m10s**  
**Total Processed Households:** **20**  
**Total Valid 510-Point Windows:** **174,072**  
**Windows with Valid Kettle Target (>=1500W):** **15,837 (9.1%)**

## 1. Summary of Integrity Checks

| Check ID | Description | Status | Details |
|---|---|---|---|
| **CHK-01** | File Existence & Basic Integrity | PASS | All 4 artifacts present (Windows JSONL: 61,135,137 B, Targets JSONL: 38,752,083 B) |
| **CHK-02** | Window Count Consistency | PASS | Exactly 174,072 windows verified across JSON overview, JSONL lines, and MD table |
| **CHK-03** | Window Structure & Target Separation | PASS | Pure structural fields only, 510 points, 4072s duration, strictly monotonic IDs |
| **CHK-04** | Target Artifact & Policy Enforcement | PASS | H12 excluded, H1/10/15/16/18 marked unmonitored, 14 eligible houses verified |
| **CHK-05** | Segment & Household Accounting | PASS | Total 386,864 segments = 23,548 with windows + 363,316 without windows |
| **CHK-06** | Markdown Cross-Verification | PASS | Markdown matches all overview counts and all 20 household rows |

## 2. Household-by-Household Audited Yield

| House | Kettle Channel | Eval Eligible | Total Segments | Segments w/ Windows | Segments w/o Windows | 510-Pt Windows | Active Kettle Windows | Active % |
|---|---|---|---|---|---|---|---|---|
| House 1 | None | No (Excluded/None) | 12,779 | 1,517 | 11,262 | 10,388 | 0 | 0.00% |
| House 2 | Appliance8 | Yes | 14,707 | 1,106 | 13,601 | 8,730 | 2,053 | 23.52% |
| House 3 | Appliance9 | Yes | 12,448 | 1,382 | 11,066 | 10,246 | 125 | 1.22% |
| House 4 | Appliance9 | Yes | 21,146 | 1,357 | 19,789 | 10,208 | 1,441 | 14.12% |
| House 5 | Appliance8 | Yes | 57,521 | 2,243 | 55,278 | 7,651 | 1,798 | 23.50% |
| House 6 | Appliance7 | Yes | 22,827 | 1,469 | 21,358 | 8,423 | 81 | 0.96% |
| House 7 | Appliance9 | Yes | 16,072 | 1,185 | 14,887 | 10,187 | 54 | 0.53% |
| House 8 | Appliance9 | Yes | 16,059 | 1,202 | 14,857 | 9,362 | 2,264 | 24.18% |
| House 9 | Appliance7 | Yes | 16,026 | 1,027 | 14,999 | 9,154 | 2,120 | 23.16% |
| House 10 | None | No (Excluded/None) | 27,333 | 1,430 | 25,903 | 9,120 | 0 | 0.00% |
| House 11 | Appliance7 | Yes | 11,209 | 538 | 10,671 | 6,807 | 1,838 | 27.00% |
| House 12 | Appliance6 | No (Excluded/None) | 7,515 | 1,062 | 6,453 | 8,758 | 0 | 0.00% |
| House 13 | Appliance9 | Yes | 50,161 | 1,400 | 48,761 | 5,778 | 145 | 2.51% |
| House 15 | None | No (Excluded/None) | 15,061 | 1,012 | 14,049 | 9,387 | 0 | 0.00% |
| House 16 | None | No (Excluded/None) | 16,734 | 1,097 | 15,637 | 8,316 | 0 | 0.00% |
| House 17 | Appliance8 | Yes | 15,333 | 909 | 14,424 | 8,353 | 2,257 | 27.02% |
| House 18 | None | No (Excluded/None) | 11,648 | 845 | 10,803 | 8,065 | 0 | 0.00% |
| House 19 | Appliance5 | Yes | 15,790 | 1,002 | 14,788 | 8,253 | 57 | 0.69% |
| House 20 | Appliance9 | Yes | 12,765 | 819 | 11,946 | 8,419 | 1,595 | 18.95% |
| House 21 | Appliance7 | Yes | 13,730 | 946 | 12,784 | 8,467 | 9 | 0.11% |
