# Primary Household Split & Weak-Supervision Dataset Specification

**Decision Reference:** D-007 (Frozen Primary Household Split)  
**Split Version:** `v1.0.0` (primary_household_split_16_2_2)  
**Split Architecture:** **16 Train / 2 Validation / 2 Test**  
**Total Processed Households:** 20  
**Total Valid 510-Point Windows:** 174,072  

## 1. Split Allocation & Window Yield Summary

| Split | Households | Household IDs | Total Windows | Window % | Positive Weak Windows (Label=1) | Unmonitored Weak Windows (Label=0) | Strong Active Target Windows (Eval Only) |
|---|---|---|---|---|---|---|---|
| **TRAIN** | 16 | H1, H3, H5, H6, H7, H8, H9, H10, H11, H12, H15, H16, H18, H19, H20, H21 | 141,003 | 81.00% | 95,727 (11 houses) | 45,276 (5 houses) | 9,941 |
| **VALIDATION** | 2 | H4, H17 | 18,561 | 10.66% | 18,561 (2 houses) | 0 (0 houses) | 3,698 |
| **TEST** | 2 | H2, H13 | 14,508 | 8.33% | 14,508 (2 houses) | 0 (0 houses) | 2,198 |
| **TOTAL** | **20** | **All 20 REFIT Houses** | **174,072** | **100.00%** | **128,796 (15 houses)** | **45,276 (5 houses)** | **15,837 (9.10%)** |

## 2. Leakage-Safety and Weak Supervision Rules

- **Zero Cross-Household Leakage:** Every household is assigned strictly to a single split. All 510-point windows inherit their household's split unconditionally.
- **Weak-Label Independence:** The weak training label is derived solely from metadata-level Kettle presence indicator (1 = positive presence, 0 = unmonitored). Timestamp-level sub-meter measurements are strictly excluded from model training.
- **Test Split Freeze:** Holdout test households (H2, H13) are strictly frozen. They must never be accessed for model training, hyperparameter tuning, or threshold selection.
- **House 12 Handling:** H12 provides weak supervision presence label (1) during training, but remains strictly excluded from primary localization evaluation (`is_evaluation_eligible=False`).
- **Unmonitored Households:** H1, H10, H15, H16, H18 are all placed in the TRAIN split to provide unmonitored negative weak-supervision examples.

## 3. Household-by-Household Allocation Table

| House | Split | Weak Label | Weak Label Meaning | Kettle Channel | 510-Pt Windows | Strong Target Windows (Audit Only) | Strong Eval Eligible |
|---|---|---|---|---|---|---|---|
| House 1 | **TRAIN** | 0 | Negative (Unmonitored) | None | 10,388 | 0 | No (Excluded/None) |
| House 2 | **TEST** | 1 | Positive (Presence) | Appliance8 | 8,730 | 2,053 | Yes |
| House 3 | **TRAIN** | 1 | Positive (Presence) | Appliance9 | 10,246 | 125 | Yes |
| House 4 | **VALIDATION** | 1 | Positive (Presence) | Appliance9 | 10,208 | 1,441 | Yes |
| House 5 | **TRAIN** | 1 | Positive (Presence) | Appliance8 | 7,651 | 1,798 | Yes |
| House 6 | **TRAIN** | 1 | Positive (Presence) | Appliance7 | 8,423 | 81 | Yes |
| House 7 | **TRAIN** | 1 | Positive (Presence) | Appliance9 | 10,187 | 54 | Yes |
| House 8 | **TRAIN** | 1 | Positive (Presence) | Appliance9 | 9,362 | 2,264 | Yes |
| House 9 | **TRAIN** | 1 | Positive (Presence) | Appliance7 | 9,154 | 2,120 | Yes |
| House 10 | **TRAIN** | 0 | Negative (Unmonitored) | None | 9,120 | 0 | No (Excluded/None) |
| House 11 | **TRAIN** | 1 | Positive (Presence) | Appliance7 | 6,807 | 1,838 | Yes |
| House 12 | **TRAIN** | 1 | Positive (Presence) | Appliance6 | 8,758 | 0 | No (Excluded/None) |
| House 13 | **TEST** | 1 | Positive (Presence) | Appliance9 | 5,778 | 145 | Yes |
| House 15 | **TRAIN** | 0 | Negative (Unmonitored) | None | 9,387 | 0 | No (Excluded/None) |
| House 16 | **TRAIN** | 0 | Negative (Unmonitored) | None | 8,316 | 0 | No (Excluded/None) |
| House 17 | **VALIDATION** | 1 | Positive (Presence) | Appliance8 | 8,353 | 2,257 | Yes |
| House 18 | **TRAIN** | 0 | Negative (Unmonitored) | None | 8,065 | 0 | No (Excluded/None) |
| House 19 | **TRAIN** | 1 | Positive (Presence) | Appliance5 | 8,253 | 57 | Yes |
| House 20 | **TRAIN** | 1 | Positive (Presence) | Appliance9 | 8,419 | 1,595 | Yes |
| House 21 | **TRAIN** | 1 | Positive (Presence) | Appliance7 | 8,467 | 9 | Yes |
