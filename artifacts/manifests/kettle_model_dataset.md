# Model-Ready Weakly Supervised Dataset Specification (Kettle)

**Dataset Version:** `v1.0.0` (primary_household_split_16_2_2)  
**Input Tensor Shape:** `(510,)` (`float32`)  
**Input Domain:** aggregate_power_watts (raw Watts)  
**Normalization Policy:** unnormalized_raw_watts (train-derived transform hook ready)  
**Storage Format:** Memory-mapped NumPy binary arrays (.npy) / indexed streaming generator  
**Target Separation:** **Strong Kettle targets strictly excluded from model inputs**

## 1. Split Allocation & Class Balance Summary

| Split | Households | Total Windows | Weak Positive (Label=1) | Weak Negative / Unmonitored (Label=0) | Weak Positive % | Strong Active Targets (Eval Only) |
|---|---|---|---|---|---|---|
| **TRAIN** | 16 (H1, H3, H5, H6, H7, H8, H9, H10, H11, H12, H15, H16, H18, H19, H20, H21) | 141,003 | 95,727 | 45,276 | 67.89% | 9,941 (7.05%) |
| **VALIDATION** | 2 (H4, H17) | 18,561 | 18,561 | 0 | 100.00% | 3,698 (19.92%) |
| **TEST (Frozen)** | 2 (H2, H13) | 14,508 | 14,508 | 0 | 100.00% | 2,198 (15.15%) |
| **TOTAL** | **20** | **174,072** | **128,796** | **45,276** | **73.99%** | **15,837 (9.10%)** |

## 2. Validation-Split Sanity Check

- **Weak Validation Single-Class Notice:** `weak_validation_is_single_class = true`. Both validation households (H4, H17) indicate positive Kettle presence in metadata. Weak-label classification accuracy on validation is therefore non-discriminative and MUST NOT be used as the primary model selection criterion.
- **Strong Target Coverage in Validation:** Ground-truth sub-meter evaluation contains **3,698 active windows (19.92%)** and **14,863 inactive windows (80.08%)**, confirming that validation contains rich active/inactive event diversity for post-training CAM localization evaluation without exposing sub-meter targets to model training loss.

## 3. Strict Methodological Protections

1. **Input Exclusivity:** The model tensor contains strictly `aggregate_w[510]`. No `household_id`, no `segment_id`, no timestamps, and no sub-meter measurements are concatenated.
2. **Weak Label Definition:** `weak_kettle_presence ∈ {0, 1}` represents household-level presence from metadata (15 positive houses, 5 unmonitored houses). It does not represent timestamp-level state.
3. **Train-Derived Normalization:** Normalization transforms must fit parameters (`mean`, `std`, `min`, `max`) exclusively on the TRAIN partition.
4. **Holdout Test Isolation:** Test households H2 and H13 are sealed and forbidden from all model training and tuning.

## 4. Household Shard Manifest

| House | Split | Weak Label | Total 510-Pt Windows | Storage File |
|---|---|---|---|---|
| House 1 | **TRAIN** | 0 | 10,388 | `house_1_aggregate.npy` |
| House 2 | **TEST** | 1 | 8,730 | `house_2_aggregate.npy` |
| House 3 | **TRAIN** | 1 | 10,246 | `house_3_aggregate.npy` |
| House 4 | **VALIDATION** | 1 | 10,208 | `house_4_aggregate.npy` |
| House 5 | **TRAIN** | 1 | 7,651 | `house_5_aggregate.npy` |
| House 6 | **TRAIN** | 1 | 8,423 | `house_6_aggregate.npy` |
| House 7 | **TRAIN** | 1 | 10,187 | `house_7_aggregate.npy` |
| House 8 | **TRAIN** | 1 | 9,362 | `house_8_aggregate.npy` |
| House 9 | **TRAIN** | 1 | 9,154 | `house_9_aggregate.npy` |
| House 10 | **TRAIN** | 0 | 9,120 | `house_10_aggregate.npy` |
| House 11 | **TRAIN** | 1 | 6,807 | `house_11_aggregate.npy` |
| House 12 | **TRAIN** | 1 | 8,758 | `house_12_aggregate.npy` |
| House 13 | **TEST** | 1 | 5,778 | `house_13_aggregate.npy` |
| House 15 | **TRAIN** | 0 | 9,387 | `house_15_aggregate.npy` |
| House 16 | **TRAIN** | 0 | 8,316 | `house_16_aggregate.npy` |
| House 17 | **VALIDATION** | 1 | 8,353 | `house_17_aggregate.npy` |
| House 18 | **TRAIN** | 0 | 8,065 | `house_18_aggregate.npy` |
| House 19 | **TRAIN** | 1 | 8,253 | `house_19_aggregate.npy` |
| House 20 | **TRAIN** | 1 | 8,419 | `house_20_aggregate.npy` |
| House 21 | **TRAIN** | 1 | 8,467 | `house_21_aggregate.npy` |
