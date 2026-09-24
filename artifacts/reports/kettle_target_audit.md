# Kettle Target Quality & Activation Profile Audit Report

## 1. Scope & Metadata Verification

| Item | Measured Finding |
|---|---|
| **Target Appliance** | **Kettle** |
| **Positive (Monitored) Households** | 15 (H2, H3, H4, H5, H6, H7, H8, H9, H11, H12, H13, H17, H19, H20, H21) |
| **Negative (Unmonitored) Households** | 5 (H1, H10, H15, H16, H18) |
| **Total 500W Descriptive Activations** | 42,665 events (5.27 / day / house) |
| **Total 1500W Standard Activations** | 40,638 events (5.02 / day / house) |

## 2. Positive Household Kettle Signal & Activation Profile

| House | Channel | Model | Events (500W) | Events/Day | Mean Dur (s) | Median Dur (s) | Peak Mean (W) | Idle Zero % | Non-Zero Max Dur (s) |
|---|---|---|---|---|---|---|---|---|---|
| House 2 | Appliance8 | Unknown Unknown | 3,587 | 5.81 | 86.3s | 91.0s | 2760 W | 98.9723% | 7171s |
| House 3 | Appliance9 | Dualit JKt3 | 4,441 | 7.23 | 107.0s | 89.0s | 1820 W | 98.7324% | 1381s |
| House 4 | Appliance9 | Swan Unknown | 2,004 | 3.16 | 122.3s | 122.0s | 1935 W | 99.3693% | 1769s |
| House 5 | Appliance8 | Logik L17SKC14 | 4,576 | 7.06 | 82.3s | 79.0s | 2750 W | 99.0509% | 7560s |
| House 6 | Appliance7 | ASDA GPK101W | 4,716 | 8.17 | 71.1s | 69.0s | 2653 W | 98.9215% | 870s |
| House 7 | Appliance9 | Sainsburys 121988254 | 1,587 | 2.59 | 105.1s | 105.0s | 2242 W | 99.542% | 555s |
| House 8 | Appliance9 | Morphy Richards 43615 | 3,593 | 6.47 | 80.7s | 91.0s | 2768 W | 99.1142% | 288s |
| House 9 | Appliance7 | Russel Hobbs Unknown | 3,322 | 5.85 | 116.9s | 124.0s | 2750 W | 98.8444% | 3199s |
| House 11 | Appliance7 | Unknown Unknown | 2,769 | 7.06 | 106.4s | 98.0s | 2125 W | 98.822% | 399s |
| House 12 | Appliance6 | Unknown Unknown | 14 | 0.03 | 10.6s | 8.0s | 1234 W | 95.2523% | 48832s |
| House 13 | Appliance9 | Unknown Unknown | 1,256 | 2.52 | 164.8s | 121.0s | 2581 W | 98.8948% | 88653s |
| House 17 | Appliance8 | Russel Hobbs 17869 | 4,737 | 10.08 | 90.7s | 89.0s | 2447 W | 98.3293% | 23525s |
| House 19 | Appliance5 | Breville VKJ336 | 2,684 | 5.7 | 64.0s | 62.0s | 2972 W | 99.3877% | 212s |
| House 20 | Appliance9 | Unknown Unknown | 2,100 | 4.56 | 74.9s | 87.0s | 2796 W | 99.4378% | 946s |
| House 21 | Appliance7 | Unknown Unknown | 1,279 | 2.61 | 96.7s | 102.0s | 1965 W | 99.4542% | 31185s |

## 3. Aggregate Context & Relationship During Kettle ON

| House | Aggregate Mean (W) | Agg > 4000W % | Wrap-around Glitches (>60kW) | Issues % | Gaps > 30s | Mean Agg During Kettle ON (W) | Mean Background Load (W) | Kettle > Agg Deficit % |
|---|---|---|---|---|---|---|---|---|
| House 2 | 465.1 W | 1.07% | 0 | 0.50% | 7652 | 3334 W | 624 W | 9.15% |
| House 3 | 678.5 W | 1.22% | 5 | 5.84% | 4471 | 2386 W | 576 W | 24.06% |
| House 4 | 381.2 W | 0.12% | 4 | 1.00% | 5124 | 2483 W | 573 W | 7.56% |
| House 5 | 738.1 W | 1.13% | 0 | 5.73% | 13810 | 3703 W | 980 W | 12.93% |
| House 6 | 482.9 W | 0.11% | 0 | 0.55% | 6681 | 3156 W | 543 W | 11.79% |
| House 7 | 565.9 W | 0.99% | 0 | 2.40% | 8171 | 3480 W | 1274 W | 12.94% |
| House 8 | 685.6 W | 1.95% | 0 | 0.41% | 8436 | 3672 W | 934 W | 10.84% |
| House 9 | 576.0 W | 1.65% | 0 | 0.52% | 8069 | 3678 W | 960 W | 5.70% |
| House 11 | 461.2 W | 0.04% | 0 | 0.91% | 7498 | 2029 W | -52 W | 34.77% |
| House 12 | 367.2 W | 0.36% | 0 | 0.24% | 2850 | 184 W | -940 W | 100.00% |
| House 13 | 557.8 W | 0.88% | 0 | 2.61% | 12025 | 2628 W | 80 W | 39.11% |
| House 17 | 405.7 W | 0.55% | 1 | 1.58% | 7948 | 2389 W | 194 W | 20.47% |
| House 19 | 291.3 W | 0.12% | 0 | 1.11% | 8672 | 2957 W | 14 W | 57.65% |
| House 20 | 378.0 W | 0.18% | 0 | 0.38% | 6832 | 3439 W | 669 W | 10.07% |
| House 21 | 646.8 W | 0.22% | 2 | 3.84% | 6772 | 2104 W | 174 W | 37.11% |

## 4. Household-Specific Target Risks & Nuances

| House | Identified Risk | CSV & Metadata Evidence | Recommended Handling |
|---|---|---|---|
| **House 3** | Kettle Hardware Change | Replaced with Vektra Vacuum Kettle on 16 Apr 2014 | Keep in dataset; record signature shift date |
| **House 11** | Solar PV Generation | Net daytime aggregate power reduced/distorted by solar panels | Reserve for validation / robustness evaluation |
| **House 17** | Shared IAM Channel | Kettle shares plug with toaster/misc items | Use >=1500W threshold for target ground truth |
| **House 21** | Shared IAM + Solar PV | Dual complexity: shared toaster plug + rooftop solar PV | Reserve for holdout / robustness evaluation |

## 5. Negative (Unmonitored) Households Verification

| House | Metadata Verification | Monitored Kettle IAM | Usable as Negative Weak-Supervision Example |
|---|---|---|---|
| House 1 | Confirmed unmonitored in metadata | None | **Yes** (Unmonitored / Weak-Negative Supervision State) |
| House 10 | Confirmed unmonitored in metadata | None | **Yes** (Unmonitored / Weak-Negative Supervision State) |
| House 15 | Confirmed unmonitored in metadata | None | **Yes** (Unmonitored / Weak-Negative Supervision State) |
| House 16 | Confirmed unmonitored in metadata | None | **Yes** (Unmonitored / Weak-Negative Supervision State) |
| House 18 | Confirmed unmonitored in metadata | None | **Yes** (Unmonitored / Weak-Negative Supervision State) |