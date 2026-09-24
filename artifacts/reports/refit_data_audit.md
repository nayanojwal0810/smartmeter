# REFIT Dataset Forensic Data Quality Audit Report

## 1. Executive Summary

| Metric | Value |
|---|---|
| **Total Households Audited** | 20 |
| **Total Uncompressed Volume** | 6.467 GB (6,943,411,063 bytes) |
| **Total Recorded Rows** | 119,495,879 |
| **Total Rows with Issues == 1** | 2,037,863 (1.7054%) |
| **Schema Conformance** | 100% Valid (All files match exact 13-column schema) |
| **Temporal Monotonicity** | Monotonic (no timestamp reversals) |

## 2. Sampling Interval & Gap Distribution

| Interval Category | Count / Percentage | Description |
|---|---|---|
| **<= 8 seconds** | 80.21% | Nominal target sampling rate |
| **9–15 seconds** | 19.34% | Minor polling / load-change delay |
| **> 15 seconds** | 0.44% | Irregular transmission gap |
| **> 30 seconds** | 0.13% | Medium gap |
| **> 60 seconds** | 0.05% | Extended gap |
| **> 120 seconds** | 0.01% | Substantial missing outage |
| **Maximum Observed Gap** | 5,358,903.0 s (62.02 days) | Maximum sensor/network outage |

## 3. Appliance Channel Distribution Across Households

| Appliance Name | Monitored Household Count | Household IDs |
|---|---|---|
| **Washing Machine** | 18 | H1, H2, H3, H5, H6, H7, H8, H9, H10, H11, H13, H15, H16, H17, H18, H19, H20, H21 |
| **Microwave** | 17 | H2, H3, H4, H5, H6, H8, H9, H10, H11, H12, H13, H13, H15, H17, H18, H19, H20 |
| **Television Site** | 16 | H1, H2, H3, H4, H5, H7, H8, H9, H10, H13, H15, H16, H17, H18, H19, H20 |
| **Dishwasher** | 15 | H1, H2, H3, H5, H6, H7, H9, H10, H11, H13, H15, H16, H18, H20, H21 |
| **Kettle** | 15 | H2, H3, H4, H5, H6, H7, H8, H9, H11, H12, H13, H17, H19, H20, H21 |
| **Fridge-Freezer** | 12 | H2, H3, H4, H5, H9, H10, H11, H12, H15, H17, H18, H21 |
| **Toaster** | 10 | H2, H3, H5, H6, H7, H8, H10, H12, H15, H19 |
| **Freezer** | 7 | H3, H4, H6, H8, H13, H17, H20 |
| **Tumble Dryer** | 7 | H3, H5, H7, H15, H17, H20, H21 |
| **Fridge** | 6 | H1, H4, H7, H8, H11, H20 |
| **Computer Site** | 6 | H11, H12, H15, H16, H17, H20 |
| **Hi-Fi** | 5 | H2, H9, H11, H15, H19 |
| **???** | 4 | H12, H12, H12, H13 |
| **Washer Dryer** | 3 | H1, H8, H9 |
| **Desktop Computer** | 3 | H4, H5, H18 |
| **Freezer(1)** | 2 | H1, H7 |
| **Freezer(2)** | 2 | H1, H7 |
| **Computer** | 2 | H1, H8 |
| **Electric Heater** | 2 | H1, H9 |
| **Television** | 2 | H12, H21 |
| **Overhead Fan** | 1 | H2 |
| **Washing Machine(1)** | 1 | H4 |
| **Washing Machine(2)** | 1 | H4 |
| **MJY Computer** | 1 | H6 |
| **TV/Satellite** | 1 | H6 |
| **PGM Computer** | 1 | H6 |
| **Magimix(Blender)** | 1 | H10 |
| **Chest Freezer** | 1 | H10 |
| **K Mix** | 1 | H10 |
| **Router** | 1 | H11 |
| **Network Site** | 1 | H13 |
| **Fridge-Freezer(1)** | 1 | H16 |
| **Fridge-Freezer(2)** | 1 | H16 |
| **Electric Heater(1)** | 1 | H16 |
| **Electric Heater(2)** | 1 | H16 |
| **Dehumidifier** | 1 | H16 |
| **TV Site(Bedroom)** | 1 | H17 |
| **Fridge(garage)** | 1 | H18 |
| **Freezer(garage)** | 1 | H18 |
| **Washer Dryer(garage)** | 1 | H18 |
| **Fridge Freezer** | 1 | H19 |
| **Bread-maker** | 1 | H19 |
| **Games Console** | 1 | H19 |
| **Food Mixer** | 1 | H21 |
| **Vivarium** | 1 | H21 |
| **Pond Pump** | 1 | H21 |

## 4. Household-by-Household Summary Table

| House | Rows | Start Date | End Date | Aggregate Mean (W) | Aggregate Max (W) | Issues % | Monotonic | Max Gap (h) |
|---|---|---|---|---|---|---|---|---|
| House 1 | 6,960,008 | 2013-10-09 | 2015-07-10 | 481.1 | 29159 | 0.84% | Yes | 998.4h |
| House 2 | 5,733,526 | 2013-09-17 | 2015-05-28 | 465.1 | 24595 | 0.50% | Yes | 1488.6h |
| House 3 | 6,994,594 | 2013-09-25 | 2015-06-02 | 678.5 | 65836 | 5.84% | Yes | 998.4h |
| House 4 | 6,760,511 | 2013-10-11 | 2015-07-07 | 381.2 | 65663 | 1.00% | Yes | 337.7h |
| House 5 | 7,430,755 | 2013-09-26 | 2015-07-06 | 738.1 | 41738 | 5.73% | Yes | 207.6h |
| House 6 | 6,241,971 | 2013-11-28 | 2015-06-28 | 482.9 | 32756 | 0.55% | Yes | 805.3h |
| House 7 | 6,756,034 | 2013-11-01 | 2015-07-08 | 565.9 | 32730 | 2.40% | Yes | 998.4h |
| House 8 | 6,118,469 | 2013-11-01 | 2015-05-10 | 685.6 | 25358 | 0.41% | Yes | 937.9h |
| House 9 | 6,169,525 | 2013-12-17 | 2015-07-08 | 576.0 | 28836 | 0.52% | Yes | 998.4h |
| House 10 | 6,739,284 | 2013-11-20 | 2015-06-30 | 776.2 | 37031 | 0.45% | Yes | 207.6h |
| House 11 | 4,431,541 | 2014-06-03 | 2015-06-30 | 461.2 | 32932 | 0.91% | Yes | 246.4h |
| House 12 | 5,859,544 | 2014-03-07 | 2015-07-08 | 367.2 | 16543 | 0.24% | Yes | 207.6h |
| House 13 | 4,737,371 | 2014-01-17 | 2015-05-31 | 557.8 | 25140 | 2.61% | Yes | 985.8h |
| House 15 | 6,225,696 | 2013-12-17 | 2015-07-08 | 254.8 | 8840 | 0.38% | Yes | 207.6h |
| House 16 | 5,722,544 | 2014-01-10 | 2015-07-08 | 555.7 | 68093 | 0.26% | Yes | 434.2h |
| House 17 | 5,431,577 | 2014-03-06 | 2015-06-19 | 405.7 | 68232 | 1.58% | Yes | 207.6h |
| House 18 | 5,007,721 | 2014-03-07 | 2015-05-24 | 449.6 | 29008 | 3.48% | Yes | 207.6h |
| House 19 | 5,622,610 | 2014-03-06 | 2015-06-20 | 291.3 | 8856 | 1.11% | Yes | 207.6h |
| House 20 | 5,168,605 | 2014-03-20 | 2015-06-23 | 378.0 | 32889 | 0.38% | Yes | 207.6h |
| House 21 | 5,383,993 | 2014-03-07 | 2015-07-10 | 646.8 | 65793 | 3.84% | Yes | 352.3h |