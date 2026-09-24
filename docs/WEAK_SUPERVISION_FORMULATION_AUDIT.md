# Weak-Supervision & Formulation Audit: Pre-Experiment 3 Diagnostic Report

**Document ID:** `AUDIT-E3-001`  
**Date:** 2026-09-24 (Revised 2026-09-25)  
**Author:** Implementation Engineer  
**Status:** Formulation Locked for Experiment 3 Implementation  
**Target Scope:** TRAIN households (16) and Validation households (H4, H17). Final test households (H2, H13) remain strictly isolated and sealed.  
**Reference Commit:** `1b4590f2278ee401681dab54892f32c38554d514`

---

## 1. Executive Summary

This diagnostic audit investigates why the initial weakly supervised baseline models (Simple 1D CNN baseline and single 1D ResNet reference) produced **near-all-positive window predictions** on validation data (CNN: 100% positive predictions, ResNet: 88.57% positive predictions), yielding high recall ($>94.9\%$) but low precision ($\approx 20\%$) and trivial balanced accuracy ($\approx 50-54\%$).

### Key Findings
1. **Weak-Label Semantic Gap:** Weak supervision assigns a binary label $y=1$ to every window from a household where a Kettle is monitored. On validation households (H4 and H17), **80.08% of weak-positive windows contain zero Kettle activity** (14,863 out of 18,561 windows are purely inactive background power).
2. **Extreme Temporal Sparsity:** Within windows containing active Kettle events, the actual Kettle burst accounts for an average of only **15.42 sample points (123.4 seconds / ~2.06 minutes)** out of the 510-point window (**3.02% duty cycle**). Across all validation timestamps, Kettle activity occupies only **0.6024% of all points**.
3. **Loss Pressure on Generic Background:** In the training split, 67.89% of windows are labeled $y=1$ and 32.11% are labeled $y=0$. Because ~80–90%+ of windows in positive households contain only background base load (refrigerators, standby power, lighting), the training process assigns positive labels to tens of thousands of generic background windows. Under Binary Cross-Entropy (BCE) loss with Global Average Pooling (GAP), this creates substantial supervisory pressure to associate generic household power patterns with the positive class.
4. **Resolution for Experiment 3:** Experiment 3 maintains two distinct input representations: (1) train-only standardized input $x_{\text{model}}(t)$ for ResNet detection and CAM extraction, and (2) nonnegative kW-scaled input $x_{\text{attn}}(t) = X(t) / 1000$ for the reference attention-sigmoid transformation $S(t) = \sigma(CAM_{\text{ens}}(t) \cdot x_{\text{attn}}(t))$ with deterministic zero-power suppression ($X(t) == 0 \implies \hat{y}_t = 0$; for $X(t) > 0$, $\hat{y}_t = 1 \iff S(t) \ge 0.50$).

---

## 2. Weak-Label Construction Verification

### 2.1 Training Split Household Inventory & Class Ratios

The training split contains **141,003 windows** across **16 households**.

- **Weak Positive Windows ($y=1$, Monitored Kettle):** 95,727 windows (**67.89%**) across 11 households.
- **Weak Negative Windows ($y=0$, Unmonitored Kettle):** 45,276 windows (**32.11%**) across 5 households.
- **Aggregate Class Ratio:** **2.114 : 1** (Positive : Negative).

| Household ID | Split | Weak Label ($y$) | Label Semantic | Kettle Sub-meter Channel | Total 510-pt Windows | % of Train Windows | Known Active Windows | Inactive Rate in $y=1$ Homes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **H01** | Train | 0 | Negative (Unmonitored) | None (unmonitored) | 10,388 | 7.37% | 0 | N/A |
| **H03** | Train | 1 | Positive (Monitored) | Appliance9 | 10,246 | 7.27% | 125 | 98.78% |
| **H05** | Train | 1 | Positive (Monitored) | Appliance8 | 7,651 | 5.43% | 1,798 | 76.50% |
| **H06** | Train | 1 | Positive (Monitored) | Appliance7 | 8,423 | 5.97% | 81 | 99.04% |
| **H07** | Train | 1 | Positive (Monitored) | Appliance9 | 10,187 | 7.22% | 54 | 99.47% |
| **H08** | Train | 1 | Positive (Monitored) | Appliance9 | 9,362 | 6.64% | 2,264 | 75.82% |
| **H09** | Train | 1 | Positive (Monitored) | Appliance7 | 9,154 | 6.49% | 2,120 | 76.84% |
| **H10** | Train | 0 | Negative (Unmonitored) | None (unmonitored) | 9,120 | 6.47% | 0 | N/A |
| **H11** | Train | 1 | Positive (Monitored) | Appliance7 | 6,807 | 4.83% | 1,838 | 72.99% |
| **H12** | Train | 1 | Positive (Monitored)* | Appliance6 | 8,758 | 6.21% | 0 | 100.00% |
| **H15** | Train | 0 | Negative (Unmonitored) | None (unmonitored) | 9,387 | 6.66% | 0 | N/A |
| **H16** | Train | 0 | Negative (Unmonitored) | None (unmonitored) | 8,316 | 5.90% | 0 | N/A |
| **H18** | Train | 0 | Negative (Unmonitored) | None (unmonitored) | 8,065 | 5.72% | 0 | N/A |
| **H19** | Train | 1 | Positive (Monitored) | Appliance5 | 8,253 | 5.85% | 57 | 99.31% |
| **H20** | Train | 1 | Positive (Monitored) | Appliance9 | 8,419 | 5.97% | 1,595 | 81.05% |
| **H21** | Train | 1 | Positive (Monitored) | Appliance7 | 8,467 | 6.00% | 9 | 99.89% |
| **Total Train** | | | **11 Pos / 5 Neg** | | **141,003** | **100.00%** | **9,941** | **89.62%** |

*\*Note: H12 was designated evaluation-ineligible in D-005 due to inconsistent sub-meter logging, but retains weak-positive presence metadata.*

### 2.2 Assignment Rules and Semantic Scope

1. **Weak-Positive Window ($y=1$):** Assigned if and only if household REFIT metadata indicates that a Kettle was monitored on one of the 9 sub-metered appliance channels.
2. **Weak-Negative / Unmonitored Window ($y=0$):** Assigned if household REFIT metadata does not list a Kettle among monitored channels.
3. **What These Labels Mean:**
   - $y=1$ indicates that the household contains a monitored Kettle somewhere in the dwelling during the recording campaign.
   - $y=0$ indicates that no Kettle was monitored by an Individual Appliance Monitor (IAM) in that dwelling.
4. **What These Labels Do NOT Prove:**
   - A weak-positive label $y=1$ **does NOT prove** that the Kettle is turned on during that specific window.
   - A weak-negative label $y=0$ **does NOT prove** the total physical absence of a kettle in the home (an unmonitored kettle or gas kettle might exist).
   - Most critically, **$y=1$ is a dwelling-level metadata presence indicator, not a signal-level activation tag.**

---

## 3. Weak-Label Informativeness Audit on Validation (H4 and H17)

Validation households **H4** and **H17** were audited against their exact strong sub-meter Kettle ground-truth targets ($P_{\text{kettle}} \ge 1500\text{ W}$, continuity gap $\le 20\text{ s}$, max duration $\le 600\text{ s}$).

| Metric | Household H04 | Household H17 | Combined Validation (H4 + H17) |
|---|:---:|:---:|:---:|
| **Total 510-point Windows** | 10,208 | 8,353 | **18,561** |
| **Weak Label Assigned ($y$)** | 1 (100.0%) | 1 (100.0%) | **1 (100.0%)** |
| **Strong Active Windows (Target Present)** | 1,441 (14.12%) | 2,257 (27.02%) | **3,698 (19.92%)** |
| **Strong Inactive Windows (Zero Target Activity)** | 8,767 (85.88%) | 6,096 (72.98%) | **14,863 (80.08%)** |
| **(a) Fraction of Weak-Positives with Target Activity** | **14.12%** | **27.02%** | **19.92%** |
| **(b) Fraction of Weak-Positives with NO Target Activity** | **85.88%** | **72.98%** | **80.08%** |
| **(c) Fraction of Weak-Negatives with Target Activity** | *Not Measurable on Split* | *Not Measurable on Split* | ***Not Measurable on Split*** |

### Note on Weak-Negative Target Contamination
Both validation households (H4 and H17) are weak-positive dwellings. Therefore, **weak-negative target contamination was not measurable on this validation split** because there are zero weak-negative validation windows. This represents an unmeasured validation condition, not a proof of zero contamination in unmonitored homes.

---

## 4. Window / Temporal Context Mismatch Audit

### 4.1 Window Duration & Timestamp Span

- **Window Length:** 510 sample points.
- **Sampling Interval:** 8.0 seconds nominal.
- **Timestamp Span:** $(510 - 1) \times 8.0\text{ seconds} = \mathbf{4,072\text{ seconds}}$ ($\approx 67.87\text{ minutes}$ / ~68 minutes).

### 4.2 Target Distribution within Active Windows (H4 & H17)

For the 3,698 validation windows containing verified Kettle events:

| Distribution Statistic | Household H04 | Household H17 | Combined Validation |
|---|:---:|:---:|:---:|
| **Mean Active Points per Active Window** | 18.41 points | 13.51 points | **15.42 points** |
| **Median Active Points per Active Window** | 16.0 points | 12.0 points | **14.0 points** |
| **Standard Deviation** | 11.03 points | 7.29 points | **9.25 points** |
| **Min / Max Active Points** | 1 / 203 points | 1 / 81 points | **1 / 203 points** |
| **Mean Active Duration per Active Window** | 147.2 s (~2.45 min) | 108.1 s (~1.80 min) | **123.4 s (~2.06 min)** |
| **Active Fraction of Window (Duty Cycle)** | 3.61% | 2.65% | **3.02%** |
| **Inactive Background Fraction of Window** | 96.39% | 97.35% | **96.98%** |
| **Mean Discrete Events per Active Window** | 1.20 events (max 8) | 1.41 events (max 8) | **1.33 events (max 8)** |

### 4.3 Global Point-Level Target Sparsity

Across the entire validation split ($18,561 \times 510 = 9,466,110$ total timestamp points):
- **Total Active Kettle Points:** $26,522\text{ (H4)} + 30,500\text{ (H17)} = \mathbf{57,022\text{ points}}$.
- **Global Validation Point-Level Target Density:** $\frac{57,022}{9,466,110} = \mathbf{0.6024\%}$.
- **Global Inactive Background Density:** $\mathbf{99.3976\%}$.

### 4.4 Coarseness Assessment

A 4,072-second window context is **fundamentally coarse** when supervised solely by dwelling-level presence metadata:
- A single Kettle boil lasts $\approx 1.5 - 3.0$ minutes (12–25 points).
- In an active window, ~97% of the timestamp points are background power.
- In ~80% of positive windows, 100% of the points are background power.
- Supervised binary classification against $y=1$ forces Global Average Pooling (GAP) to aggregate across the full 4,072 seconds, blending background load with sparse bursts.

---

## 5. Root-Cause Analysis of Near-All-Positive Predictions

### 5.1 Measured Baseline Results

The baseline models evaluated on the 18,561 validation windows exhibited the following behavior:

| Model | Decision Threshold | Predicted Positives | Predicted Negatives | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Precision | Recall | Balanced Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1D CNN Baseline** | 0.50 | 18,561 (100.0%) | 0 (0.0%) | 3,698 | 14,863 | 0 | 0 | 19.92% | 100.0% | 50.00% |
| **1D ResNet Reference** | 0.50 | 16,440 (88.57%) | 2,121 (11.43%) | 3,512 | 12,928 | 1,935 | 186 | 21.36% | 94.97% | 53.99% |
| **Trivial Majority Baseline** | — | 0 (0.0%) | 18,561 (100.0%) | 0 | 0 | 14,863 | 3,698 | 0.00% | 0.00% | 50.00% |

### 5.2 Separation of Measured Evidence vs Theoretical Inference

#### Measured Facts (Empirical Evidence)
1. In the training split, 67.89% of windows are labeled $y=1$ ($95,727$) and 32.11% are labeled $y=0$ ($45,276$).
2. Monitored ($y=1$) and unmonitored ($y=0$) training households share generic residential base loads (refrigerators, standby devices, baseline consumption in the 50–300 W range).
3. In positive households, 80.08% (val) to 89.62% (train) of 4,072-second windows contain no active Kettle usage.
4. The BCE loss penalizes predictions on background windows from positive households as false negatives ($y=1$), while penalizing predictions on background windows from unmonitored households as false positives ($y=0$).

#### Theoretical Inference (Mechanism Assessment)
*(Marked as architectural/optimization inference based on observed empirical data)*:
- Because the weak-label construction assigns positive labels to tens of thousands of inactive, generic background windows, the network experiences substantial BCE loss pressure to associate generic household context (e.g., typical base load profiles) with the positive class.
- When trained with Global Average Pooling across the entire 4,072-second context, features that capture generic residential power patterns receive positive gradient updates more frequently than negative updates due to the 68% positive training prior.
- Consequently, at validation time against households H4 and H17 (where 100% of windows are from monitored homes and 80.08% contain only background power), the network outputs high probabilities on background power, classifying 88.57% to 100.0% of windows as positive.

---

## 6. Official CamAL Literature Context & Project Adaptations

### 6.1 CAM Sign Handling
- **Published Normalization Operation:** Section IV-B4 of the CamAL paper specifies:
  $$\widetilde{\text{CAM}}^{(i)}(t) = \frac{\text{CAM}^{(i)}(t)}{\max_t \text{CAM}^{(i)}(t)}$$
- **Sign Semantics:** The published normalization formula does not introduce an explicit ReLU/clamping step. Because the published method describes the normalized CAM as $[0, 1]$, the implementation must preserve the intended nonnegative attention behavior; our project therefore does not add an unverified ReLU operation.

### 6.2 Distinct Input Representations
Our project explicitly distinguishes between two input representations:
1. **Neural-Network Model Input:**
   $$x_{\text{model}}(t) = \frac{X(t) - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   This uses the project's existing, locked train-only z-score standardization ($\mu_{\text{train}} = 523.94\text{ W}, \sigma_{\text{train}} = 764.91\text{ W}$, Decision D-007) for stable ResNet training.
2. **CAM Attention Input:**
   $$x_{\text{attn}}(t) = \frac{X(t)}{1000}$$
   This uses the nonnegative aggregate power scaled by $1/1000$ (kW), following the input scaling described by CamAL (Section V-B, line 421).
3. **Rationale for Separation:**
   CamAL explicitly states that aggregate input consumption is scaled by dividing by 1000 before its processing pipeline. Using z-score input in the attention stage would allow negative CAM $\times$ negative input to become positive and would materially change the localization behavior.
4. **Honest Attribution:** This separation is an **explicit project implementation adaptation** and not a claim of byte-for-byte reproduction of every unverified detail.

---

## 7. Reference CamAL Localization Pipeline for Experiment 3

```
Input Signal X (510 points, 4,072s)
        │
        ├──► Model Input: x_model(t) = (X(t) - μ_train) / σ_train
        └──► Attention Input: x_attn(t) = X(t) / 1000 (kW, ≥ 0)
                │
        ┌───────┴────────────────────────────────────────┐
        │ Run Multi-Scale Ensemble on x_model:          │
        │ ResNet(k=5), ResNet(k=7), ..., ResNet(k=25)    │
        └───────┬────────────────────────────────────────┘
                │
                ├──► Ensemble Probability: P_ens = (1/M) ∑ σ(logit_m)
                │       │
                │       ├── If P_ens < 0.50 ──► Undetected: ŷ_t = 0 (all t ∈ [1, 510])
                │       └── If P_ens ≥ 0.50 ──► Detected (extract CAMs)
                │
                └──► CAM Extraction & Attention:
                        1. CAM_norm^(m)(t) = CAM^(m)(t) / max_t CAM^(m)(t)
                           (If max_t CAM^(m)(t) ≤ 0, CAM_norm^(m)(t) = 0 for all t)
                        2. Ensemble Average: CAM_ens(t) = (1/M) ∑ CAM_norm^(m)(t)
                        3. Attention-Sigmoid: S(t) = σ( CAM_ens(t) · x_attn(t) )
                        4. Binary Thresholding with Zero-Power Suppression:
                           - If X(t) == 0 ──► ŷ_t = 0
                           - If X(t) > 0  ──► ŷ_t = 1 if S(t) ≥ 0.50 else 0
```

---

## 8. CAM Alignment, Degenerate Cases, and Timeline Stitching

### 8.1 Temporal Resolution & Feature Map Alignment
- **Architecture Resolution:** All convolutional layers in `ResNetBlock1D` operate with `stride=1` and `padding="same"`. The feature map $A_c(t)$ from the final block prior to Global Average Pooling has dimension $(B, 128, 510)$.
- **Temporal Alignment:** The feature map temporal length exactly matches the 510-point input window ($T=510$). **No temporal interpolation, resizing, or downsampling alignment is required.**

### 8.2 Zero-Max CAM Handling
- If $\max_t CAM(t) \le 0$, $CAM_{\text{norm}}(t)$ is defined as **all zeros ($0.0$ for all $t \in [1, 510]$)**, preventing division by zero.

### 8.3 Window-to-Timeline Stitching Rules
- **Segment-Bounded Non-Overlapping Windows:** Processed windows are non-overlapping ($510$-point step) generated strictly within continuous recording segments (where inter-sample gaps $\le 16\text{ s}$).
- **Forbidden Boundary Preservation:** Windows **never cross segment boundaries** (raw gaps $> 16\text{ s}$).
- **Timeline Stitching:**
  1. For each recording segment $s$ of household $h$, window predictions $[\hat{y}_1^{(w)}, \dots, \hat{y}_{510}^{(w)}]$ are placed contiguously at their recorded start index within segment $s$.
  2. Any unwindowed segment tail points (leftover samples $< 510$ at the end of a segment) or segments shorter than 510 points receive default inactive state $0$.
  3. This produces a continuous, gap-consistent household timeline array $\hat{Y}_{h, s}[t]$ aligned with ground-truth target timestamps.

---

## 9. Locked Metric Specification for Experiment 3

### 9.1 Primary Metric: Point-Level Localization F1 (Gate A Acceptance)

Point-level Localization F1 evaluated across all 9,466,110 validation points ($18,561 \times 510$) is the **sole quantitative criterion** for Gate A evaluation (acceptance target: $\text{Localization F1} \ge 0.50$).

For every validation window $w \in \{1, \dots, 18561\}$ and timestamp $t \in \{1, \dots, 510\}$:
- **Ground Truth Mask:** $y_t \in \{0, 1\}$ (from sub-meter target definition $P_{\text{kettle}} \ge 1500\text{ W}$, gap $\le 20\text{ s}$, duration $\le 600\text{ s}$).
- **Predicted Mask:** $\hat{y}_t \in \{0, 1\}$ (from reference CamAL localization pipeline).
- **Point-Level Contingency:**
  - $TP_{\text{pt}} = \sum \mathbb{I}(\hat{y}_t = 1 \land y_t = 1)$
  - $FP_{\text{pt}} = \sum \mathbb{I}(\hat{y}_t = 1 \land y_t = 0)$
  - $FN_{\text{pt}} = \sum \mathbb{I}(\hat{y}_t = 0 \land y_t = 1)$
  - $TN_{\text{pt}} = \sum \mathbb{I}(\hat{y}_t = 0 \land y_t = 0)$
- **Metrics:**
  $$\text{Precision}_{\text{pt}} = \frac{TP_{\text{pt}}}{TP_{\text{pt}} + FP_{\text{pt}}}, \quad \text{Recall}_{\text{pt}} = \frac{TP_{\text{pt}}}{TP_{\text{pt}} + FN_{\text{pt}}}$$
  $$\text{Localization F1}_{\text{pt}} = \frac{2 \cdot \text{Precision}_{\text{pt}} \cdot \text{Recall}_{\text{pt}}}{\text{Precision}_{\text{pt}} + \text{Recall}_{\text{pt}}}$$

### 9.2 Secondary Metric: Stitched Deterministic 1-to-1 Event-Level F1 (Reporting)

Event-level metrics are computed on the **stitched segment timeline**:

1. **Event Extraction:**
   - Contiguous runs of active points ($y=1$) separated by gaps $\le 20\text{ s}$ form ground-truth events $E_{\text{gt}} = [t_{\text{start}}, t_{\text{end}}]$.
   - Contiguous runs of predicted active points ($\hat{y}=1$) separated by gaps $\le 20\text{ s}$ form predicted candidate events $E_{\text{pred}} = [\hat{t}_{\text{start}}, \hat{t}_{\text{end}}]$.
2. **Temporal Intersection over Union (IoU):**
   $$\text{IoU}(E_{\text{gt}}, E_{\text{pred}}) = \frac{|E_{\text{gt}} \cap E_{\text{pred}}|}{|E_{\text{gt}} \cup E_{\text{pred}}|}$$
3. **Deterministic 1-to-1 Matching Protocol:**
   - All candidate pairs $(E_{\text{gt}}, E_{\text{pred}})$ with $\text{IoU} \ge 0.50$ are sorted in descending order of IoU.
   - Matches are assigned greedily: each ground-truth event and each predicted event is matched at most once.
   - Matched pairs are True Positives ($TP_{\text{event}}$).
   - Unmatched predicted events are False Positives ($FP_{\text{event}}$).
   - Unmatched ground-truth events are False Negatives ($FN_{\text{event}}$).
4. **Event-Level Scores:**
   $$\text{Event Precision} = \frac{TP_{\text{event}}}{TP_{\text{event}} + FP_{\text{event}}}, \quad \text{Event Recall} = \frac{TP_{\text{event}}}{TP_{\text{event}} + FN_{\text{event}}}, \quad \text{Event F1} = \frac{2 \cdot P_{\text{ev}} \cdot R_{\text{ev}}}{P_{\text{ev}} + R_{\text{ev}}}$$

---

## 10. Summary & Sign-off

- The formulation, normalization rules ($CAM / \max(CAM)$), dual input representations ($x_{\text{model}}$ and $x_{\text{attn}}$), and deterministic matching protocol are formally locked.
- Decision **D-016** in `docs/DECISIONS.md` is updated and locked.
- Pre-training unit tests in `tests/test_localization.py` validate all pipeline components and distinct input properties.
