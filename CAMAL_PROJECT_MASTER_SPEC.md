# SmartMeter Appliance Intelligence & Localization Platform

## Project Master Specification / Single Source of Truth

**Status:** Selected project — execution planning phase  
**Primary dataset:** REFIT Electrical Load Measurements (Cleaned)  
**Research foundation:** Petralia et al., *Few Labels are All You Need: A Weakly Supervised Framework for Appliance Localization in Smart-Meter Series*, IEEE ICDE 2025  
**Primary implementation goal:** Build an independently implemented, production-oriented appliance detection and localization system inspired by the CamAL research approach, without claiming algorithmic novelty.

---

## 0. Mission

Build a complete ML system that takes **aggregate household electricity consumption** and estimates **when a selected appliance was active**, while training primarily from a **cheap household-level appliance-presence label** rather than timestamp-by-timestamp appliance labels.

The project must combine:

- time-series deep learning;
- weak supervision;
- CNN/ResNet-based classification;
- CAM-based temporal localization;
- rigorous household-level evaluation;
- data-quality validation;
- reproducible experiments;
- model serving;
- model versioning;
- monitoring and drift detection;
- a practical retraining workflow.

### What this project is NOT

This is **not** a claim that we invented the CamAL model or the weakly supervised localization idea. The research method is based on the published CamAL work and must be cited in the project documentation. Our project differentiates itself through independent implementation, disciplined evaluation, student-scale adaptation, and the production ML lifecycle built around the research idea.

The public project title must describe what we built and **must not use “CamAL” as the product/project name**. However, references and attribution must remain honest and explicit in technical documentation.

---

# 1. The Problem — Beginner Explanation

A normal smart meter tells us the **total electricity being used by a house**.

Example:

```text
House total power
      |
      +-- fridge
      +-- kettle
      +-- microwave
      +-- washing machine
      +-- lights
      +-- other appliances
```

So if total power suddenly rises, we do not directly know which appliance caused it.

This is the Non-Intrusive Load Monitoring (NILM) problem: infer individual appliance usage from the aggregate household signal.

## Two different questions

### Question A — Detection

> Was this appliance used in the given period?

Example:

```text
1-hour aggregate signal -> “Was the kettle present/used in this window?”
```

### Question B — Localization

> Exactly when inside the period was the appliance active?

Example:

```text
10:00 ------------------------------ 11:00
                 ^^^^^
                 kettle active
```

Question B is harder because it normally needs timestamp-level ground truth.

CamAL's central idea is to train using much cheaper labels and then use the model's explanation signal to localize the likely appliance activity in time.

---

# 2. Research Foundation: What the CamAL Paper Actually Does

## 2.1 Paper

**Title:** *Few Labels are all you need: A Weakly Supervised Framework for Appliance Localization in Smart-Meter Series*  
**Authors:** Adrien Petralia, Paul Boniol, Philippe Charpentier, Themis Palpanas  
**Venue:** IEEE ICDE 2025  
**DOI:** 10.1109/ICDE65448.2025.00329  
**arXiv:** https://arxiv.org/abs/2506.05895  
**Paper PDF:** https://helios2.mi.parisdescartes.fr/~themisp/publications/icde25-camal.pdf  
**Official code:** https://github.com/adrienpetralia/CamAL

The paper says CamAL combines an ensemble of deep-learning classifiers with an explainability method to localize appliance patterns while requiring only appliance-presence information for training.

Sources: CamAL paper and official repository.

## 2.2 The paper's central insight

Normal fully supervised NILM needs strong labels such as:

```text
10:00 -> washing machine OFF
10:01 -> OFF
10:02 -> ON
10:03 -> ON
10:04 -> OFF
...
```

Those labels are expensive to collect.

CamAL instead uses a much cheaper label such as:

```text
“This household has a washing machine.”
```

The model is then trained to detect whether the appliance is represented in a time window. After training, CAM is used to identify the parts of the time series that most contributed to that classification decision.

## 2.3 Weak supervision in this project

Think of three label levels:

| Label type | Example | Cost/availability |
|---|---|---|
| Strong label | exact appliance state at every timestamp | expensive |
| Weak window label | appliance was used somewhere in a window | cheaper |
| Possession label | appliance exists in the household | cheapest/simple |

The extreme weak-supervision setting studied by CamAL is the possession-only scenario: the practitioner only knows whether the appliance belongs to the household.

For our project, **timestamp-level appliance measurements are not allowed to become training labels** for the main weakly supervised model. They are reserved for evaluation/ground truth analysis and dataset auditing.

---

# 3. CamAL Model — Beginner-Friendly Technical Understanding

## 3.1 High-level pipeline

```text
Aggregate household power
          |
          v
   ResNet ensemble
          |
          v
 Appliance probability
          |
   if detected
          v
       CAMs
          |
          v
   Average CAMs
          |
          v
 Attention + sigmoid
          |
          v
 Appliance ON/OFF timeline
```

## 3.2 The CNN/ResNet role

The model receives a one-dimensional sequence of power readings.

A 1D CNN looks for patterns across time rather than looking at images.

Examples of useful temporal patterns:

- a sharp rise in power;
- a sustained high-power region;
- repeated switching behavior;
- combinations of local changes that may indicate a specific appliance.

A ResNet adds **skip/residual connections**, which help the network learn deeper feature transformations without making optimization unnecessarily difficult.

## 3.3 Architecture reported in the paper

The paper describes each CamAL ResNet as:

- 3 stacked residual blocks;
- each residual block contains 3 convolutional blocks;
- convolutional layers inside a block use kernel sizes `{8, 5, 3}`;
- residual block filter counts are `{64, 128, 128}`;
- global average pooling over time;
- linear layer;
- softmax classification.

The CamAL ensemble uses multiple ResNets with different top-level kernel sizes:

```text
Kp = {5, 7, 9, 15, 25}
```

The default ensemble size in the paper is **5 networks**.

The motivation is that different kernel sizes create different receptive fields, so the models can respond to temporal patterns at different scales.

Source: CamAL methodology section.

## 3.4 Training procedure reported in the paper

The paper's ensemble procedure is approximately:

```text
For each kernel size:
    run multiple training trials
    evaluate on validation data
    keep the strongest models

Select the best n models
Combine them as the ensemble
```

More specifically, the paper reports 3 trials for each kernel size and selects the 5 models with the lowest validation loss, with the default ensemble size n=5.

**Our student-scale implementation will not blindly reproduce every trial.** We will reproduce the core logic first, then increase the ensemble only when compute and runtime justify it.

## 3.5 How detection works

For a given input window:

```text
ResNet 1 -> probability
ResNet 2 -> probability
ResNet 3 -> probability
...
ResNet n -> probability

Average probabilities
        |
        v
Final detection probability
```

The paper gives an example threshold of 0.5 for deciding whether the appliance is detected in the current window.

We will treat **0.5 as a paper-reference baseline, not as a permanently fixed production threshold**. Threshold calibration will be an explicit experiment later.

## 3.6 How CAM localization works

CAM = Class Activation Map.

Very simply:

> “Which parts of the input signal were most responsible for the model saying this appliance is present?”

For each ResNet:

```text
feature maps + class weights
           |
           v
          CAM
```

CamAL then:

1. extracts the CAM for the appliance-present class;
2. normalizes each CAM;
3. averages the CAMs across the ensemble;
4. uses the resulting map as an attention mask;
5. multiplies the mask with the input signal;
6. passes the result through sigmoid;
7. thresholds it to obtain a binary appliance activity signal.

The paper's reported localization procedure therefore combines:

**classification -> explanation -> temporal localization**

Source: CamAL methodology section.

## 3.7 Optional power estimation in the paper

CamAL also describes a simple post-processing route for estimating appliance power:

```text
predicted ON/OFF
      x
appliance mean power
      |
      v
estimated appliance power
      |
clip so it does not exceed aggregate power
```

This is **not a core requirement for our MVP**. Our primary objective is reliable appliance activity detection/localization. Power estimation can be a stretch feature after localization is working.

---

# 4. What the CamAL Paper Reports vs What We Will Build

| Area | CamAL paper | Our project |
|---|---|---|
| Core idea | weakly supervised appliance localization | same research idea, independently implemented |
| Input | aggregate smart-meter time series | REFIT aggregate signal |
| Main model | ensemble of 1D ResNets | our implementation of a ResNet-style time-series classifier |
| Explainability | CAM | CAM implemented and tested |
| Datasets | multiple real-world datasets | REFIT only for controlled scope |
| Appliance scope | multiple appliances/datasets | start with one appliance, expand only if justified |
| Evaluation | paper experiments and baselines | household-safe primary test + robustness evaluation |
| Data quality | research preprocessing | explicit validation and data-quality pipeline |
| Serving | research code/demo | reproducible API/batch inference |
| Model management | research experiment workflow | MLflow/model registry/versioning |
| Monitoring | paper research evaluation | data quality, drift, prediction monitoring |
| Retraining | not our central focus | explicit feedback + retraining workflow |
| Product orientation | research framework | production-oriented student ML system |
| Novelty claim | research contribution by paper authors | no algorithmic novelty claim; engineering extension |

### Core rule

Do not describe our project as “a new CamAL algorithm.”

Describe it as:

> An independently implemented weakly supervised appliance-localization system, inspired by recent research, extended with rigorous evaluation and a production ML lifecycle.

---

# 5. What the Paper Actually Achieved

These numbers belong to the **published CamAL experiments**. They are reference evidence, not our promised results and must never be copied into the resume as our own.

## 5.1 REFIT weakly supervised results reported by the paper

The paper reports the following weakly supervised REFIT results, averaged over 5 runs, for its evaluated appliances:

| REFIT appliance | CamAL F1 reported by paper |
|---|---:|
| Dishwasher | 0.54 |
| Kettle | 0.70 |
| Microwave | 0.16 |
| Washing Machine | 0.14 |

The large variation is important. It shows that appliance type matters and that we should **not assume every appliance will produce strong results**. It is one reason the project starts with one carefully selected appliance and expands only after the first model is validated.

## 5.2 Design ablation reported by the paper

For REFIT, the paper reports an average across the evaluated cases of:

| Metric | Full CamAL | Without Attention-Sigmoid | Without Different Kernel Sizes |
|---|---:|---:|---:|
| Localization F1 | 0.336 | 0.165 | 0.317 |
| Precision | 0.511 | 0.159 | 0.499 |
| Recall | 0.291 | 0.300 | 0.275 |
| MAE | 21.096 | 26.843 | 21.336 |
| MR | 0.162 | 0.114 | 0.156 |

The paper reports that removing the Attention-Sigmoid module substantially hurts F1 and precision, while using the same kernel size across ensemble members causes a smaller degradation. The paper also reports that localization performance peaks around 4–5 ResNets in its REFIT ablation.

These findings directly inform our experiment plan:

```text
Single model
     -> ensemble
     -> diverse kernels
     -> attention/sigmoid
     -> localization evaluation
```

## 5.3 What the paper's results mean for us

The correct interpretation is **not**:

> “We will get F1 = 0.70.”

The correct interpretation is:

> “The paper demonstrates that this approach can work on REFIT, but the result varies heavily by appliance and experimental setup. Our job is to independently measure what our implementation achieves.”

Source: CamAL paper Tables III–IV and Section V-G.

---

# 6. Our Dataset — REFIT

## 5.1 Primary dataset

**REFIT: Electrical Load Measurements (Cleaned)**  
Official source: University of Strathclyde  
Dataset page: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/  
DOI: 10.15129/9ab14b0e-19ac-4279-938f-27f643078cec  
License: CC BY 4.0

The cleaned REFIT dataset contains aggregate and appliance-level electrical consumption measurements for **20 households**, timestamped at **8-second intervals**. The official page lists the cleaned data and its license. The raw dataset is also publicly available.

Source: University of Strathclyde REFIT dataset record.

## 5.2 Why REFIT

REFIT is selected because it gives us:

- public access;
- both aggregate and appliance-level measurements;
- enough structure to train a meaningful time-series model;
- ground truth suitable for localization evaluation;
- direct relevance to NILM/smart-energy use cases;
- a dataset also evaluated by the CamAL paper.

## 5.3 Appliances available/used in the CamAL REFIT setup

The paper's REFIT setup includes:

- Dishwasher;
- Washing Machine;
- Microwave;
- Kettle.

The paper reports the following ON thresholds for its evaluation configuration:

| Appliance | ON threshold used in paper |
|---|---:|
| Dishwasher | 300 W |
| Washing Machine | 300 W |
| Microwave | 200 W |
| Kettle | 500 W |

These are **reference values from the paper**, not automatically our final ground-truth policy. We must verify the raw/cleaned REFIT data and decide the exact evaluation rule before final experiments.

## 5.4 Our initial appliance strategy

### MVP

Start with **one appliance** to keep the project understandable and to establish a reliable baseline.

**Initial candidate: Kettle.**

Reason:

- it is included in the CamAL REFIT experiments;
- it is comparatively easy to interpret;
- its activation events are more visually distinct than some harder appliances;
- an initial successful model gives us a working end-to-end system faster.

However, **Phase 1 data audit is the final authority**. If the actual REFIT quality/coverage makes another appliance more defensible, document the change and use the best-supported candidate.

### Target

Expand to a second appliance only after the first appliance passes the ML quality gate.

### Stretch

Evaluate all four CamAL/REFIT candidate appliances and report per-appliance and aggregate results.

Do not add more appliances merely to make the project look larger.

---

# 7. Data Leakage Policy — Non-Negotiable

The biggest evaluation risk is accidentally allowing the same household to appear in both training and testing.

Bad:

```text
House 1 windows -> train
House 1 other windows -> test
```

This can produce an unrealistically optimistic result because the model has already seen the household's consumption style.

Good:

```text
Train households -> training only
Validation households -> validation only
Test households -> completely unseen until final evaluation
```

## Primary split

Use a **household-level split**. For paper comparability, the reference REFIT setup uses 20 houses with 2 test houses and 2 validation houses, leaving 16 for training.

Our implementation should freeze the selected household IDs in a versioned configuration file.

## Additional robustness evaluation

Because REFIT has only 20 households, a single 2-household test split can have high variance.

After the primary result is working, run a repeated grouped evaluation for robustness without touching the final holdout.

Possible design:

- repeated GroupShuffleSplit or GroupKFold at the household level;
- report mean and standard deviation across held-out household groups;
- preserve appliance-class balance as far as the small dataset allows;
- document any split that cannot meet the balance constraints.

This robustness study is **development evidence**, not permission to peek at the final test set.

---

# 8. Data Pipeline We Will Build

```text
REFIT raw/cleaned files
        |
        v
File/schema validation
        |
        v
Timestamp normalization
        |
        v
Duplicate/missing checks
        |
        v
Sampling alignment / resampling policy
        |
        v
Forward-fill policy with max-gap rule
        |
        v
Physical-value sanity checks
        |
        v
Household-level split
        |
        v
Window generation
        |
        v
Weak labels for training
        |
        v
Strong labels retained only for evaluation
        |
        v
Model-ready tensors
```

## Required data-quality checks

At minimum:

- duplicate timestamps;
- missing timestamps / gaps;
- excessive missing values;
- values below physically meaningful bounds;
- suspicious extreme readings;
- inconsistent sampling intervals;
- household ID mixing;
- appliance column availability;
- empty windows;
- windows with unresolved missing values.

The official cleaned REFIT release states that duplicate timestamps were merged and individual-appliance readings above 4000 W were set to zero because they exceeded the sensor rated limit. We still perform our own validation because production pipelines should not blindly trust input files.

Source: University of Strathclyde cleaned REFIT record.

---

# 9. Windowing Strategy

The CamAL paper uses **510-point non-overlapping subsequences** for its main cross-baseline evaluation. At REFIT's 8-second sampling interval, 510 points represent approximately **68 minutes**.

We will use that as the **paper-reference baseline**, but we will not assume it is automatically optimal for our implementation.

### Window experiments

Start with:

- reference window: 510 points;
- one smaller window candidate;
- one larger window candidate if compute/data volume permits.

The experiment must answer:

> Does a different temporal context improve appliance localization without making training unnecessarily expensive or weakening the weak-label assumption?

Do not search dozens of window sizes without a clear research question.

---

# 10. Model Development Plan

This section is the heart of the ML work.

## Model V0 — Sanity Baseline: Simple 1D CNN

Purpose:

> Prove that the data pipeline and problem formulation can produce learnable signal.

Concept:

```text
Input window
   |
 Conv1D
   |
 ReLU
   |
 Conv1D
   |
 Pooling
   |
 Dense
   |
 Appliance probability
```

Requirements:

- simple enough to debug;
- deterministic seed/config;
- train/validation curves;
- no CAM requirement yet if architecture does not support a clean CAM implementation;
- use the same household-safe split.

### V0 checkpoint

Pass only when:

- training runs reproducibly;
- validation loss changes meaningfully;
- model beats a trivial baseline better than chance/majority behavior;
- no data leakage is detected;
- predictions can be inspected.

If V0 fails, stop and debug the data formulation before proceeding.

---

## Model V1 — Single ResNet Reference

Implement our own 1D ResNet-style architecture based on the published structure.

Concept:

```text
Input
  |
  v
Residual Block 1
  |
Residual Block 2
  |
Residual Block 3
  |
Global Average Pooling
  |
Linear layer
  |
Appliance-present probability
```

Use the paper's architecture as a reference point:

- 3 residual blocks;
- 3 convolutional layers per residual block;
- filter groups `{64, 128, 128}`;
- ConvNet kernel pattern `{8, 5, 3}` inside each residual block;
- global average pooling;
- linear classifier.

The implementation must be written and understood by us rather than copying the research repository wholesale.

### V1 checkpoint

Pass when:

- model trains reproducibly;
- classification metrics are produced;
- training/validation curves are plausible;
- model parameters and input shapes are documented;
- at least one CAM can be extracted from the architecture;
- no test-set information influenced training decisions.

---

## Model V2 — CamAL-Style Ensemble

Once V1 is stable, build the ensemble logic.

### MVP ensemble

Use a small number of networks first, for example **3 models with distinct kernel choices**, to verify the ensemble/CAM workflow.

### Target ensemble

Move toward the paper's default design:

```text
ResNet A -> k=5
ResNet B -> k=7
ResNet C -> k=9
ResNet D -> k=15
ResNet E -> k=25
```

Average their detection probabilities.

Then:

```text
CAM A
CAM B
CAM C
CAM D
CAM E
  |
  v
Average CAM
  |
Attention + sigmoid
  |
Binary activity timeline
```

### Compute rule

Do not run the paper's full experimental grid simply because it exists.

Start with the smallest experiment that validates the core mechanism. Expand to the five-model ensemble when runtime is acceptable on normal CPU/free GPU resources.

The paper itself reports 5 networks as the default and finds localization strongest around 4–5 networks on REFIT in its ablation study.

Source: CamAL paper ablation results.

---

# 11. Evaluation Plan

We need **two different types of numbers**.

## 10.1 Detection/classification numbers

Measure:

- Precision;
- Recall;
- F1;
- Balanced Accuracy.

Balanced Accuracy is important because appliance activity classes can be imbalanced.

## 10.2 Localization numbers

The core project result is not merely “the appliance exists.”

It is:

> **Did we identify the correct timestamps where it was active?**

Measure at minimum:

- timestamp-level Precision;
- timestamp-level Recall;
- timestamp-level F1.

Where appropriate, also report the same error measures used by the paper for appliance power estimation, but only if power estimation is actually part of the implementation.

## 10.3 Operational/model numbers

Separately measure:

- inference latency;
- throughput;
- model size;
- training runtime;
- resource usage;
- number of invalid records detected;
- percentage of rejected windows;
- drift alert rate;
- retraining improvement when applicable.

Do **not** mix model-quality metrics and infrastructure metrics into one fake “overall score.”

---

# 12. The Main Experimental Questions

Every experiment should answer a question.

## RQ1 — Can a simple model learn the signal?

Compare:

```text
Trivial baseline
vs
Simple 1D CNN
```

## RQ2 — Does the ResNet improve representation quality?

Compare:

```text
Simple CNN
vs
Single ResNet
```

## RQ3 — Does the ensemble improve localization?

Compare:

```text
Single ResNet
vs
3-model ensemble
vs
5-model ensemble
```

## RQ4 — Do different receptive fields matter?

Compare:

```text
same kernel
vs
different kernels
```

## RQ5 — Does the localization mechanism actually help?

Compare:

```text
raw/naive CAM aggregation
vs
CAM + attention/sigmoid processing
```

This is directly motivated by the paper's ablation study.

## RQ6 — Does temporal context matter?

Compare a small set of meaningful window sizes.

## RQ7 — How much does threshold calibration matter?

Compare the reference threshold with a validation-selected threshold.

Do not tune the production threshold on the final test set.

---

# 13. Our Differentiation — What We Add

Our value should come from **engineering depth + rigorous evaluation + measurable improvements**, not fake algorithmic novelty.

## Addition 1 — Stronger leakage controls

Use household-level train/validation/test isolation and freeze the split configuration.

## Addition 2 — Explicit data-quality layer

Validate timestamps, gaps, duplicates, ranges, sampling, nulls, and windows before training/inference.

## Addition 3 — Calibration/confidence layer

Instead of treating every prediction as equally trustworthy:

```text
high confidence -> normal prediction
medium confidence -> flagged
low confidence -> review/uncertain
```

The exact policy will be learned from validation data and documented.

## Addition 4 — Production inference

Expose the trained model through a reproducible API and/or batch job.

## Addition 5 — Experiment tracking + model registry

Track:

- dataset version;
- configuration;
- seed;
- model version;
- metrics;
- artifacts;
- training run;
- promotion status.

## Addition 6 — Monitoring

Monitor:

- input quality;
- input distribution/drift;
- prediction distribution;
- confidence distribution;
- service latency/errors;
- delayed ground-truth performance when labels become available.

## Addition 7 — Retraining loop

```text
new data
   |
quality validation
   |
model inference
   |
monitoring
   |
drift/performance signal
   |
label review / ground truth when available
   |
retrain candidate
   |
compare candidate vs current model
   |
deploy only if acceptance criteria pass
```

This is the main production/MLOps extension.

---

# 14. MLOps Architecture

```text
                   DATA SOURCE
                       |
                       v
              Ingestion / Storage
                       |
                       v
               Data Validation
                       |
                       v
              Preprocessing
                       |
                       v
              Window Generation
                       |
                       v
                 Training
                       |
             +---------+---------+
             |                   |
             v                   v
        Experiment Log      Model Artifact
             |                   |
             +---------+---------+
                       v
                  Model Registry
                       |
             +---------+---------+
             |                   |
             v                   v
        Batch Inference       API Serving
             |                   |
             +---------+---------+
                       v
                   Monitoring
             +---------+---------+
             |                   |
             v                   v
         Drift/Quality       Performance
             |                   |
             +---------+---------+
                       v
                 Review / Retrain
                       |
                       v
                  New Model
```

### Important sequencing rule

**Do not build the full MLOps stack before the ML model passes its quality gate.**

A weak model wrapped in excellent Docker/MLflow code is still a weak ML project.

---

# 15. Serving Design

The serving interface should be simple.

## Example API concept

```text
POST /predict

Input:
- household identifier or request ID
- aggregate power window
- model version (optional)

Output:
- appliance
- detection probability
- detected/not detected
- localized activity intervals
- confidence
- model version
```

A batch inference interface may be built first because it is simpler and closer to the dataset workflow.

Then expose the same inference logic through FastAPI so the API is a thin serving layer, not a second implementation of the model.

---

# 16. Monitoring Design

Monitoring must answer real questions.

## Data quality monitoring

Examples:

```text
missing timestamps ↑
duplicate timestamps ↑
invalid power values ↑
window rejection rate ↑
```

## Data drift monitoring

Track changes in features such as:

- mean aggregate power;
- variance;
- quantiles;
- peak power;
- proportion of near-zero readings;
- window length/completeness.

## Prediction drift

Track:

- detection probability distribution;
- positive prediction rate;
- confidence distribution;
- estimated appliance-active duration.

## Performance monitoring

When delayed labels become available:

- F1;
- precision;
- recall;
- localization error.

### Critical principle

Drift is **not automatically model failure**.

A drift alert means:

> “The new data looks different from what the model was trained on.”

We retrain only when the agreed evidence threshold is met.

---

# 17. Retraining Policy

Retraining should not be a blind “run train.py” button.

Proposed candidate-model process:

```text
new training data
      |
validation
      |
train candidate
      |
compare against current model
      |
check core metrics
      |
check regression/latency constraints
      |
approve candidate
      |
register new model
      |
deploy
```

Candidate promotion requires measurable evidence.

Example acceptance rule:

- candidate must not materially degrade localization F1;
- candidate must not exceed latency/resource guardrails;
- data-quality tests must pass;
- inference tests must pass;
- model artifact must be versioned.

Exact numeric thresholds will be established after the baseline is measured.

---

# 18. Experiment Tracking

Every meaningful training run must record:

```text
run_id
commit/version
household split version
dataset version
appliance
window size
model architecture
kernel sizes
seed
learning rate
batch size
epochs
training loss
validation loss
precision
recall
F1
balanced accuracy
localization metrics
training time
inference time
artifact location
```

The final report must be reproducible from these records.

---

# 19. Project Phases and Hard Checkpoints

## Phase 0 — Project skeleton

### Work

- create repository structure;
- create environment;
- create configuration system;
- define experiment logging format;
- define code-quality/test policy.

### Output

Runnable project skeleton.

### Checkpoint P0

Pass only when a clean environment can run a minimal end-to-end smoke test.

---

## Phase 1 — REFIT acquisition and forensic data audit

### Work

- obtain cleaned REFIT from official source;
- inspect schema;
- inspect households;
- inspect appliances;
- measure missingness;
- inspect timestamp regularity;
- inspect value ranges;
- verify data volume;
- build reproducible local data snapshot/index;
- document provenance/license.

### Output

`DATASET_REPORT.md` + validated data-loading pipeline.

### Checkpoint P1

Must answer:

- What columns exist?
- What does one row represent?
- Which household IDs exist?
- Which appliances are usable?
- How many positive/negative households exist per appliance?
- What is the actual sampling behavior?
- How many windows survive cleaning?
- Is kettle still the best first appliance?

**Do not begin model training until these questions are answered.**

---

## Phase 2 — Leakage-safe preprocessing and window generation

### Work

- freeze household split;
- define missing-gap policy;
- scale/normalize only using training-derived statistics where applicable;
- generate fixed windows;
- create weak training labels;
- preserve strong labels separately for evaluation.

### Output

Versioned model-ready dataset/index.

### Checkpoint P2

Pass only when:

- no household crosses train/validation/test;
- weak labels are correct;
- strong labels are isolated from training;
- sample counts are auditable;
- a few windows can be visualized manually.

---

## Phase 3 — Simple 1D CNN baseline

### Work

- implement V0;
- train on training households;
- tune on validation only;
- evaluate on validation;
- produce learning curves and predictions.

### Output

Baseline metrics + model artifact.

### Checkpoint P3

The model must clearly beat a trivial baseline and behave sensibly enough to continue.

Otherwise stop and fix the data/model formulation.

---

## Phase 4 — Single ResNet

### Work

- implement V1;
- validate tensor shapes;
- verify residual connections;
- train;
- compare against V0;
- implement CAM extraction.

### Output

ResNet model + classification metrics + first CAM visualizations.

### Checkpoint P4

Must have:

- stable training;
- reproducible metrics;
- working CAM extraction;
- human-readable signal/CAM plots;
- no test-set tuning.

---

## Phase 5 — CamAL-style ensemble and localization

### Work

- start with 3 models;
- combine detection probabilities;
- extract/normalize/average CAMs;
- implement attention/sigmoid step;
- generate binary activity timeline;
- compare against strong evaluation labels.

Then, if viable, expand toward the 5-model target ensemble.

### Output

Localization system + per-window/per-timestamp metrics.

### Checkpoint P5 — CORE ML GATE

Proceed only if the system demonstrates **credible, non-trivial localization performance** on unseen households.

The exact minimum score must not be invented before the baseline is known. Establish an evidence-based target after the first controlled run.

If results are poor:

1. inspect data;
2. inspect labels;
3. inspect windowing;
4. inspect thresholding;
5. inspect CAM behavior;
6. compare simple CNN vs ResNet;
7. only then consider architecture changes.

Do not hide a poor result by changing the test set.

---

## Phase 6 — Controlled experiments / ablations

Run only experiments tied to defined questions:

- CNN vs ResNet;
- single vs ensemble;
- fixed vs diverse kernels;
- CAM without attention vs CAM + attention/sigmoid;
- selected window sizes;
- reference threshold vs calibrated threshold.

### Output

Experiment matrix + plots + conclusions.

### Checkpoint P6

Every experiment must have:

```text
question -> setup -> result -> interpretation
```

No random hyperparameter hunting.

---

## Phase 7 — Confidence/calibration + audit layer

### Work

- inspect probability calibration;
- select threshold using validation data;
- define uncertain cases;
- generate human-auditable prediction outputs;
- measure whether confidence correlates with correctness.

### Output

Confidence policy + calibration report.

### Checkpoint P7

Threshold is frozen before final test evaluation.

---

## Phase 8 — Batch inference + API

### Work

- package preprocessing + model + CAM logic;
- build batch inference path;
- expose FastAPI endpoint;
- add request validation;
- return model version with prediction;
- add tests.

### Output

Reproducible local inference service.

### Checkpoint P8

A fresh environment can send a valid request and receive a correct prediction from the registered model artifact.

---

## Phase 9 — MLflow + model registry

### Work

- log experiments;
- save metrics/artifacts;
- version models;
- define candidate/staging/production lifecycle;
- attach dataset/config metadata.

### Output

Reproducible experiment and model registry workflow.

### Checkpoint P9

A reviewer can identify exactly which data/config/model generated the reported metric.

---

## Phase 10 — Monitoring

### Work

- data-quality monitoring;
- feature drift;
- prediction drift;
- service latency/errors;
- delayed performance where labels exist.

### Output

Monitoring dashboard/report + alert rules.

### Checkpoint P10

At least one controlled drift scenario triggers a meaningful alert without changing the underlying model.

---

## Phase 11 — Retraining workflow

### Work

- simulate incoming data;
- detect drift/performance degradation;
- produce new training set;
- retrain candidate;
- compare candidate vs current;
- promote only when acceptance criteria pass.

### Output

End-to-end retraining demonstration.

### Checkpoint P11

We can show:

```text
old model -> drift/performance issue -> retrain -> candidate evaluation -> new version
```

with auditable evidence.

---

## Phase 12 — Final evaluation and portfolio packaging

### Work

- unfreeze final test set;
- run once using final configuration;
- record final metrics;
- record inference/runtime measurements;
- document limitations;
- prepare architecture diagram;
- prepare README/demo;
- write resume bullets only from measured facts.

### Output

Final portfolio-ready repository.

### Checkpoint P12

All numbers in the README/resume can be traced to stored experiment artifacts.

---

# 20. MVP vs Target vs Stretch

## MVP — Must be completed

```text
REFIT cleaned data
+ data validation
+ household-safe split
+ simple 1D CNN
+ 1D ResNet
+ CAM localization
+ core metrics
+ unseen-household evaluation
+ reproducible training
```

## Target — Main portfolio version

```text
MVP
+ multi-model ensemble
+ confidence/calibration
+ batch inference
+ FastAPI
+ MLflow
+ model registry
+ Docker
+ monitoring
+ drift detection
+ retraining workflow
```

## Stretch — Only if the target is stable

```text
+ second/third appliance
+ stronger robustness study
+ richer dashboard
+ power estimation
+ CI/CD deployment
+ cloud deployment
+ automated model promotion
```

Do not sacrifice the core ML result to finish stretch infrastructure.

---

# 21. Acceptance Criteria for the Final Project

The project is considered complete only when all of the following are true:

### ML

- model trains reproducibly;
- model predicts appliance presence/activity;
- CAM localization works;
- unseen-household evaluation is used;
- core metrics are reported;
- baseline comparisons are included;
- at least one meaningful ablation is completed.

### Data

- dataset provenance is documented;
- data-quality validation exists;
- leakage checks exist;
- weak vs strong labels are clearly separated.

### Engineering

- modular preprocessing;
- configuration-driven experiments;
- model artifact/versioning;
- reproducible inference;
- tests for critical components.

### MLOps

- experiment tracking;
- model registry;
- serving;
- monitoring;
- drift detection;
- retraining demonstration.

### Communication

- beginner-readable README;
- architecture diagram;
- research-reference section;
- explicit limitations;
- exact reproducibility instructions;
- resume numbers traceable to artifacts.

---

# 22. What Counts as a Strong Result

We will **not** choose a target such as “F1 must be 0.90” before running the baseline.

That would encourage gaming the experiment.

Instead, strong evidence comes from a combination of:

1. meaningful performance above trivial baselines;
2. useful localization quality on unseen households;
3. improvement from justified design choices;
4. reproducibility;
5. robustness across held-out household groups;
6. useful engineering performance;
7. a working production lifecycle.

A strong resume bullet must use **observed values**, for example:

> Built a weakly supervised time-series deep learning system for appliance localization on REFIT, achieving **[measured F1]** on unseen households while requiring only household-level appliance-presence labels for model training.

The placeholder must remain a placeholder until the final experiment produces the number.

---

# 23. Academic/Attribution Rules

## We can do

- independently implement the published method;
- study the official research repository;
- use the paper's design as a technical reference;
- compare our implementation to published results where evaluation conditions are genuinely comparable;
- extend the system with our own engineering components.

## We cannot honestly claim

- that we invented weakly supervised appliance localization;
- that we invented the CamAL architecture;
- that our system is scientifically novel when it is not;
- that we reproduced the paper exactly if we changed data/splits/architecture/compute;
- paper performance as our own experiment result.

## Public project presentation

Use the product title:

**SmartMeter Appliance Intelligence & Localization Platform**

The README should include a concise **Research Foundation / References** section that cites the CamAL paper and official repository. Attribution is required; the product name does not need to contain the research paper's name.

---

# 24. Antigravity Operating Instructions

Antigravity should treat this document as the **source of truth**.

## Rule 1 — Follow the phase order

Do not jump to MLOps before the core model passes the ML gate.

## Rule 2 — Stop at checkpoints

At every checkpoint, report:

```text
STATUS: PASS / FAIL / BLOCKED
WHAT WAS DONE
KEY RESULTS
EVIDENCE / ARTIFACTS
PROBLEMS FOUND
RECOMMENDED NEXT STEP
```

Do not silently continue after a failed gate.

## Rule 3 — Never fabricate metrics

Never invent:

- F1;
- precision;
- recall;
- latency;
- drift percentage;
- retraining improvement;
- label savings;
- training-time reduction.

All final numbers must come from actual experiment artifacts.

## Rule 4 — No leakage

Never split windows first and households later.

Always split by household before creating model windows.

## Rule 5 — Protect the final test set

The final test households cannot be used for:

- architecture selection;
- threshold tuning;
- feature selection;
- model selection;
- early stopping decisions;
- repeated experimentation.

## Rule 6 — Do not copy the reference repository

The official CamAL repository is a **reference implementation** and reproducibility resource.

Do not copy large sections of the repository into our project. Implement the needed components in our own structure, explain them, and cite the source of the underlying method.

## Rule 7 — Prefer the simplest defensible solution

Do not introduce a technology unless it solves a real project requirement.

## Rule 8 — Keep the student able to defend everything

Every component must be explainable at interview level:

```text
what it does
why it is there
what input it receives
what output it produces
what can fail
how we tested it
```

## Rule 9 — Preserve reproducibility

Every meaningful result must be tied to:

```text
data version + split + config + code version + seed + model artifact
```

## Rule 10 — Do not confuse infrastructure with ML quality

Docker, MLflow, API, CI/CD and monitoring are supporting layers.

The core evidence is still:

> **Can the model correctly localize appliance activity on unseen households using weak supervision?**

---

# 25. Expected Repository Structure

This is a recommended starting structure. Antigravity may adjust it only when there is a clear reason.

```text
smartmeter-appliance-intelligence/
|
|-- README.md
|-- PROJECT_SPEC.md
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
|-- .gitignore
|
|-- configs/
|   |-- data.yaml
|   |-- model.yaml
|   |-- train.yaml
|   |-- serving.yaml
|   `-- monitoring.yaml
|
|-- data/
|   |-- README.md
|   |-- raw/              # not committed
|   |-- processed/        # not committed
|   `-- manifests/
|
|-- notebooks/
|   |-- 01_data_audit.ipynb
|   |-- 02_baseline_experiments.ipynb
|   `-- 03_error_analysis.ipynb
|
|-- src/
|   |-- data/
|   |-- validation/
|   |-- features/
|   |-- models/
|   |-- localization/
|   |-- evaluation/
|   |-- serving/
|   |-- monitoring/
|   `-- retraining/
|
|-- scripts/
|   |-- prepare_data.py
|   |-- train.py
|   |-- evaluate.py
|   |-- infer.py
|   |-- monitor.py
|   `-- retrain.py
|
|-- tests/
|
|-- reports/
|   |-- dataset_report.md
|   |-- experiments/
|   |-- figures/
|   `-- final_metrics.md
|
|-- deployment/
|   |-- Dockerfile
|   `-- compose.yaml
|
`-- .github/
    `-- workflows/
```

Do not create this entire structure on day one if it causes empty folders and unnecessary complexity. Start small and expand phase by phase.

---

# 26. Deliverables by the End

The final portfolio should contain:

### Research understanding

- paper summary;
- method explanation;
- comparison to alternatives;
- explicit statement of what was adapted.

### Data

- dataset provenance;
- data audit;
- preprocessing rules;
- household split definition;
- weak-label construction.

### ML

- baseline CNN;
- ResNet;
- ensemble;
- CAM localization;
- evaluation;
- ablations;
- error analysis.

### Engineering

- reproducible pipeline;
- batch inference;
- API;
- tests;
- model registry.

### MLOps

- monitoring;
- drift detection;
- retraining workflow;
- model promotion policy.

### Portfolio

- README;
- architecture diagram;
- selected visualizations;
- demo;
- final measured metrics;
- limitations;
- resume-ready bullets.

---

# 27. Interview Understanding Checklist

Before calling the project finished, the student should be able to explain all of these without reading the paper:

### Problem

- What is NILM?
- Why is aggregate electricity difficult?
- Detection vs localization?

### Weak supervision

- What is a strong label?
- What is a weak label?
- Why is possession information cheaper?
- Why does copying one household label to every timestamp create a problem?

### Model

- Why 1D CNN?
- What does a convolution learn here?
- What is a residual connection?
- Why use multiple kernel sizes?
- Why use an ensemble?

### CAM

- What is a CAM?
- How does it tell us which time regions influenced a prediction?
- Why can classification become localization?

### Evaluation

- Why split by household?
- Why is random-window splitting dangerous?
- Why use F1/Balanced Accuracy?
- How do we evaluate timestamp localization?

### MLOps

- Why register the model?
- What is drift?
- Why does drift not automatically mean retraining?
- What happens before a new model is deployed?

### Project ownership

- Which part came from published research?
- What did we independently implement?
- What did we add?
- What results are ours?
- What limitations remain?

---

# 28. Final Mental Model

The entire project should be remembered like this:

```text
REAL PROBLEM
Infer appliance activity from total household electricity
                    |
                    v
RESEARCH IDEA
Weak labels + time-series ResNet + CAM
                    |
                    v
OUR ML IMPLEMENTATION
CNN -> ResNet -> ensemble -> CAM localization
                    |
                    v
OUR RIGOR
Household-safe evaluation + data-quality controls
                    |
                    v
OUR ENGINEERING
Inference API + versioning + testing
                    |
                    v
OUR MLOPS
Monitoring + drift + retraining
                    |
                    v
FINAL EVIDENCE
Measured ML performance + production metrics + working demo
```

### One-sentence project definition

> **Build a production-oriented smart-meter appliance intelligence system that learns from weak household-level appliance information and uses time-series deep learning plus CAM-based localization to estimate when appliance activity occurred, then monitors and retrains the model as data changes.**

---

# 29. Primary References

1. Petralia, A., Boniol, P., Charpentier, P., Palpanas, T. (2025). *Few Labels are All You Need: A Weakly Supervised Framework for Appliance Localization in Smart-Meter Series.* IEEE ICDE 2025. DOI: 10.1109/ICDE65448.2025.00329.  
   https://arxiv.org/abs/2506.05895

2. Official CamAL implementation.  
   https://github.com/adrienpetralia/CamAL

3. Murray, D., Stankovic, L., Stankovic, V. *REFIT: Electrical Load Measurements (Cleaned).* University of Strathclyde. DOI: 10.15129/9ab14b0e-19ac-4279-938f-27f643078cec.  
   https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/

4. REFIT raw dataset record. University of Strathclyde. DOI: 10.15129/31da3ece-f902-4e95-a093-e0a9536983c4.  
   https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/

---

# 30. Current Decision Record

**Project selected:** Yes  
**Project title:** SmartMeter Appliance Intelligence & Localization Platform  
**Core research foundation:** CamAL (ICDE 2025)  
**Dataset:** REFIT cleaned  
**Initial appliance:** Kettle candidate; confirm after Phase 1 audit  
**Core ML:** 1D CNN -> 1D ResNet -> CamAL-style ensemble -> CAM localization  
**Primary evaluation:** unseen households  
**Primary ML deliverable:** appliance localization, not just classification  
**Production layer:** API/batch + experiment tracking + registry + monitoring + drift + retraining  
**Scientific novelty claim:** None  
**Engineering differentiation:** Yes  
**Metric policy:** no fabricated numbers; final resume metrics come only from stored experiments

---

# 31. First Action After This Spec Is Approved

**Do Phase 0 + Phase 1 only.**

The first concrete work should be:

```text
1. Create repository skeleton
2. Obtain official cleaned REFIT
3. Verify provenance/license
4. Audit schema and households
5. Audit appliances
6. Measure missingness and timestamp quality
7. Determine candidate appliance support
8. Freeze the initial household split
9. Produce DATASET_REPORT.md
10. Stop and report checkpoint P1
```

**Do not implement the CNN until P1 is passed.**

That sequencing is intentional: the dataset and split determine whether the entire ML story is defensible.
