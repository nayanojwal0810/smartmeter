# Documentation Design Standard

## Purpose

This document defines how project-facing Markdown files, reports, experiment summaries, and README content should be written.

The documentation should feel like a professional ML engineering repository.

## General Style

Use:

- clear headings;
- short paragraphs;
- compact tables where useful;
- concise technical language;
- simple English;
- only information that helps the reader understand the system.

Avoid:

- unnecessary repetition;
- long motivational sections;
- internal chat language;
- exaggerated claims;
- cluttered formatting;
- excessive emojis;
- generated-sounding filler.

## GitHub-Facing Language

Do not write internal workflow labels such as:

```text
Phase 1
Step 1
Checkpoint 1
STATUS: PASS
```

unless a file is explicitly an internal control document.

Prefer professional names:

```text
Data Acquisition
Data Quality Audit
Baseline Model
ResNet Evaluation
Localization Evaluation
Calibration
Serving
Monitoring
```

## README

The final README should normally contain:

```text
Project Overview
Problem
Why the Approach
System Architecture
Data
Method
Evaluation
Results
Engineering
Reproducibility
Limitations
Research Foundation
```

Do not fill Results with placeholders after implementation begins. Use measured values only.

## Dataset Report

A dataset report should answer:

- source and license;
- dataset version/snapshot;
- schema;
- row meaning;
- household coverage;
- appliance availability;
- timestamp behavior;
- missingness;
- invalid values;
- sampling behavior;
- cleaning policy;
- usable candidate appliance;
- final household split;
- data limitations.

Keep raw data unchanged.

## Experiment Report

Every meaningful experiment should present:

```text
Question
Setup
Configuration
Metric
Result
Interpretation
Decision
Artifact
```

The reader should be able to understand why the experiment existed and what it changed.

## Results Writing

Separate:

### External evidence

What the paper or dataset documentation reports.

### Project results

What our own implementation measured.

### Interpretation

What the measured result may indicate.

### Assumptions

What has not been established experimentally.

Never turn an external result into a project result.

## Metrics

Always identify:

- metric name;
- population;
- target definition;
- aggregation method.

Do not present one number without enough context to interpret it.

Keep ML quality metrics separate from operational metrics.

## Tables

Use tables when they make comparison easier.

Avoid giant tables containing every training parameter unless a reproducibility appendix needs them.

## Figures

Every important figure should have:

- descriptive title/caption;
- readable axis labels;
- units where applicable;
- enough context to interpret the result.

Do not create decorative figures that do not support a technical point.

## Claims

A documented claim should map to evidence.

Recommended structure:

```text
Claim
→ source/result
→ interpretation
```

For example:

```text
The ensemble improved validation localization F1 by 0.07.
Source: experiment run ...
Interpretation: the result supports the ensemble hypothesis under this split.
```

Do not say:

```text
The ensemble is better.
```

without identifying the evidence and evaluation setting.

## Limitations

Limitations should be specific and honest.

Examples:

- small number of households;
- appliance-specific performance variation;
- weak-label noise;
- computational limits;
- limited robustness coverage;
- delayed ground truth;
- assumptions in localization post-processing.

Do not hide material limitations.

## Research Attribution

The CamAL paper and official implementation must be cited whenever their method is discussed.

Documentation should clearly state:

```text
what came from the published research
what was independently implemented
what was adapted
what was added as engineering work
what results are ours
```

## Resume Claims

Resume bullets must use measured values only.

Good structure:

```text
Built [system] using [method], achieving [measured result] on [evaluation setting], with [measured engineering outcome].
```

Do not claim:

```text
SOTA
production-ready
real-time
hallucination-free
```

without direct supporting evidence.

## Documentation Update Rule

Update documentation when a technical decision becomes established.

Do not continuously rewrite documents for speculative ideas.

Do not silently change historical experiment results.

Historical evidence should remain traceable.

## Final Reader Test

A new engineer should be able to understand:

```text
What is the problem?
What data is used?
What does the model do?
How is it trained?
How is localization evaluated?
How do we know the result is credible?
How can the project be reproduced?
What are the limitations?
```
