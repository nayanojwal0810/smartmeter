# Python Design Standard

## Purpose

This document defines how Python code in this project should be structured, written, tested, and maintained.

The goal is clean, modular engineering that remains easy to understand and defend.

## Project Structure

Use logical packages under `src/`.

Recommended areas:

```text
src/
├── data/
├── validation/
├── models/
├── localization/
├── evaluation/
├── serving/
├── monitoring/
└── retraining/
```

Keep entry points in `scripts/`.

Keep tests in `tests/`.

Do not put experiment-specific implementation into one giant training script.

## Naming

Use `snake_case` for:

- modules;
- functions;
- variables;
- configuration keys.

Use `PascalCase` for classes.

Use descriptive names.

## Functions

Prefer small functions with one responsibility.

Good:

```text
load_household()
validate_schema()
generate_windows()
compute_metrics()
save_predictions()
```

Avoid functions that perform an entire workflow plus plotting plus logging plus artifact management.

## Classes

Use classes when state or lifecycle improves clarity.

Good candidates include:

- dataset loaders with configuration/state;
- model classes;
- trainers;
- inference services;
- registry/serving adapters.

Do not create classes only to satisfy an OOP rule.

## Imports

Keep imports:

- explicit;
- minimal;
- organized;
- free of avoidable circular dependencies.

Avoid wildcard imports.

## Configuration

Do not hard-code:

- dataset paths;
- household IDs;
- thresholds;
- hyperparameters;
- experiment names;
- output directories.

Load them from project configuration.

Code should accept explicit configuration rather than silently depending on local machine state.

## Determinism

Every experiment that claims reproducibility must define:

- seed;
- data split;
- preprocessing configuration;
- model configuration;
- training configuration;
- environment/dependency versions where practical.

Set random seeds deliberately and document components that remain nondeterministic.

## Data Boundaries

Keep these responsibilities separate:

```text
raw data loading
→ validation
→ preprocessing
→ window generation
→ model input
→ inference
→ evaluation
```

Do not let model code quietly load raw files.

Do not let evaluation code modify training data.

Do not let the API contain a second copy of preprocessing logic.

## Weak Supervision Boundary

The main weakly supervised training code may use household-level appliance-presence labels.

Timestamp-level appliance measurements must not silently enter:

- training targets;
- training loss;
- feature generation;
- normalization fitting;
- threshold tuning;
- model selection.

Evaluation code may access timestamp-level ground truth through an explicit evaluation interface.

The separation should be obvious from function names and data structures.

## Model Interfaces

Keep model interfaces predictable.

A model should have clear boundaries for:

```text
input
forward/prediction
probability output
feature/CAM extraction where supported
serialization
```

Do not duplicate model logic across training and inference.

## Training Code

Training code should separate:

```text
configuration
data loading
training loop
validation
checkpointing
metrics
artifact saving
```

Do not mix training with ad-hoc exploratory analysis.

The final training entry point should be runnable from a clean environment using configuration.

## Evaluation Code

Evaluation must use the same prediction interface as inference.

Keep metric calculations deterministic and independently testable.

Record:

- metric definitions;
- evaluation population;
- configuration/version;
- artifact location.

Do not write evaluation code that can accidentally consume the final test set during model development.

## Logging

Use professional structured messages.

Examples:

```text
[INFO] Loaded 20 households.
[INFO] Generated 12,480 valid windows.
[INFO] Training completed. Validation F1: 0.612.
[WARNING] 3.4% of windows rejected during validation.
[ERROR] Missing required appliance column: Kettle.
```

Avoid casual debug prints.

Use logging rather than scattered `print()` calls when the application grows.

## Exceptions

Fail clearly when required inputs are invalid.

Do not silently replace missing critical data with guessed values.

Exception messages should identify:

```text
what failed
why it failed
what input was involved
```

## Type Hints

Use type hints for public functions, model interfaces, configuration objects, and non-trivial internal functions.

Keep types readable rather than excessively complex.

## Docstrings

Public modules, classes, and important functions should have concise docstrings covering purpose and important inputs/outputs.

Avoid redundant docstrings that simply restate the function name.

## Testing

Test important deterministic logic, especially:

- schema validation;
- timestamp handling;
- window generation;
- household split logic;
- label construction;
- preprocessing;
- metric calculations;
- model input/output shapes;
- CAM extraction;
- inference request validation.

Tests should be fast enough for regular local validation unless explicitly designated as expensive integration tests.

## Artifacts

Code must save experiment outputs to deterministic, versioned locations.

Do not overwrite prior evidence.

Prefer:

```text
artifacts/
models/
runs/
reports/
```

with explicit identifiers.

## Dependency Discipline

Add a dependency only when it solves a demonstrated requirement.

Prefer a small, stable dependency set.

Do not introduce libraries simply because they are popular.

## Reuse

Build reusable components when the same logic is needed in training, evaluation, batch inference, and serving.

Do not copy/paste preprocessing or localization logic across scripts.

## CLI Entry Points

Scripts should expose clear commands and arguments.

Avoid requiring users to edit Python source just to change a path or experiment configuration.

## Code Review Standard

Before considering Python work complete, check:

```text
Does this file have one clear purpose?
Can another module reuse the important logic?
Are paths/configs hard-coded?
Can tests cover the important behavior?
Could this leak evaluation information?
Is the output reproducible?
Could a beginner explain it?
```
