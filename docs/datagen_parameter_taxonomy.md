# Datagen Parameter Taxonomy

This document is now scoped to offline data generation.

Datagen parameters still matter for building the cold-start dataset, but they are no longer the main online action space for `AgentExplorer`.

## Current Boundary

```text
offline:
  DatagenFEMEvaluator uses datagen parameters to generate large datasets.

online:
  AgentExplorer proposes target schedules.
  InverseDesigner generates structures for scheduled targets.
```

## Offline Datagen Variables

Datagen may still expose variables such as:

```text
group / symmetry
unit cell family
topology family
connectivity pattern
max_bars
rho_target
density range
sampling strategy
validity constraints
```

These variables should be used to shape dataset coverage:

```text
cover broad structure families
cover target-property regions
include valid, near-boundary, and failure cases
avoid excessive duplicates
preserve provenance for later analysis
```

## Online Replacement

The online closed loop should use target-space variables:

```text
final_target
scheduled_target
target perturbation
target curriculum
target-region probe
samples_per_target
schedule strategy
```

This replaces the old action:

```text
AgentExplorer -> DatagenConfig
```

with:

```text
AgentExplorer -> TargetSchedule
```

## Migration Note

Legacy code may still contain `DatagenConfig`, `MetaCandidate`, and datagen-backed scheduler paths.

During migration:

```text
keep datagen contracts for offline bootstrap
introduce TargetSchedule contracts for online closed loop
move online scheduler away from datagen()
retain explicit-structure FEM evaluation
```

## Principle

```text
Datagen builds the initial world.
Target schedules explore within it.
```
