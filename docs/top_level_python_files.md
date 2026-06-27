# Top-Level Python Files

This document summarizes the top-level API files and how they relate to the target-schedule closed loop.

## Main Entry Points

```text
src/api.py
  function-style public API

src/agent_api.py
  class-style wrapper for agent callers

src/cli.py
  command-line entry

src/__init__.py
  lazy package exports

src/closed_loop_contracts.py
  shared contracts for scheduler, knowledge, feedback, and observations

src/datagen_contracts.py
  contracts for offline datagen pipeline
```

## Current Architecture

```text
Offline cold start:
  DatagenFEMEvaluator
    -> large structure-property dataset
    -> InverseDesigner pretraining

Online closed loop:
  AgentExplorer
    -> TargetSchedule
  InverseDesigner
    -> explicit structures
  FEMEvaluator
    -> evaluated properties
  RawExperimentStore
    -> raw observations
  KnowledgeBase
    -> target-schedule evidence
  FeedbackSignal
    -> stop / continue / next anchor
```

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `DatagenFEMEvaluator/` | Offline data generation, conversion, optional proxy/FEM evaluation, bootstrap datasets |
| `InverseDesigner/` | Learn and sample `target -> explicit structure` |
| `AgentExplorer/` | Propose target schedules from knowledge and feedback |
| `Scheduler/` | Orchestrate the online loop |
| `ExperimentStore/` | Store append-only raw observations |
| `KnowledgeBase/` | Store and query target-schedule evidence |
| `KnowledgeRefiner/` | Summarize evidence into agent-readable knowledge |
| `TrainingDataset/` | Export or curate data for InverseDesigner |

## Important API Functions

| API | Purpose |
| --- | --- |
| `run_closed_loop_discovery(...)` | Run the closed-loop discovery task |
| `bootstrap_seed_dataset(...)` | Build offline seed data and optionally seed the KB |
| `refine_agent_knowledge(...)` | Build a knowledge summary from KB evidence |
| `kb_query_samples(...)` | Query success, near miss, failure, or similar samples |
| `export_inverse_designer_dataset(...)` | Export training data for InverseDesigner |

## Contract Direction

Legacy closed-loop contracts include:

```text
DatagenConfig
MetaCandidate
BatchProposal
```

These represent the old online action space: `AgentExplorer -> datagen meta`.

The target architecture should introduce or migrate toward:

```text
TargetSchedule
TargetScheduleCandidate
TargetScheduleProposal
TargetScheduleObservation
```

These represent the new online action space: `AgentExplorer -> target schedule`.

## Observation Semantics

Online observations should preserve:

```text
final_target
scheduled_target
schedule_meta
explicit_structure
evaluated_property
error_to_scheduled_target
error_to_final_target
validity
label
provenance
```

`error_to_scheduled_target` measures how well InverseDesigner satisfied the requested scheduled target.

`error_to_final_target` measures whether the sample helps solve the user's final target.

## Recommended Reading Order

1. `docs/context.md`
2. `src/closed_loop_contracts.py`
3. `src/Scheduler/closed_loop.py`
4. `src/Scheduler/feedback.py`
5. `src/KnowledgeBase/`
6. `src/AgentExplorer/`
7. `src/InverseDesigner/`
8. `src/DatagenFEMEvaluator/`

## One-Line Summary

```text
Datagen is offline bootstrap; AgentExplorer schedules targets; InverseDesigner generates structures; FEM validates; Knowledge guides the next schedule.
```
