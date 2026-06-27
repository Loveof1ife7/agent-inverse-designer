# Closed-Loop Discovery Demo

This document describes the target-schedule closed-loop design.

The current implementation may still contain legacy datagen-in-loop paths during migration. The intended workflow is below.

## Workflow

```text
offline:
  DatagenFEMEvaluator generates a large pretraining dataset
  InverseDesigner is pretrained on target/property -> explicit structure

online:
  final target T
  -> AgentExplorer proposes target schedule S_n = [T_1, T_2, ..., T_k]
  -> InverseDesigner samples structures for scheduled targets
  -> FEMEvaluator evaluates explicit structures with Abaqus when available
  -> RawExperimentStore records observations
  -> KnowledgeBase stores target-schedule evidence
  -> InverseDesigner replay / finetune dataset is updated
  -> FeedbackSignal summarizes current-round control
```

## What The Demo Should Show

1. `InverseDesigner` starts from a cold-start pretrained dataset.
2. The user gives a final target `T`.
3. `AgentExplorer` reads feedback and knowledge, then proposes a target schedule.
4. `InverseDesigner` generates explicit structures for each scheduled target.
5. FEM evaluation produces true properties, validity, and raw metrics.
6. Each observation records both `final_target` and `scheduled_target`.
7. Knowledge evidence captures whether the schedule helped reach `T`.
8. New FEM-validated samples update the inverse-designer dataset.
9. `FeedbackSignal` reports `should_stop`, `next_anchor`, failure modes, and suggested next action.
10. The loop stops when a generated structure satisfies the final target.

## Observation Payload

Each raw observation should include:

```text
final_target
scheduled_target
schedule_id
schedule_strategy
hypothesis
reason
structure
evaluated_property
error_to_scheduled_target
error_to_final_target
geometry_status
fem_status
label
provenance
```

## Outputs

Recommended output layout:

```text
workspace/closed_loop_discovery_demo/
  inverse_pretraining_dataset.jsonl
  inverse_replay_dataset.jsonl
  knowledge.sqlite
  closed_loop_events.jsonl
  experiments/<task_id>/raw_experiments.jsonl
  experiments/<task_id>/run_summary.md
  demo_summary.json
  demo_summary.md
```

## FEM Note

The online loop only needs explicit-structure evaluation:

```text
explicit structure -> FEM property
```

When `fem_backend=abaqus`, explicit coordinate/edge structures are first written as truss txt files, then evaluated through the same Abaqus job and ODB extraction path used by `dynamic.py`. When `fem_backend=auto`, the loop uses Abaqus if it is discoverable and falls back to proxy otherwise. It should not call datagen to produce online candidates. Datagen remains the offline data factory for cold start.
