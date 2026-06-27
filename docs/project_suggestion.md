# Project Design Suggestion

This document records the current closed-loop direction.

The project should use a target-schedule closed loop:

```text
AgentExplorer chooses targets.
InverseDesigner generates structures.
FEMEvaluator measures truth.
KnowledgeBase records what target-space moves work.
```

## System Objective

Given a final property target `T`, discover explicit material structures whose FEM-evaluated properties satisfy `T`.

The online loop should not use datagen as the search engine. Datagen is reserved for offline data generation and pretraining.

## Offline Stage

```text
DatagenFEMEvaluator
  -> generate large structure-property dataset
  -> FEM / proxy-FEM evaluate
  -> build training pairs
  -> pretrain InverseDesigner
```

Training pair:

```text
property / target -> explicit structure
```

## Online Stage

```text
input final target T

loop:
  AgentExplorer proposes target schedule S_n
  InverseDesigner samples structures for S_n
  FEMEvaluator evaluates structures
  RawExperimentStore records observations
  KnowledgeBase stores schedule evidence
  Dataset curator updates InverseDesigner replay / finetune data
  FeedbackSignal checks progress against T
```

## KnowledgeBase

Knowledge is target-space exploration memory.

It stores:

```text
final_target
scheduled_target
schedule_meta
structure
evaluated_property
error_to_scheduled_target
error_to_final_target
label
provenance
```

It supports:

```text
AgentExplorer target schedule design
InverseDesigner capability analysis
finetune dataset selection
```

## AgentExplorer

AgentExplorer no longer outputs datagen parameters in the closed loop.

It outputs a target schedule:

```text
TargetSchedule:
  final_target
  scheduled_targets
  strategy
  hypothesis
  reason
  confidence
```

It should use knowledge to decide whether to exploit near `T`, probe reachable regions, run counterfactual targets, or build a curriculum toward `T`.

## InverseDesigner

InverseDesigner owns the inverse mapping:

```text
target -> explicit structure
```

During the loop it should support:

```text
sample_structure(target)
sample_schedule(target_schedule)
update_replay_dataset(observations)
finetune(selected_samples)
```

## FEMEvaluator

The online loop only needs:

```text
explicit structure -> evaluated property / validity / metrics
```

Datagen-specific generation APIs should stay outside the online scheduler.

## Principle

```text
Explore target space, not datagen parameter space.
```
