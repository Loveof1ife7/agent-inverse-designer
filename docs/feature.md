# Feature Design

The main feature is a target-schedule closed loop for inverse material design.

## Core Loop

```text
final target T
  -> AgentExplorer proposes target schedule S_n
  -> InverseDesigner generates explicit structures
  -> FEMEvaluator evaluates structures
  -> RawExperimentStore records observations
  -> KnowledgeBase stores schedule evidence
  -> InverseDesigner dataset is updated
  -> FeedbackSignal decides stop or continue
```

## Feature 1: Target Schedule Design

AgentExplorer chooses which targets to ask InverseDesigner to realize.

Schedule examples:

```text
near-target exploitation
reachable-region probing
property-boundary exploration
counterfactual targets
curriculum targets moving toward final T
```

AgentExplorer should use:

```text
KnowledgeBase evidence
refined knowledge summaries
previous FeedbackSignal
final target T
```

## Feature 2: InverseDesigner Self-Improvement

InverseDesigner generates structures from scheduled targets.

FEM results become new supervised data:

```text
evaluated_property -> explicit structure
scheduled_target -> explicit structure
```

Dataset curation should weight samples by:

```text
validity
distance to scheduled target
distance to final target
novelty
coverage of underexplored target regions
```

## Feature 3: Knowledge Evidence

KnowledgeBase stores schedule-level evidence:

```text
final_target
scheduled_target
schedule_strategy
hypothesis
structure
evaluated_property
error_to_scheduled_target
error_to_final_target
label
provenance
```

Knowledge should answer:

```text
Which scheduled targets are reachable?
Which schedules improve progress toward T?
Where does InverseDesigner over/under-shoot?
Which failures are repeated?
Which samples should be used for finetuning?
```

## Feature 4: FeedbackSignal

FeedbackSignal is not long-term knowledge. It is the current control state:

```text
success_found
best_success
best_near_miss
best_sample
representative_failures
next_anchor
suggested_strategy
```

It is extracted from the current round and final target.

## Feature 5: Offline Datagen

DatagenFEMEvaluator is an offline data factory:

```text
generate many structures
evaluate properties
build cold-start dataset
pretrain InverseDesigner
```

It should not be the online exploration engine.

## Summary

```text
Dataset trains InverseDesigner.
Knowledge guides AgentExplorer.
FeedbackSignal controls the next step.
FEM supplies truth.
```
