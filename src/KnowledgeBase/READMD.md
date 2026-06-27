# KnowledgeBase Role

`KnowledgeBase` is the long-term memory for target-space exploration.

It is not just a sample table and not just a log. In the new closed loop, it stores evidence about:

```text
target schedule -> InverseDesigner structure -> FEM property -> errors -> label
```

## Core Use

Knowledge has three jobs:

```text
1. guide AgentExplorer when designing the next target schedule
2. describe where InverseDesigner is reliable or biased
3. select useful samples for InverseDesigner finetune / replay
```

## Evidence Unit

A `KnowledgeEvidence` record should preserve:

```text
final_target
scheduled_target
schedule_id
schedule_step
schedule_strategy
hypothesis
reason
structure_id
structure / structure_path
evaluated_property
error_to_scheduled_target
error_to_final_target
geometry_status
fem_status
label
provenance
```

The two errors have different meanings:

```text
error_to_scheduled_target:
  Did InverseDesigner realize the requested target T_i?

error_to_final_target:
  Did this sample help solve the user's final target T?
```

## What Knowledge Answers

AgentExplorer should be able to ask:

```text
Which scheduled targets were reachable?
Which target offsets improved distance to final target?
Which schedule strategies produced near misses?
Which target regions caused invalid structures or FEM failures?
Where does InverseDesigner systematically over/under-shoot?
Which regions need more exploration data?
```

The scheduler / dataset curator should be able to ask:

```text
Which new observations should enter the finetune dataset?
Which samples are high-value corrections?
Which samples are duplicates and can be downweighted?
Which invalid samples should be kept as negative evidence?
```

## Dataset vs Knowledge

The same raw observation can produce two projections:

```text
Dataset projection:
  target/property -> explicit structure
  used to train InverseDesigner

Knowledge projection:
  schedule intent + generated structure + FEM result + error + evidence
  used by AgentExplorer and scheduler decisions
```

Dataset is for learning the inverse mapping. Knowledge is for deciding where to explore next.

## Explicit Structure

`explicit_structure` should be persisted when available. It is used for:

```text
replay / finetune
re-evaluation
visualization
similarity search
provenance
```

Knowledge does not need to fully reason over raw nodes and edges in the first version. It should still retain the structure payload so later versions can add geometric reasoning.

## Legacy Compatibility

Existing fields such as `meta_json`, `parameter_config`, `structure_code`, and old datagen provenance may remain for migration.

New closed-loop evidence should prefer target-schedule semantics:

```text
final_target
scheduled_target
schedule_meta
explicit_structure
evaluated_property
error_to_scheduled_target
error_to_final_target
```
