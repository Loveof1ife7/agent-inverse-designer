# Knowledge Base Design

The Knowledge Base is a semantic memory built on structured storage.

It is different from raw logs and different from the training dataset.

## Layered Design

```text
Layer 0: RawExperimentStore
  append-only record of evaluated observations

Layer 1: Dataset projection
  compact training rows for InverseDesigner

Layer 2: SQLite storage
  structured persistence for samples, evidence, runs, artifacts

Layer 3: KnowledgeBase interface
  query and reasoning-oriented memory for AgentExplorer
```

## Raw Observation

Each online observation should preserve:

```text
observation_id
final_target
scheduled_target
schedule_id
schedule_step
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
raw_metrics
artifacts
provenance
```

Raw observations answer:

```text
What exactly happened?
```

## Dataset Projection

Dataset rows train or finetune `InverseDesigner`.

Typical row:

```text
input_property
output_structure
weight
validity_flag
fidelity_flag
source
```

Weights can reflect:

```text
success / near_miss / failure
validity
novelty
distance to scheduled target
distance to final target
coverage of underexplored target regions
```

Dataset answers:

```text
What should InverseDesigner learn from?
```

## Knowledge Evidence

Knowledge evidence is the semantic projection:

```text
target schedule -> structure -> FEM property -> errors -> label
```

It should include:

```text
final_target
scheduled_target
schedule_meta
structure_features
property_result
error_to_scheduled_target
error_to_final_target
label
hypothesis
reasoning
supports_hypothesis
contradicts_hypothesis
provenance
confidence
```

Knowledge answers:

```text
Which target schedules worked?
Which target regions are reachable?
Where is InverseDesigner biased?
Which failures repeat?
Which evidence should guide the next schedule?
```

## Query Interface

AgentExplorer should use high-level queries, not raw SQL.

Recommended query families:

```text
kb.query_reachable_targets(final_target, top_k)
kb.query_near_miss_schedules(final_target, top_k)
kb.query_failed_schedules(final_target, top_k)
kb.query_inverse_bias(target_region)
kb.query_schedule_strategy_stats(strategy=None)
kb.query_finetune_candidates(target_region=None, top_k=100)
kb.get_observation_evidence(observation_id)
kb.get_schedule_evidence(schedule_id)
```

Legacy sample queries can remain during migration:

```text
success
near_miss
failure
similar_property
```

## Principle

```text
RawStore preserves facts.
Dataset trains InverseDesigner.
KnowledgeBase guides target schedule design.
```
