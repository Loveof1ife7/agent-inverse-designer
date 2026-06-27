# Feedback Signal

`FeedbackSignal` is the scheduler control signal produced after one online loop iteration.

It is not knowledge, not a second memory, and not the AgentExplorer reasoning module. It is a small deterministic summary of the current round, always measured against the user's final target `T`.

## Purpose

After one target schedule is executed, the system has many observations:

```text
scheduled_target T_i
generated structure_i
evaluated_property_i
error_to_scheduled_target_i
error_to_final_target_i
label_i
validity_i
```

`FeedbackSignalExtractor` compresses these observations into:

```text
Did we solve final target T?
If not, what is the best current anchor?
What failure mode should the next schedule know about?
Should the scheduler stop, continue, exploit, repair, or explore?
```

## Inputs

```text
final_target T
current iteration observations
```

It may read the current round's observations and labels, but it should not read the full historical KB or call an LLM.

## Outputs

Recommended fields:

```text
success_found
should_stop
best_success
best_near_miss
best_sample
next_anchor
representative_failures
failure_modes
main_error_direction
suggested_next_action
feedback_samples
```

Meanings:

```text
success_found:
  at least one observation satisfies final target T

should_stop:
  scheduler should stop; first version can set this equal to success_found

best_success:
  best successful observation measured by error_to_final_target

best_near_miss:
  closest near-miss observation measured by error_to_final_target

best_sample:
  closest observation overall measured by error_to_final_target

next_anchor:
  the observation the next schedule should condition on if the loop continues

representative_failures:
  compact set of failures that describe what went wrong this round

failure_modes:
  short structured labels such as invalid_geometry, fem_failed, stiffness_low, density_high

main_error_direction:
  property dimension with the largest remaining error to final target T

suggested_next_action:
  stop, exploit_near_miss, repair_failure, probe_reachability, or explore

feedback_samples:
  small display/report subset for run summary and result payload
```

## Scheduler Use

The scheduler uses `FeedbackSignal` for immediate control:

```text
if feedback.should_stop:
  return success

else:
  pass feedback.next_anchor and failure summary to AgentExplorer
  continue with another target schedule
```

It should not decide long-term strategy by itself. Long-term strategy comes from `AgentExplorer` reading `KnowledgeBase`.

## AgentExplorer Use

AgentExplorer can consume `FeedbackSignal` as local context:

```text
best current anchor
main error direction
current failure modes
suggested next action
```

Then AgentExplorer combines it with long-term knowledge to design the next target schedule.

Example:

```text
FeedbackSignal:
  best_near_miss has stiffness too low
  density is acceptable
  suggested_next_action = exploit_near_miss

AgentExplorer:
  reads KB to find target offsets that improved stiffness without increasing density
  proposes next target schedule around those offsets
```

## Knowledge Boundary

`FeedbackSignal` answers:

```text
Given final target T and this round's observations, what should happen next?
```

`KnowledgeBase` answers:

```text
Across all rounds, what target schedules worked, failed, or revealed bias?
```

So:

```text
FeedbackSignal = local control state
KnowledgeBase = accumulated target-space memory
```

## Minimal First Version

Distance rules:

```text
distance_to_final_target = max(error_to_final_target.values())
distance_to_scheduled_target = max(error_to_scheduled_target.values())
```

Selection rules:

```text
best_success:
  success sample with smallest distance_to_final_target

best_near_miss:
  near_miss with smallest distance_to_final_target

best_sample:
  any sample with smallest distance_to_final_target

next_anchor:
  best_success if found
  else best_near_miss
  else best_sample

should_stop:
  success_found
```

Suggested action rules:

```text
success_found:
  stop

best_near_miss exists:
  exploit_near_miss

representative failure is invalid/fem_failed:
  repair_failure

no useful sample:
  probe_reachability

otherwise:
  explore
```

## Information Flow

```text
FEM observations
  -> RawExperimentStore
  -> KnowledgeEvidence
  -> KnowledgeBase
  -> AgentExplorer long-term reasoning

FEM observations + final target T
  -> FeedbackSignalExtractor
  -> scheduler control
  -> AgentExplorer local context
```

## Principle

```text
FeedbackSignal decides the next control move.
Knowledge decides what the system has learned.
AgentExplorer designs the next target schedule.
```
