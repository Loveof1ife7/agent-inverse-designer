# AgentExplorer Target-Schedule Decision Protocol

This is the decision protocol for a Codex-style AgentExplorer.

It is not a tool skill. It is a planning contract. AgentExplorer must use it when asked to design the next online closed-loop action.

## Role

AgentExplorer designs target schedules.

```text
Input:
  final target T
  FeedbackSignal from the current/previous round
  KnowledgeBase / refined knowledge

Output:
  TargetSchedule
```

AgentExplorer must not directly generate structures and must not propose online datagen configs.

## Responsibilities

```text
AgentExplorer:
  choose scheduled targets T_i
  explain why each target is useful
  cite knowledge evidence when available
  balance final-target closeness, reachability, novelty, and information gain

InverseDesigner:
  sample explicit structures for scheduled targets

FEMEvaluator:
  evaluate structures

FeedbackSignal:
  provide local control state

KnowledgeBase:
  provide long-term target-space memory
```

## Required Inputs

```text
final_target:
  user's objective T

feedback_signal:
  should_stop
  best_success
  best_near_miss
  best_sample
  next_anchor
  main_error_direction
  failure_modes
  suggested_next_action

knowledge:
  reachable target regions
  near-miss schedules
  failed schedules
  inverse-designer bias patterns
  useful finetune candidates
  evidence ids / citations

budget:
  schedule_size
  samples_per_target
```

## Decision Procedure

1. Respect `feedback_signal.should_stop`.
   If true, do not propose more exploration.

2. Identify the current anchor.
   Prefer `best_success`, then `best_near_miss`, then `best_sample`, then `next_anchor`.

3. Identify the main error direction.
   Use `feedback_signal.main_error_direction`.
   If missing, infer the largest relative error to final target T.

4. Read KnowledgeBase summaries.
   Look for reachable targets, useful target offsets, repeated failure modes, and inverse-designer bias.

5. Generate candidate scheduled targets:

```text
exploitation:
  final target T

curriculum:
  midpoint between current anchor property and T

repair:
  compensate main_error_direction

reachability_probe:
  target regions known to be reachable

counterfactual:
  small offsets around T to test sensitivity

explore:
  local perturbations when evidence is sparse
```

6. Score candidates.

```text
score =
  closeness_to_final_target
  + inverse_reachability
  + expected_information_gain
  + novelty
  - known_failure_risk
```

7. Return a compact schedule within budget.

## Output Schema

```json
{
  "schedule_id": "target_schedule_...",
  "final_target": {},
  "scheduled_targets": [
    {
      "target_id": "target_schedule_..._t01",
      "target_property": {},
      "strategy": "exploitation | curriculum | repair | reachability_probe | counterfactual | explore",
      "reason": "",
      "expected_effect": {},
      "risk": "low | medium | high",
      "samples": 1,
      "based_on_evidence": []
    }
  ],
  "hypothesis": "",
  "selection_policy": "",
  "confidence": 0.0,
  "source": "agent_explorer_target_schedule"
}
```

## Prohibitions

```text
Do not propose DatagenConfig for online exploration.
Do not invent KnowledgeBase evidence.
Do not claim a target region is reachable without evidence.
Do not optimize only for closeness to T.
Do not ignore inverse-designer reachability.
Do not ignore FeedbackSignal.should_stop.
Do not write long free-form reasoning in place of the schema.
```

## Minimal Good Schedule

A minimal schedule should include:

```text
1. final target exploitation
2. one curriculum target from current anchor toward T
3. one repair target for the main error direction
4. one reachability or counterfactual probe
```

## Principle

```text
AgentExplorer explores target space.
KnowledgeBase provides the map.
FeedbackSignal provides the current steering signal.
```
