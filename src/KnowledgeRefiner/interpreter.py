from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import requests

from ..closed_loop_contracts import AgentKnowledgeInterpretation, ReasonedAgentKnowledgeSnapshot


DEFAULT_KNOWLEDGE_INTERPRETER_API_KEY = "e339901aa965318c253573770ae65bd0cfce8b93ca057ddc79725f80186382db"
DEFAULT_KNOWLEDGE_INTERPRETER_BASE_URL = "http://127.0.0.1:17777/v1"
DEFAULT_KNOWLEDGE_INTERPRETER_MODEL = "gpt-5.5"
MODEL_PREFERENCE = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_evidence_ids(rows: list[dict[str, Any]], limit: int = 20) -> list[str]:
    citations: list[str] = []
    seen = set()
    for row in rows:
        for key in ("example_evidence_ids", "evidence_ids"):
            for evidence_id in row.get(key, []) or []:
                if not evidence_id or evidence_id in seen:
                    continue
                seen.add(evidence_id)
                citations.append(str(evidence_id))
                if len(citations) >= limit:
                    return citations
    return citations


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("llm response does not contain a JSON object")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("llm response JSON must be an object")
    return payload


def _iter_sse_events(raw_text: str):
    event_name = ""
    data_lines: list[str] = []
    for line in raw_text.splitlines():
        if not line:
            if data_lines:
                yield event_name or "message", "\n".join(data_lines)
                event_name = ""
                data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if data_lines:
        yield event_name or "message", "\n".join(data_lines)


class AgentKnowledgeInterpreter:
    """
    Interpret deterministic statistics into target-conditioned mechanism guidance.

    Agent mode tries a role-specific LLM interpretation first. Deterministic
    interpretation remains the fallback and never lets the LLM rewrite facts.
    """

    name = "deterministic_agent_knowledge_interpreter_v1"
    llm_name = "llm_agent_knowledge_interpreter_v1"
    _global_llm_disable_reason = ""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        enable_llm: bool | None = None,
    ):
        env_flag = os.getenv("KNOWLEDGE_INTERPRETER_ENABLE_LLM", "").strip().lower()
        if enable_llm is None:
            if env_flag:
                enable_llm = env_flag not in {"0", "false", "no", "off"}
            else:
                enable_llm = True

        self.enable_llm = bool(enable_llm)
        self.api_key = (
            api_key
            or os.getenv("KNOWLEDGE_INTERPRETER_API_KEY")
            or os.getenv("AGENT_EXPLORER_API_KEY")
            or DEFAULT_KNOWLEDGE_INTERPRETER_API_KEY
        )
        self.base_url = (
            base_url
            or os.getenv("KNOWLEDGE_INTERPRETER_BASE_URL")
            or os.getenv("AGENT_EXPLORER_BASE_URL")
            or DEFAULT_KNOWLEDGE_INTERPRETER_BASE_URL
        ).rstrip("/")
        self.model = model or os.getenv("KNOWLEDGE_INTERPRETER_MODEL") or DEFAULT_KNOWLEDGE_INTERPRETER_MODEL
        self.timeout = float(timeout)
        self._session: requests.Session | None = None
        self._instance_llm_disable_reason = ""

    def interpret(
        self,
        statistical_snapshot: dict[str, Any],
        target_property: dict[str, float] | None = None,
        failed_candidate: dict[str, Any] | None = None,
        scheduler_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target_property = dict(target_property or {})
        failed_candidate = dict(failed_candidate or {})
        scheduler_context = dict(scheduler_context or {})

        deterministic = self._deterministic_interpret(
            statistical_snapshot=statistical_snapshot,
            target_property=target_property,
            failed_candidate=failed_candidate,
            scheduler_context=scheduler_context,
        )
        llm_payload, llm_error = self._try_llm_interpretation(
            statistical_snapshot=statistical_snapshot,
            target_property=target_property,
            failed_candidate=failed_candidate,
            scheduler_context=scheduler_context,
            deterministic=deterministic,
        )
        if llm_payload is not None:
            return self._interpretation_from_llm_payload(
                payload=llm_payload,
                deterministic=deterministic,
                target_property=target_property,
            )
        if llm_error:
            deterministic = dict(deterministic)
            deterministic["fallback_reason"] = llm_error
        return deterministic

    def _deterministic_interpret(
        self,
        statistical_snapshot: dict[str, Any],
        target_property: dict[str, float],
        failed_candidate: dict[str, Any],
        scheduler_context: dict[str, Any],
    ) -> dict[str, Any]:
        mechanism_explanations = self._mechanism_explanations(statistical_snapshot)
        dominant_failure_causes = self._dominant_failure_causes(statistical_snapshot, failed_candidate)
        promising = self._promising_interventions(statistical_snapshot)
        rejected = self._rejected_interventions(statistical_snapshot)
        causal_tests = self._causal_tests(statistical_snapshot, dominant_failure_causes, promising, rejected)
        strategy_prior = self._target_strategy_prior(
            target_property=target_property,
            failed_candidate=failed_candidate,
            dominant_failure_causes=dominant_failure_causes,
            promising_interventions=promising,
            scheduler_context=scheduler_context,
        )
        evidence_citations = _collect_evidence_ids(
            mechanism_explanations + dominant_failure_causes + promising + rejected + causal_tests
        )

        interpretation = AgentKnowledgeInterpretation(
            interpreter=self.name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            target_property=target_property,
            mechanism_explanations=mechanism_explanations,
            dominant_failure_causes=dominant_failure_causes,
            promising_intervention_families=promising,
            rejected_intervention_families=rejected,
            target_specific_strategy_prior=strategy_prior,
            causal_tests_to_run=causal_tests,
            evidence_citations=evidence_citations,
            confidence=self._confidence(statistical_snapshot, evidence_citations),
        )
        return interpretation.to_dict()

    def _try_llm_interpretation(
        self,
        statistical_snapshot: dict[str, Any],
        target_property: dict[str, float],
        failed_candidate: dict[str, Any],
        scheduler_context: dict[str, Any],
        deterministic: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not self.enable_llm:
            return None, None
        if self._instance_llm_disable_reason:
            return None, self._instance_llm_disable_reason
        if self.__class__._global_llm_disable_reason:
            return None, self.__class__._global_llm_disable_reason
        try:
            payload = self._request_llm_payload(
                statistical_snapshot=statistical_snapshot,
                target_property=target_property,
                failed_candidate=failed_candidate,
                scheduler_context=scheduler_context,
                deterministic=deterministic,
            )
            return payload, None
        except Exception as exc:  # pragma: no cover - exercised by integration fallback
            reason = f"llm_unavailable: {type(exc).__name__}: {exc}"
            self._instance_llm_disable_reason = reason
            self.__class__._global_llm_disable_reason = reason
            return None, reason

    def _http_session(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            session.trust_env = False
            self._session = session
        return self._session

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        response = self._http_session().get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        model_ids = [str(item.get("id", "")).strip() for item in payload.get("data", []) if item.get("id")]
        for preferred in MODEL_PREFERENCE:
            if preferred in model_ids:
                self.model = preferred
                return self.model
        if not model_ids:
            raise RuntimeError("no models available from knowledge interpreter endpoint")
        self.model = model_ids[0]
        return self.model

    def _request_llm_payload(
        self,
        statistical_snapshot: dict[str, Any],
        target_property: dict[str, float],
        failed_candidate: dict[str, Any],
        scheduler_context: dict[str, Any],
        deterministic: dict[str, Any],
    ) -> dict[str, Any]:
        model = self._resolve_model()
        prompt_payload = {
            "target_property": dict(target_property),
            "failed_candidate": failed_candidate,
            "scheduler_context": scheduler_context,
            "statistical_snapshot": {
                "total_evidence": statistical_snapshot.get("total_evidence", statistical_snapshot.get("total_samples", 0)),
                "mechanism_patterns": self._limit_payload(statistical_snapshot.get("mechanism_patterns", []), 8),
                "failure_patterns": self._limit_payload(statistical_snapshot.get("failure_patterns", []), 8),
                "intervention_effects": self._limit_payload(statistical_snapshot.get("intervention_effects", []), 8),
                "hypothesis_status": self._limit_payload(statistical_snapshot.get("hypothesis_status", []), 8),
                "bad_regions": self._limit_payload(statistical_snapshot.get("bad_regions", []), 8),
                "useful_exemplars": self._limit_payload(statistical_snapshot.get("useful_exemplars", {}), 3),
            },
            "deterministic_baseline": deterministic,
        }
        system_prompt = (
            "You are KnowledgeInterpreter Agent in an agent-guided closed-loop scientific discovery system. "
            "Your task is evidence -> explanation only. You must not propose executable Meta batches, "
            "must not invent evidence IDs, and must distinguish correlation from causal claims. "
            "Use conservative mechanism language and cite evidence IDs from the input."
        )
        user_prompt = (
            "Return one JSON object with exactly these fields: "
            "mechanism_explanations, dominant_failure_causes, promising_intervention_families, "
            "rejected_intervention_families, target_specific_strategy_prior, causal_tests_to_run, "
            "evidence_citations, confidence. "
            "Each explanation/test should include evidence IDs when available. "
            "Use hypothesis status labels supported, partially_supported, contradicted, or inconclusive when relevant. "
            "Do not output markdown.\n\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
        )
        response = self._http_session().post(
            f"{self.base_url}/responses",
            headers=self._headers(),
            json={
                "model": model,
                "input": f"{system_prompt}\n\n{user_prompt}",
                "reasoning": {"effort": "high"},
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            message = response.text.strip().replace("\n", " ")
            raise RuntimeError(f"responses_failed status={response.status_code} body={message[:300]}")
        content = self._extract_responses_output(response.text)
        return _extract_json_object(content)

    @staticmethod
    def _extract_responses_output(raw_text: str) -> str:
        output_chunks: list[str] = []
        done_texts: list[str] = []
        for event_name, data in _iter_sse_events(raw_text):
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event_name == "response.output_text.delta":
                delta = payload.get("delta", "")
                if isinstance(delta, str) and delta:
                    output_chunks.append(delta)
                continue
            if event_name == "response.failed":
                error = payload.get("response", {}).get("error") or payload.get("error") or payload
                raise RuntimeError(json.dumps(error, ensure_ascii=False))
            if event_name.endswith(".done"):
                text = payload.get("text")
                if isinstance(text, str) and text:
                    done_texts.append(text)
        content = "".join(output_chunks).strip()
        if content:
            return content
        fallback = "".join(done_texts).strip()
        if fallback:
            return fallback
        raw = raw_text.strip()
        if raw:
            return raw
        raise RuntimeError("responses returned empty output")

    def _interpretation_from_llm_payload(
        self,
        payload: dict[str, Any],
        deterministic: dict[str, Any],
        target_property: dict[str, float],
    ) -> dict[str, Any]:
        def list_of_dicts(key: str) -> list[dict[str, Any]]:
            value = payload.get(key, deterministic.get(key, []))
            if not isinstance(value, list):
                return list(deterministic.get(key, []))
            return [dict(item) for item in value if isinstance(item, dict)]

        evidence_citations = payload.get("evidence_citations", deterministic.get("evidence_citations", []))
        if not isinstance(evidence_citations, list):
            evidence_citations = deterministic.get("evidence_citations", [])
        evidence_citations = [str(item) for item in evidence_citations if item]
        if not evidence_citations:
            evidence_citations = _collect_evidence_ids(
                list_of_dicts("mechanism_explanations")
                + list_of_dicts("dominant_failure_causes")
                + list_of_dicts("promising_intervention_families")
                + list_of_dicts("rejected_intervention_families")
                + list_of_dicts("causal_tests_to_run")
            )

        strategy_prior = payload.get("target_specific_strategy_prior", deterministic.get("target_specific_strategy_prior", {}))
        if not isinstance(strategy_prior, dict):
            strategy_prior = dict(deterministic.get("target_specific_strategy_prior", {}))

        interpretation = AgentKnowledgeInterpretation(
            interpreter=self.llm_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            target_property=target_property,
            mechanism_explanations=list_of_dicts("mechanism_explanations"),
            dominant_failure_causes=list_of_dicts("dominant_failure_causes"),
            promising_intervention_families=list_of_dicts("promising_intervention_families"),
            rejected_intervention_families=list_of_dicts("rejected_intervention_families"),
            target_specific_strategy_prior=dict(strategy_prior),
            causal_tests_to_run=list_of_dicts("causal_tests_to_run"),
            evidence_citations=evidence_citations[:30],
            confidence=_clamp(_safe_float(payload.get("confidence"), deterministic.get("confidence", 0.5)), 0.05, 0.99),
        )
        return interpretation.to_dict()

    @staticmethod
    def _limit_payload(payload: Any, limit: int) -> Any:
        if isinstance(payload, list):
            return payload[:limit]
        if isinstance(payload, dict):
            limited = {}
            for key, value in payload.items():
                if isinstance(value, list):
                    limited[key] = value[:limit]
                elif isinstance(value, dict):
                    limited[key] = dict(list(value.items())[:limit])
                else:
                    limited[key] = value
            return limited
        return payload

    def build_reasoned_snapshot(
        self,
        statistical_snapshot: dict[str, Any],
        target_property: dict[str, float] | None = None,
        failed_candidate: dict[str, Any] | None = None,
        scheduler_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        interpretation = self.interpret(
            statistical_snapshot=statistical_snapshot,
            target_property=target_property,
            failed_candidate=failed_candidate,
            scheduler_context=scheduler_context,
        )
        return ReasonedAgentKnowledgeSnapshot(
            generated_at=datetime.now(timezone.utc).isoformat(),
            statistical_snapshot=dict(statistical_snapshot),
            interpretation=interpretation,
        ).to_dict()

    def _mechanism_explanations(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        explanations = []
        for pattern in snapshot.get("mechanism_patterns", [])[:5]:
            useful_rate = _safe_float(pattern.get("useful_rate"))
            avg_error = _safe_float(pattern.get("avg_property_error"))
            label = "promising_mechanism" if useful_rate >= 0.5 else "weak_mechanism"
            explanations.append(
                {
                    "mechanism": label,
                    "summary": (
                        f"{pattern.get('symmetry', '')}/{pattern.get('connectivity_pattern', '')} "
                        f"with max_bars={pattern.get('max_bars')} has useful_rate={useful_rate:.3f} "
                        f"and avg_error={avg_error:.3f}."
                    ),
                    "meta_pattern": {
                        "symmetry": pattern.get("symmetry", ""),
                        "topology_type": pattern.get("topology_type", ""),
                        "connectivity_pattern": pattern.get("connectivity_pattern", ""),
                        "max_bars": pattern.get("max_bars"),
                        "density_range": pattern.get("density_range", []),
                    },
                    "supporting_metrics": {
                        "num_samples": pattern.get("num_samples", 0),
                        "success_rate": pattern.get("success_rate", 0.0),
                        "useful_rate": useful_rate,
                        "avg_property_error": avg_error,
                    },
                    "example_evidence_ids": list(pattern.get("example_evidence_ids", []) or []),
                    "confidence": min(0.9, 0.35 + useful_rate + 0.03 * _safe_float(pattern.get("num_samples"))),
                }
            )
        return explanations

    def _dominant_failure_causes(
        self,
        snapshot: dict[str, Any],
        failed_candidate: dict[str, Any],
    ) -> list[dict[str, Any]]:
        failed_eval = dict(failed_candidate.get("evaluation") or {})
        failed_label = failed_eval.get("label") or failed_candidate.get("label") or ""
        rows = []
        for failure in snapshot.get("failure_patterns", [])[:5]:
            reason = str(failure.get("failure_reason") or "property_mismatch")
            count = int(failure.get("count") or 0)
            rows.append(
                {
                    "failure_cause": reason,
                    "summary": f"{reason} appears in {count} evidence rows.",
                    "count": count,
                    "groups": dict(failure.get("groups") or {}),
                    "matches_current_failure": bool(failed_label and failed_label != "success"),
                    "example_evidence_ids": list(failure.get("example_evidence_ids", []) or []),
                    "confidence": min(0.9, 0.4 + 0.05 * count),
                }
            )
        return rows

    def _promising_interventions(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for intervention in snapshot.get("intervention_effects", []):
            useful_rate = _safe_float(intervention.get("useful_rate"))
            if useful_rate < 0.4:
                continue
            rows.append(
                {
                    "intervention_type": intervention.get("intervention_type", "unspecified"),
                    "summary": (
                        f"{intervention.get('intervention_type', 'unspecified')} has useful_rate={useful_rate:.3f} "
                        f"over {intervention.get('count', 0)} trials."
                    ),
                    "useful_rate": useful_rate,
                    "avg_error": _safe_float(intervention.get("avg_error")),
                    "example_evidence_ids": list(intervention.get("example_evidence_ids", []) or []),
                    "confidence": min(0.9, 0.35 + useful_rate),
                }
            )
        return rows[:5]

    def _rejected_interventions(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for intervention in snapshot.get("intervention_effects", []):
            useful_rate = _safe_float(intervention.get("useful_rate"))
            if useful_rate >= 0.25:
                continue
            rows.append(
                {
                    "intervention_type": intervention.get("intervention_type", "unspecified"),
                    "summary": (
                        f"{intervention.get('intervention_type', 'unspecified')} is weak in current evidence "
                        f"(useful_rate={useful_rate:.3f})."
                    ),
                    "useful_rate": useful_rate,
                    "avg_error": _safe_float(intervention.get("avg_error")),
                    "example_evidence_ids": list(intervention.get("example_evidence_ids", []) or []),
                    "confidence": min(0.85, 0.45 + (0.25 - useful_rate)),
                }
            )
        return rows[:5]

    def _causal_tests(
        self,
        snapshot: dict[str, Any],
        failures: list[dict[str, Any]],
        promising: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tests = []
        bad_regions = snapshot.get("bad_regions", [])
        if failures:
            dominant = failures[0]
            tests.append(
                {
                    "test_type": "repair_dominant_failure",
                    "hypothesis": f"Repairing {dominant['failure_cause']} should reduce target error.",
                    "recommended_intervention": promising[0]["intervention_type"] if promising else "repair",
                    "control": rejected[0]["intervention_type"] if rejected else "no_change",
                    "example_evidence_ids": list(dominant.get("example_evidence_ids", []) or []),
                    "confidence": dominant.get("confidence", 0.5),
                }
            )
        for region in bad_regions[:3]:
            tests.append(
                {
                    "test_type": "avoid_or_contrast_bad_region",
                    "hypothesis": f"Region {region.get('symmetry', '')}/{region.get('connectivity_pattern', '')} is failure-prone.",
                    "recommended_intervention": "counterfactual",
                    "control": "repeat_bad_region",
                    "bad_region": {
                        "symmetry": region.get("symmetry", ""),
                        "topology_type": region.get("topology_type", ""),
                        "connectivity_pattern": region.get("connectivity_pattern", ""),
                        "max_bars": region.get("max_bars"),
                    },
                    "example_evidence_ids": list(region.get("example_evidence_ids", []) or []),
                    "confidence": min(0.85, 0.35 + 0.05 * _safe_float(region.get("failure_count"))),
                }
            )
        return tests[:5]

    def _target_strategy_prior(
        self,
        target_property: dict[str, float],
        failed_candidate: dict[str, Any],
        dominant_failure_causes: list[dict[str, Any]],
        promising_interventions: list[dict[str, Any]],
        scheduler_context: dict[str, Any],
    ) -> dict[str, Any]:
        del scheduler_context
        target_stiffness = _safe_float(target_property.get("stiffness_proxy"))
        evaluation = dict(failed_candidate.get("evaluation") or {})
        observed = dict(evaluation.get("evaluated_property") or {})
        observed_stiffness = _safe_float(observed.get("stiffness_proxy"))
        dominant_failure = dominant_failure_causes[0]["failure_cause"] if dominant_failure_causes else ""
        preferred_intervention = promising_interventions[0]["intervention_type"] if promising_interventions else "exploitation"

        if dominant_failure == "stiffness_insufficient" or observed_stiffness < target_stiffness:
            direction = "increase_connectivity_or_load_paths"
        elif dominant_failure == "density_mismatch":
            direction = "tighten_density_control"
        else:
            direction = "local_refinement_with_diversity"

        return {
            "preferred_direction": direction,
            "preferred_intervention": preferred_intervention,
            "target_stiffness": target_stiffness,
            "observed_stiffness": observed_stiffness,
            "rationale": f"Dominant failure is {dominant_failure or 'unknown'}; use {preferred_intervention}.",
        }

    def _confidence(self, snapshot: dict[str, Any], evidence_citations: list[str]) -> float:
        total = _safe_float(snapshot.get("total_evidence") or snapshot.get("total_samples"))
        citation_factor = min(len(evidence_citations) / 10.0, 1.0)
        total_factor = min(total / 30.0, 1.0)
        return min(0.9, 0.35 + 0.3 * citation_factor + 0.25 * total_factor)
