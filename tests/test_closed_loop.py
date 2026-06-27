from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
import json
import os
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_api import AgentDatagenAPI
from src.AgentExplorer import AgentExplorer
from src.api import (
    export_inverse_designer_dataset,
    kb_get_group_statistics,
    kb_get_run_provenance,
    kb_get_sample_evidence,
    kb_query_samples,
    refine_agent_knowledge,
)
from src.closed_loop_contracts import BatchProposal, DatagenConfig, DesignSearchParameters, FEMResult, MetaCandidate, Observation
from src.DatagenFEMEvaluator import DatagenFEMEvaluator
from src.ExperimentStore import RawExperimentStore
from src.InverseDesigner import InverseDesigner
from src.KnowledgeRefiner import AgentKnowledgeInterpreter, KnowledgeRefiner
from src.KnowledgeBase import KnowledgeBase
from src.Scheduler import StructureDiscoverySystem
from src.TrainingDataset import TrainingDatasetExporter


class ClosedLoopSystemTests(unittest.TestCase):
    def setUp(self):
        AgentExplorer._global_llm_disable_reason = "tests_disable_network"
        AgentKnowledgeInterpreter._global_llm_disable_reason = "tests_disable_network"

    def test_knowledge_base_roundtrip_and_queries(self):
        with tempfile.TemporaryDirectory(prefix="truss_kb_") as tmp_dir:
            kb = KnowledgeBase(Path(tmp_dir) / "kb.sqlite")
            try:
                sample = self._make_sample(
                    structure_id="s1",
                    target_property={"stiffness_proxy": 10.0, "density_proxy": 0.1},
                    evaluated_property={"stiffness_proxy": 10.5, "density_proxy": 0.11},
                    label="near_miss",
                )
                kb.add_sample(sample)

                self.assertEqual(kb.get_new_sample_count(), 1)
                self.assertEqual(len(kb.query_top_near_miss(sample.target_property, 1)), 1)
                self.assertEqual(kb.dataset_statistics()["near_miss"], 1)

                exported = kb.export_training_dataset(mark_used=True)
                self.assertEqual(len(exported), 1)
                self.assertEqual(kb.get_new_sample_count(), 0)
            finally:
                kb.close()

    def test_raw_experiment_store_projects_dataset_and_knowledge_views(self):
        with tempfile.TemporaryDirectory(prefix="truss_raw_projection_") as tmp_dir:
            store = RawExperimentStore(tmp_dir)
            observation = Observation(
                observation_id="obs_001",
                iteration=1,
                meta={
                    "suggestion_id": "m1",
                    "group": "P222",
                    "symmetry": "P222",
                    "basic_unit_type": "edge_face_center_19node",
                    "unit_cell_type": "symmetry_expanded_truss",
                    "topology_type": "sparse_truss",
                    "connectivity_pattern": "higher_diagonal_density",
                    "max_bars": 12,
                    "rho_target": 0.1,
                    "density_range": [0.08, 0.12],
                    "sampling_strategy": "random_discrete",
                    "tags": ["agent_suggestion", "repair"],
                },
                structure={"structure_id": "s1", "symmetry": "P222"},
                property={"stiffness_proxy": 9.5, "density_proxy": 0.11},
                error={"stiffness_proxy": 0.05, "density_proxy": 0.1},
                label="near_miss",
                fem_status="success",
                geometry_status="valid",
                provenance={
                    "source": "agent_exploration",
                    "hypothesis": "Increase diagonal load paths.",
                    "candidate": {"candidate_id": "c1", "strategy": "repair", "confidence": 0.7},
                    "fidelity": "proxy",
                },
                raw_metrics={"node_count": 19, "edge_count": 12},
            )
            store.append(observation)

            dataset_rows = store.project_dataset()
            self.assertEqual(len(dataset_rows), 1)
            self.assertEqual(dataset_rows[0]["property"]["stiffness_proxy"], 9.5)
            self.assertEqual(dataset_rows[0]["validity"]["fem_status"], "success")
            self.assertEqual(dataset_rows[0]["explicit_structure"]["structure_id"], "s1")
            self.assertNotIn("hypothesis", dataset_rows[0])

            knowledge_rows = store.project_knowledge()
            self.assertEqual(len(knowledge_rows), 1)
            self.assertEqual(knowledge_rows[0].hypothesis, "Increase diagonal load paths.")
            self.assertEqual(knowledge_rows[0].intervention_type, "repair")
            self.assertEqual(knowledge_rows[0].meta_summary["max_bars"], 12)

    def test_knowledge_base_stores_reasoning_evidence(self):
        with tempfile.TemporaryDirectory(prefix="truss_kb_evidence_") as tmp_dir:
            store = RawExperimentStore(Path(tmp_dir) / "raw")
            kb = KnowledgeBase(Path(tmp_dir) / "kb.sqlite")
            try:
                observation = Observation(
                    observation_id="obs_001",
                    iteration=1,
                    meta={"suggestion_id": "m1", "group": "P222", "max_bars": 10},
                    structure={"structure_id": "s1"},
                    property={"stiffness_proxy": 8.0},
                    error={"stiffness_proxy": 0.4},
                    label="failure",
                    fem_status="success",
                    geometry_status="valid",
                    provenance={"hypothesis": "Bars are insufficient.", "candidate": {"strategy": "counterfactual"}},
                )
                evidence = store.project_knowledge_from([observation])
                kb.add_knowledge_evidences(evidence)

                rows = kb.list_knowledge_evidence()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].label, "failure")
                self.assertEqual(rows[0].hypothesis, "Bars are insufficient.")
                self.assertEqual(kb.get_new_evidence_count(), 1)
                kb.mark_evidence_used_for_training()
                self.assertEqual(kb.get_new_evidence_count(), 0)
            finally:
                kb.close()

    def test_agent_knowledge_interpreter_builds_reasoned_snapshot(self):
        statistical_snapshot = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "total_evidence": 3,
            "mechanism_patterns": [
                {
                    "symmetry": "P222",
                    "topology_type": "sparse_truss",
                    "connectivity_pattern": "higher_diagonal_density",
                    "max_bars": 12,
                    "density_range": [0.08, 0.12],
                    "num_samples": 3,
                    "success_rate": 0.33,
                    "useful_rate": 0.67,
                    "avg_property_error": 0.18,
                    "example_evidence_ids": ["ke_1", "ke_2"],
                }
            ],
            "failure_patterns": [
                {
                    "failure_reason": "stiffness_insufficient",
                    "count": 2,
                    "groups": {"P222": 2},
                    "example_evidence_ids": ["ke_3"],
                }
            ],
            "intervention_effects": [
                {
                    "intervention_type": "repair",
                    "count": 3,
                    "useful_rate": 0.67,
                    "avg_error": 0.2,
                    "example_evidence_ids": ["ke_1"],
                }
            ],
            "bad_regions": [],
            "useful_exemplars": {},
            "group_stats": [],
            "config_pattern_stats": [],
            "exemplar_samples": {},
        }

        interpreter = AgentKnowledgeInterpreter()
        reasoned = interpreter.build_reasoned_snapshot(
            statistical_snapshot,
            target_property={"stiffness_proxy": 10.0, "density_proxy": 0.1},
            failed_candidate={
                "evaluation": {
                    "label": "failure",
                    "evaluated_property": {"stiffness_proxy": 6.0, "density_proxy": 0.1},
                }
            },
        )

        self.assertIn("statistical", reasoned)
        self.assertIn("interpretation", reasoned)
        self.assertIn("mechanism_patterns", reasoned)
        self.assertEqual(reasoned["statistical"]["total_evidence"], 3)
        self.assertGreaterEqual(len(reasoned["interpretation"]["mechanism_explanations"]), 1)
        self.assertEqual(
            reasoned["interpretation"]["target_specific_strategy_prior"]["preferred_direction"],
            "increase_connectivity_or_load_paths",
        )
        self.assertIn("ke_1", reasoned["interpretation"]["evidence_citations"])

    def test_two_agent_roles_share_configured_local_endpoint(self):
        with patch.dict(
            os.environ,
            {
                "AGENT_EXPLORER_BASE_URL": "",
                "KNOWLEDGE_INTERPRETER_BASE_URL": "",
                "AGENT_EXPLORER_API_KEY": "",
                "KNOWLEDGE_INTERPRETER_API_KEY": "",
            },
            clear=False,
        ):
            explorer = AgentExplorer()
            interpreter = AgentKnowledgeInterpreter()

            self.assertEqual(explorer.base_url, "http://127.0.0.1:17777/v1")
            self.assertEqual(interpreter.base_url, "http://127.0.0.1:17777/v1")
            self.assertEqual(explorer.api_key, "e339901aa965318c253573770ae65bd0cfce8b93ca057ddc79725f80186382db")
            self.assertEqual(interpreter.api_key, "e339901aa965318c253573770ae65bd0cfce8b93ca057ddc79725f80186382db")

    def test_agent_knowledge_interpreter_accepts_llm_interpretation_payload(self):
        AgentKnowledgeInterpreter._global_llm_disable_reason = ""
        interpreter = AgentKnowledgeInterpreter(enable_llm=True)
        statistical_snapshot = {
            "total_evidence": 2,
            "mechanism_patterns": [],
            "failure_patterns": [],
            "intervention_effects": [],
            "hypothesis_status": [],
            "bad_regions": [],
            "useful_exemplars": {},
        }
        with patch.object(
            interpreter,
            "_request_llm_payload",
            return_value={
                "mechanism_explanations": [
                    {
                        "mechanism": "diagonal_load_path_support",
                        "summary": "Higher diagonal density is correlated with better stiffness.",
                        "example_evidence_ids": ["ke_1"],
                    }
                ],
                "dominant_failure_causes": [
                    {
                        "failure_cause": "stiffness_insufficient_due_to_sparse_diagonals",
                        "example_evidence_ids": ["ke_2"],
                    }
                ],
                "promising_intervention_families": [
                    {"intervention_type": "repair", "example_evidence_ids": ["ke_1"]}
                ],
                "rejected_intervention_families": [],
                "target_specific_strategy_prior": {
                    "preferred_direction": "increase_connectivity_or_load_paths",
                    "preferred_intervention": "repair",
                },
                "causal_tests_to_run": [
                    {"test_type": "single_variable_diagonal_density", "example_evidence_ids": ["ke_1", "ke_2"]}
                ],
                "evidence_citations": ["ke_1", "ke_2"],
                "confidence": 0.72,
            },
        ):
            interpretation = interpreter.interpret(
                statistical_snapshot=statistical_snapshot,
                target_property={"stiffness_proxy": 10.0},
                failed_candidate={"evaluation": {"label": "failure"}},
            )

        self.assertEqual(interpretation["interpreter"], "llm_agent_knowledge_interpreter_v1")
        self.assertEqual(interpretation["confidence"], 0.72)
        self.assertEqual(
            interpretation["target_specific_strategy_prior"]["preferred_intervention"],
            "repair",
        )
        self.assertIn("ke_2", interpretation["evidence_citations"])

    def test_closed_loop_can_exploit_existing_explicit_structure_success(self):
        with tempfile.TemporaryDirectory(prefix="truss_closed_loop_success_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            kb_path = Path(tmp_dir) / "kb.sqlite"
            kb = KnowledgeBase(kb_path)
            try:
                evaluator = DatagenFEMEvaluator(workspace_root=workspace)
                structures = evaluator.datagen(DatagenConfig(num_samples=1, workers=1, batch=1, print_every=1))
                results = evaluator.fem_evaluate(structures)
                samples = evaluator.collect_samples(
                    structures,
                    results,
                    target_property=results[0].evaluated_property,
                    datagen_config=DatagenConfig(num_samples=1, workers=1, batch=1, print_every=1),
                )
                samples[0].label = "success"
                samples[0].explicit_structure = {
                    "structure_id": samples[0].structure_id,
                    "structure_path": samples[0].structure_path,
                    "symmetry": samples[0].symmetry,
                    "source": "kb_seed_explicit_structure",
                }
                kb.add_sample(samples[0])

                system = StructureDiscoverySystem(
                    knowledge_base=kb,
                    inverse_designer=InverseDesigner(kb),
                    agent_explorer=AgentExplorer(),
                    evaluator=evaluator,
                )
                loop_result = system.run(target_property=results[0].evaluated_property, max_iterations=2)
                self.assertTrue(loop_result.success)
                self.assertEqual(loop_result.iterations, 0)
                self.assertEqual(len(loop_result.discovered_samples), 1)
                self.assertEqual(loop_result.task_status, "succeeded")
                self.assertIsNotNone(loop_result.experiment_paths)
            finally:
                kb.close()

    def test_closed_loop_retries_inverse_designer_after_finetune(self):
        class FinetuneAwareInverseDesigner:
            def __init__(self):
                self.training_examples = []
                self.finetune_calls = 0
                self.sample_structure_calls = 0

            def sample_structure(self, target_property):
                self.sample_structure_calls += 1
                if not self.training_examples:
                    return None
                return dict(self.training_examples[-1]["explicit_structure"])

            def propose(self, target_property):  # pragma: no cover - must remain unused by the new scheduler path.
                raise AssertionError("legacy propose() should not be called")

            def finetune(self, new_samples):
                self.finetune_calls += 1
                for sample in new_samples:
                    if sample.get("property") and sample.get("explicit_structure"):
                        self.training_examples.append(sample)

        class OneShotAgentExplorer:
            def propose_batch(
                self,
                target_property,
                failed_candidate,
                fem_result,
                context,
                datagen_schema,
                batch_size=1,
            ):
                meta = DatagenConfig(
                    suggestion_id="agent_meta_001",
                    source="agent_exploration",
                    target_property=dict(target_property),
                    expected_property=dict(target_property),
                    group="P222",
                    symmetry="P222",
                    num_samples=1,
                    hypothesis="Generate one explicit structure for inverse-designer finetuning.",
                    reason="Test fixture creates a successful augmentation sample.",
                    exploration_strategy="exploitation",
                )
                candidate = MetaCandidate(
                    candidate_id="agent_candidate_001",
                    meta=meta,
                    source_backend="agent_designer",
                    strategy="exploitation",
                    score=1.0,
                    confidence=1.0,
                    predicted_property=dict(target_property),
                )
                return BatchProposal(
                    proposal_id="agent_batch_001",
                    source="agent_designer",
                    target_property=dict(target_property),
                    candidates=[candidate],
                    hypothesis=meta.hypothesis,
                    rationale=meta.reason,
                    strategy_counts={"exploitation": 1},
                )

        class ExplicitSuccessEvaluator:
            target_property = {"stiffness_proxy": 1.0, "density_proxy": 0.2}

            def datagen_schema(self):
                return {"default_group": "P222"}

            def datagen(self, config):
                return [
                    {
                        "structure_id": "agent_generated_success",
                        "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        "edges": [[0, 1]],
                        "symmetry": "P222",
                        "datagen_config": config.to_dict(),
                    }
                ]

            def fem_evaluate(self, structures):
                return [
                    FEMResult(
                        structure_id=structures[0]["structure_id"],
                        evaluated_property=dict(self.target_property),
                        fem_status="success",
                        geometry_status="valid",
                        raw_metrics={"node_count": 2, "edge_count": 1},
                    )
                ]

            def collect_samples(self, structures, fem_results, target_property, datagen_config):
                sample = self._sample_from_structure(
                    structure=structures[0],
                    evaluated_property=fem_results[0].evaluated_property,
                    target_property=target_property,
                    label="success",
                    source="agent_exploration",
                )
                sample.parameter_config = datagen_config.to_dict()
                sample.metadata = {
                    "datagen_config": datagen_config.to_dict(),
                    "fidelity": "proxy",
                    "raw_metrics": dict(fem_results[0].raw_metrics),
                }
                return [sample]

            def evaluate_explicit_structure(self, structure, target_property):
                property_error = {
                    key: abs(float(self.target_property[key]) - float(value)) / (abs(float(value)) or 1.0)
                    for key, value in target_property.items()
                }
                return {
                    "structure_id": structure["structure_id"],
                    "evaluated_property": dict(self.target_property),
                    "property_error": property_error,
                    "label": "success" if max(property_error.values()) <= 0.1 else "failure",
                    "fem_status": "success",
                    "geometry_status": "valid",
                    "raw_metrics": {"node_count": 2, "edge_count": 1},
                }

            def collect_explicit_structure_sample(self, structure, evaluation, target_property, source="inverse_designer"):
                return self._sample_from_structure(
                    structure=structure,
                    evaluated_property=evaluation["evaluated_property"],
                    target_property=target_property,
                    label=evaluation["label"],
                    source=source,
                )

            @staticmethod
            def _sample_from_structure(structure, evaluated_property, target_property, label, source):
                from src.closed_loop_contracts import KnowledgeSample

                property_error = {
                    key: abs(float(evaluated_property.get(key, 0.0)) - float(value)) / (abs(float(value)) or 1.0)
                    for key, value in target_property.items()
                }
                return KnowledgeSample(
                    structure_id=structure["structure_id"],
                    structure_path=str(structure.get("structure_path") or ""),
                    unit_cell_type="symmetry_expanded_truss",
                    basic_unit_type="edge_face_center_19node",
                    topology_type="sparse_truss",
                    symmetry=str(structure.get("symmetry") or "P222"),
                    connectivity_pattern="default",
                    parameter_config={"max_bars": 10, "rho_target": 0.2},
                    target_property=dict(target_property),
                    evaluated_property=dict(evaluated_property),
                    property_error=property_error,
                    fem_status="success",
                    geometry_status="valid",
                    label=label,
                    source=source,
                    timestamp="2026-01-01T00:00:00+00:00",
                    metadata={},
                    explicit_structure=dict(structure),
                )

        with tempfile.TemporaryDirectory(prefix="truss_closed_loop_retry_inverse_") as tmp_dir:
            kb = KnowledgeBase(Path(tmp_dir) / "kb.sqlite")
            inverse = FinetuneAwareInverseDesigner()
            try:
                system = StructureDiscoverySystem(
                    knowledge_base=kb,
                    inverse_designer=inverse,
                    agent_explorer=OneShotAgentExplorer(),
                    evaluator=ExplicitSuccessEvaluator(),
                    retrain_trigger=1,
                    workspace_root=Path(tmp_dir) / "workspace",
                    agent_batch_size=1,
                    experiment_budget=1,
                )

                result = system.run(
                    target_property=ExplicitSuccessEvaluator.target_property,
                    max_iterations=1,
                )

                self.assertFalse(result.success)
                self.assertEqual(result.iterations, 1)
                self.assertEqual(inverse.finetune_calls, 1)
                self.assertGreaterEqual(inverse.sample_structure_calls, 2)
                inverse_events = [event for event in result.events if event.stage == "inverse_designer"]
                self.assertEqual(inverse_events[0].payload["phase"], "initial")
                self.assertEqual(inverse_events[-1].payload["phase"], "post_finetune")
                self.assertTrue(any(event.status == "schedule_targets_missing_candidates" for event in inverse_events))
                self.assertGreaterEqual(len(kb.list_knowledge_evidence()), 1)
            finally:
                kb.close()

    def test_closed_loop_from_empty_kb_generates_samples(self):
        with tempfile.TemporaryDirectory(prefix="truss_closed_loop_empty_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            kb_path = Path(tmp_dir) / "kb.sqlite"
            kb = KnowledgeBase(kb_path)
            try:
                system = StructureDiscoverySystem(
                    knowledge_base=kb,
                    inverse_designer=InverseDesigner(kb),
                    agent_explorer=AgentExplorer(),
                    evaluator=DatagenFEMEvaluator(workspace_root=workspace),
                    retrain_trigger=2,
                    log_path=Path(tmp_dir) / "loop.jsonl",
                )
                loop_result = system.run(
                    target_property={"stiffness_proxy": 30.0, "density_proxy": 0.1},
                    max_iterations=1,
                )
                self.assertGreaterEqual(len(loop_result.logs), 2)
                self.assertGreaterEqual(len(loop_result.events), 2)
                self.assertGreaterEqual(kb.dataset_statistics().get("total", 0), 1)
                self.assertTrue((Path(tmp_dir) / "loop.jsonl").exists())
                self.assertTrue(Path(loop_result.experiment_paths.events_dir).exists())
                self.assertTrue(Path(loop_result.experiment_paths.runs_dir).exists())
                self.assertTrue((Path(loop_result.experiment_paths.root_dir) / "events.jsonl").exists())
                self.assertTrue((Path(loop_result.experiment_paths.root_dir) / "raw_experiments.jsonl").exists())
            finally:
                kb.close()

    def test_agent_api_closed_loop_smoke(self):
        with tempfile.TemporaryDirectory(prefix="truss_agent_closed_loop_") as tmp_dir:
            result = AgentDatagenAPI.run_closed_loop(
                target_property={"stiffness_proxy": 25.0, "density_proxy": 0.1},
                workspace_root=str(Path(tmp_dir) / "workspace"),
                kb_path=str(Path(tmp_dir) / "kb.sqlite"),
                max_iterations=1,
                retrain_trigger=2,
                log_path=str(Path(tmp_dir) / "loop.jsonl"),
            )
            self.assertIn("success", result)
            self.assertIn("logs", result)
            self.assertIn("events", result)
            self.assertIn("task_status", result)
            self.assertIn("experiment_paths", result)
            self.assertGreaterEqual(len(result["logs"]), 2)
            self.assertTrue((Path(tmp_dir) / "kb.sqlite").exists())

    def test_bootstrap_populates_normalized_kb_tables_and_queries(self):
        with tempfile.TemporaryDirectory(prefix="truss_bootstrap_kb_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            output_dir = Path(tmp_dir) / "output"
            kb_path = Path(tmp_dir) / "seed_kb.sqlite"
            evaluator = DatagenFEMEvaluator(workspace_root=workspace)

            result = evaluator.bootstrap_dataset_and_kb(
                datagen_configs=[
                    DatagenConfig(
                        suggestion_id="bootstrap_001_P222",
                        source="bootstrap_seed",
                        group="P222",
                        symmetry="P222",
                        num_samples=1,
                        workers=1,
                        batch=1,
                        print_every=1,
                        exploration_strategy="bootstrap_seed",
                    )
                ],
                kb_path=kb_path,
                output_dir=output_dir,
            )

            self.assertEqual(result.total_samples, 1)
            conn = sqlite3.connect(kb_path)
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                tables = {row[0] for row in cur.fetchall()}
                self.assertTrue({"runs", "samples", "datagen_configs", "artifacts"}.issubset(tables))

                cur.execute("SELECT COUNT(*) FROM runs")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute("SELECT COUNT(*) FROM datagen_configs")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute("SELECT COUNT(*) FROM artifacts")
                self.assertGreaterEqual(cur.fetchone()[0], 4)
            finally:
                conn.close()

            kb = KnowledgeBase(kb_path)
            try:
                stats = kb.get_group_statistics()
                self.assertIn("P222", stats)
                sample = kb.get_similar_property_samples({"stiffness_proxy": 0.0, "density_proxy": 0.0}, top_k=1)[0]
                evidence = kb.get_sample_evidence(sample.structure_id)
                self.assertIsNotNone(evidence)
                self.assertEqual(evidence["sample"]["run_id"], "bootstrap_001_P222")
                provenance = kb.get_run_provenance("bootstrap_001_P222")
                self.assertIsNotNone(provenance)
                self.assertEqual(len(provenance["datagen_configs"]), 1)
                self.assertGreaterEqual(len(provenance["artifacts"]), 4)
            finally:
                kb.close()

    def test_api_and_agent_kb_query_interfaces(self):
        with tempfile.TemporaryDirectory(prefix="truss_kb_api_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            output_dir = Path(tmp_dir) / "output"
            kb_path = Path(tmp_dir) / "seed_kb.sqlite"
            evaluator = DatagenFEMEvaluator(workspace_root=workspace)

            evaluator.bootstrap_dataset_and_kb(
                datagen_configs=[
                    DatagenConfig(
                        suggestion_id="bootstrap_001_P222",
                        source="bootstrap_seed",
                        group="P222",
                        symmetry="P222",
                        num_samples=1,
                        workers=1,
                        batch=1,
                        print_every=1,
                        exploration_strategy="bootstrap_seed",
                    )
                ],
                kb_path=kb_path,
                output_dir=output_dir,
            )

            stats = kb_get_group_statistics(str(kb_path))
            self.assertIn("P222", stats)

            samples = kb_query_samples(
                kb_path=str(kb_path),
                query_type="success",
                top_k=1,
            )
            self.assertEqual(len(samples), 1)
            sample_id = samples[0]["structure_id"]

            evidence = kb_get_sample_evidence(str(kb_path), sample_id)
            self.assertIsNotNone(evidence)
            self.assertEqual(evidence["sample"]["structure_id"], sample_id)

            provenance = kb_get_run_provenance(str(kb_path), "bootstrap_001_P222")
            self.assertIsNotNone(provenance)
            self.assertEqual(provenance["run"]["run_id"], "bootstrap_001_P222")

            agent_stats = AgentDatagenAPI.kb_group_statistics(str(kb_path))
            self.assertEqual(agent_stats["P222"]["total"], stats["P222"]["total"])
            agent_evidence = AgentDatagenAPI.kb_sample_evidence(str(kb_path), sample_id)
            self.assertEqual(agent_evidence["sample"]["structure_id"], sample_id)

    def test_training_export_and_knowledge_refinement_alignment(self):
        with tempfile.TemporaryDirectory(prefix="truss_refine_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            output_dir = Path(tmp_dir) / "output"
            kb_path = Path(tmp_dir) / "seed_kb.sqlite"
            evaluator = DatagenFEMEvaluator(workspace_root=workspace)

            evaluator.bootstrap_dataset_and_kb(
                datagen_configs=[
                    DatagenConfig(
                        suggestion_id="bootstrap_001_P222",
                        source="bootstrap_seed",
                        group="P222",
                        symmetry="P222",
                        num_samples=2,
                        workers=1,
                        batch=1,
                        print_every=1,
                        exploration_strategy="bootstrap_seed",
                    )
                ],
                kb_path=kb_path,
                output_dir=output_dir,
            )

            exported = export_inverse_designer_dataset(str(kb_path), mark_used=False)
            self.assertGreaterEqual(len(exported), 1)
            self.assertIn("structure", exported[0])
            self.assertIn("property", exported[0])
            self.assertNotIn("metadata", exported[0])

            refined = refine_agent_knowledge(str(kb_path))
            self.assertIn("group_stats", refined)
            self.assertIn("config_pattern_stats", refined)
            self.assertIn("failure_patterns", refined)
            self.assertIn("exemplar_samples", refined)

            kb = KnowledgeBase(kb_path)
            try:
                exporter = TrainingDatasetExporter()
                compact = exporter.export_from_knowledge_base(kb, mark_used=False)
                self.assertEqual(len(compact), len(exported))

                refiner = KnowledgeRefiner()
                snapshot = refiner.build_from_knowledge_base(kb)
                self.assertGreaterEqual(len(snapshot["group_stats"]), 1)
            finally:
                kb.close()

    def test_scheduler_writes_agent_knowledge_snapshot(self):
        with tempfile.TemporaryDirectory(prefix="truss_scheduler_knowledge_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            kb_path = Path(tmp_dir) / "kb.sqlite"
            kb = KnowledgeBase(kb_path)
            try:
                system = StructureDiscoverySystem(
                    knowledge_base=kb,
                    inverse_designer=InverseDesigner(kb),
                    agent_explorer=AgentExplorer(),
                    evaluator=DatagenFEMEvaluator(workspace_root=workspace),
                    retrain_trigger=2,
                    workspace_root=workspace,
                )
                loop_result = system.run(
                    target_property={"stiffness_proxy": 30.0, "density_proxy": 0.1},
                    max_iterations=1,
                )
                knowledge_dir = Path(loop_result.experiment_paths.knowledge_dir)
                snapshots = sorted(knowledge_dir.glob("agent_knowledge_iter_*.json"))
                statistical_snapshots = sorted(knowledge_dir.glob("statistical_knowledge_iter_*.json"))
                self.assertGreaterEqual(len(snapshots), 1)
                self.assertGreaterEqual(len(statistical_snapshots), 1)
                reasoned = json.loads(snapshots[0].read_text(encoding="utf-8"))
                statistical = json.loads(statistical_snapshots[0].read_text(encoding="utf-8"))
                self.assertIn("statistical", reasoned)
                self.assertIn("interpretation", reasoned)
                self.assertEqual(reasoned["statistical"]["generated_at"], statistical["generated_at"])
                self.assertGreaterEqual(len(kb.list_knowledge_evidence()), 1)
            finally:
                kb.close()

    def test_closed_loop_logs_are_concise_and_structured(self):
        with tempfile.TemporaryDirectory(prefix="truss_scheduler_logs_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            kb_path = Path(tmp_dir) / "kb.sqlite"
            kb = KnowledgeBase(kb_path)
            try:
                system = StructureDiscoverySystem(
                    knowledge_base=kb,
                    inverse_designer=InverseDesigner(kb),
                    agent_explorer=AgentExplorer(),
                    evaluator=DatagenFEMEvaluator(workspace_root=workspace),
                    retrain_trigger=2,
                    workspace_root=workspace,
                )
                loop_result = system.run(
                    target_property={"stiffness_proxy": 0.3, "density_proxy": 0.1},
                    max_iterations=1,
                )

                stages = {event.stage for event in loop_result.events}
                self.assertIn("task", stages)
                self.assertIn("inverse_designer", stages)
                self.assertIn("dataset_state", stages)
                self.assertIn("agent_explorer", stages)
                self.assertIn("evaluation", stages)
                self.assertIn("knowledge_update", stages)

                task_event = next(event for event in loop_result.events if event.stage == "task")
                self.assertIn("dataset_distribution", task_event.payload)
                self.assertIn("target_property", task_event.payload)

                inverse_event = next(event for event in loop_result.events if event.stage == "inverse_designer")
                self.assertIn("satisfies_target", inverse_event.payload)

                agent_event = next(event for event in loop_result.events if event.stage == "agent_explorer")
                self.assertIn("hypothesis", agent_event.payload)
                self.assertIn("structure_parameters", agent_event.payload)

                evaluation_event = next(event for event in loop_result.events if event.stage == "evaluation")
                self.assertIn("best_result", evaluation_event.payload)
                self.assertIn("satisfies_target", evaluation_event.payload)

                update_event = next(event for event in loop_result.events if event.stage == "knowledge_update")
                self.assertIn("dataset_distribution", update_event.payload)
                self.assertIn("knowledge_snapshot_path", update_event.payload)
                self.assertIn("added_knowledge_evidence", update_event.payload)

                dataset_event = next(event for event in loop_result.events if event.stage == "dataset_state")
                self.assertIn("statistical_snapshot_path", dataset_event.payload)

                summary_path = Path(loop_result.experiment_paths.root_dir) / "run_summary.md"
                self.assertTrue(summary_path.exists())
                summary_text = summary_path.read_text(encoding="utf-8")
                self.assertIn("# Run Summary", summary_text)
                self.assertIn("## Overview", summary_text)
                self.assertIn("## InverseDesigner", summary_text)
                self.assertIn("Agent reasoning:", summary_text)
                self.assertIn("Datagen + evaluation:", summary_text)
                self.assertIn("## Final Outcome", summary_text)
            finally:
                kb.close()

    def test_inverse_designer_trains_and_retrieves_explicit_structures(self):
        with tempfile.TemporaryDirectory(prefix="truss_inverse_") as tmp_dir:
            kb_path = Path(tmp_dir) / "kb.sqlite"
            kb = KnowledgeBase(kb_path)
            try:
                success_sample = self._make_sample(
                    structure_id="s_success",
                    target_property={"stiffness_proxy": 10.0, "density_proxy": 0.10},
                    evaluated_property={"stiffness_proxy": 10.5, "density_proxy": 0.11},
                    label="success",
                )
                success_sample.parameter_config.update(
                    {
                        "max_bars": 10,
                        "rho_target": 0.10,
                        "density_range": [0.08, 0.12],
                        "sampling_strategy": "random_discrete",
                    }
                )
                success_sample.explicit_structure = {
                    "structure_id": "s_success",
                    "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "edges": [[0, 1]],
                }

                valid_failure = self._make_sample(
                    structure_id="s_failure",
                    target_property={"stiffness_proxy": 12.0, "density_proxy": 0.09},
                    evaluated_property={"stiffness_proxy": 11.0, "density_proxy": 0.09},
                    label="failure",
                )
                valid_failure.parameter_config.update(
                    {
                        "max_bars": 12,
                        "rho_target": 0.09,
                        "density_range": [0.07, 0.11],
                        "sampling_strategy": "random_discrete",
                    }
                )
                valid_failure.explicit_structure = {
                    "structure_id": "s_failure",
                    "coordinates": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    "edges": [[0, 1]],
                }

                invalid_sample = self._make_sample(
                    structure_id="s_invalid",
                    target_property={"stiffness_proxy": 14.0, "density_proxy": 0.08},
                    evaluated_property={"stiffness_proxy": 14.0, "density_proxy": 0.08},
                    label="failure",
                )
                invalid_sample.geometry_status = "invalid"

                kb.add_samples([success_sample, valid_failure, invalid_sample])

                exporter = TrainingDatasetExporter()
                training_data = exporter.export_from_knowledge_base(kb, mark_used=False)

                inverse = InverseDesigner(kb)
                inverse.train(training_data)

                self.assertEqual(len(inverse.training_examples), 2)
                self.assertIn("explicit_structure", training_data[0])
                self.assertIn("training_target", training_data[0])

                retrieved = inverse.sample_structure({"stiffness_proxy": 11.2, "density_proxy": 0.095})
                self.assertIsNotNone(retrieved)
                self.assertIn("structure_id", retrieved)
                self.assertIn("retrieved_property", retrieved)
                self.assertIn("retrieval_distance", retrieved)

                inverse.finetune(
                    [
                        {
                            "sample_id": "explicit_new",
                            "property": {"stiffness_proxy": 20.0, "density_proxy": 0.05},
                            "explicit_structure": {
                                "structure_id": "explicit_new",
                                "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                "edges": [[0, 1]],
                            },
                            "validity": {"geometry_status": "valid", "fem_status": "success"},
                        }
                    ]
                )
                self.assertEqual(len(inverse.training_examples), 3)
                retrieved_after_finetune = inverse.sample_structure({"stiffness_proxy": 20.0, "density_proxy": 0.05})
                self.assertEqual(retrieved_after_finetune["structure_id"], "explicit_new")
            finally:
                kb.close()

    def test_inverse_designer_retrieves_explicit_structure(self):
        with tempfile.TemporaryDirectory(prefix="truss_inverse_structure_") as tmp_dir:
            kb = KnowledgeBase(Path(tmp_dir) / "kb.sqlite")
            try:
                inverse = InverseDesigner(kb)
                explicit_structure = {
                    "structure_id": "explicit_001",
                    "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "edges": [[0, 1]],
                }
                inverse.train(
                    [
                        {
                            "sample_id": "explicit_001",
                            "property": {"stiffness_proxy": 0.75, "density_proxy": 1.0},
                            "explicit_structure": explicit_structure,
                            "validity": {"geometry_status": "valid", "fem_status": "success"},
                        }
                    ]
                )

                retrieved = inverse.sample_structure({"stiffness_proxy": 0.74, "density_proxy": 1.0})

                self.assertIsNotNone(retrieved)
                self.assertEqual(retrieved["structure_id"], "explicit_001")
                self.assertIn("coordinates", retrieved)
                self.assertIn("retrieval_distance", retrieved)
            finally:
                kb.close()

    def test_closed_loop_returns_when_retrieved_structure_satisfies_target(self):
        with tempfile.TemporaryDirectory(prefix="truss_inverse_structure_loop_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            kb = KnowledgeBase(Path(tmp_dir) / "kb.sqlite")
            try:
                evaluator = DatagenFEMEvaluator(workspace_root=workspace)
                explicit_structure = {
                    "structure_id": "explicit_success",
                    "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "edges": [[0, 1]],
                }
                target_property = evaluator.evaluate_explicit_structure(
                    explicit_structure,
                    {"stiffness_proxy": 0.0, "density_proxy": 0.0},
                )["evaluated_property"]
                inverse = InverseDesigner(kb)
                inverse.train(
                    [
                        {
                            "sample_id": "explicit_success",
                            "property": target_property,
                            "explicit_structure": explicit_structure,
                            "validity": {"geometry_status": "valid", "fem_status": "success"},
                        }
                    ]
                )
                system = StructureDiscoverySystem(
                    knowledge_base=kb,
                    inverse_designer=inverse,
                    agent_explorer=AgentExplorer(),
                    evaluator=evaluator,
                    workspace_root=workspace,
                )

                loop_result = system.run(target_property=target_property, max_iterations=1)

                self.assertTrue(loop_result.success)
                self.assertEqual(loop_result.iterations, 0)
                stages = [event.stage for event in loop_result.events]
                self.assertIn("inverse_designer", stages)
                self.assertNotIn("agent_explorer", stages)
                self.assertEqual(loop_result.discovered_samples[0].explicit_structure["structure_id"], "explicit_success")
            finally:
                kb.close()

    def test_agent_explorer_proposes_structured_meta_batch(self):
        explorer = AgentExplorer(enable_llm=False)
        batch = explorer.propose_batch(
            target_property={"stiffness_proxy": 0.5, "density_proxy": 0.1},
            failed_candidate=None,
            fem_result=None,
            context={},
            datagen_schema={"default_group": "P222"},
            batch_size=12,
        )
        self.assertEqual(batch.source, "agent_designer")
        self.assertGreaterEqual(len(batch.candidates), 1)
        self.assertLessEqual(len(batch.candidates), 12)
        strategies = {candidate.strategy for candidate in batch.candidates}
        self.assertIn("exploitation", strategies)
        self.assertTrue(batch.strategy_counts)
        first_meta = batch.candidates[0].meta
        self.assertIsInstance(first_meta.design_search_parameters, DesignSearchParameters)
        self.assertEqual(first_meta.design_search_parameters.group, first_meta.group)
        self.assertEqual(first_meta.design_search_parameters.max_bars, first_meta.max_bars)
        self.assertEqual(first_meta.design_search_parameters.rho_target, first_meta.rho_target)

    def test_agent_explorer_consumes_reasoned_interpretation_prior(self):
        explorer = AgentExplorer(enable_llm=False)
        proposal = explorer.propose(
            target_property={"stiffness_proxy": 10.0, "density_proxy": 0.1},
            failed_candidate=None,
            fem_result={"evaluated_property": {"stiffness_proxy": 4.0}, "label": "failure"},
            context={
                "reasoned_agent_knowledge_snapshot": {
                    "statistical": {},
                    "interpretation": {
                        "target_specific_strategy_prior": {
                            "preferred_direction": "increase_connectivity_or_load_paths",
                            "preferred_intervention": "repair",
                            "rationale": "Evidence suggests diagonal load paths are insufficient.",
                        }
                    }
                }
            },
            datagen_schema={"default_group": "P222"},
        )

        self.assertEqual(proposal.connectivity_pattern, "higher_diagonal_density")
        self.assertEqual(proposal.exploration_strategy, "repair")
        self.assertIn("Reasoned knowledge", proposal.hypothesis)

    def test_scheduler_records_raw_experiment_store(self):
        with tempfile.TemporaryDirectory(prefix="truss_raw_store_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            kb_path = Path(tmp_dir) / "kb.sqlite"
            kb = KnowledgeBase(kb_path)
            try:
                system = StructureDiscoverySystem(
                    knowledge_base=kb,
                    inverse_designer=InverseDesigner(kb),
                    agent_explorer=AgentExplorer(),
                    evaluator=DatagenFEMEvaluator(workspace_root=workspace),
                    retrain_trigger=2,
                    workspace_root=workspace,
                    agent_batch_size=12,
                    experiment_budget=1,
                )
                loop_result = system.run(
                    target_property={"stiffness_proxy": 30.0, "density_proxy": 0.1},
                    max_iterations=1,
                )
                raw_path = Path(loop_result.experiment_paths.root_dir) / "raw_experiments.jsonl"
                self.assertTrue(raw_path.exists())
                raw_count = len(raw_path.read_text(encoding="utf-8").splitlines())
                self.assertGreaterEqual(raw_count, 1)
                self.assertEqual(len(kb.list_knowledge_evidence()), raw_count)
                agent_event = next(event for event in loop_result.events if event.stage == "agent_explorer")
                self.assertIn("batch_design", agent_event.payload)
                self.assertGreaterEqual(agent_event.payload["batch_design"]["agent_candidate_count"], 1)
            finally:
                kb.close()

    def test_agent_explorer_accepts_llm_suggestion_payload(self):
        AgentExplorer._global_llm_disable_reason = ""
        explorer = AgentExplorer(enable_llm=True)
        with patch.object(
            explorer,
            "_request_llm_payload",
            return_value={
                "group": "P222",
                "symmetry": "P222",
                "basic_unit_type": "edge_face_center_19node",
                "unit_cell_type": "symmetry_expanded_truss",
                "topology_type": "sparse_truss",
                "connectivity_pattern": "higher_diagonal_density",
                "max_bars": 12,
                "rho_target": 0.11,
                "density_range": [0.08, 0.12],
                "num_samples": 6,
                "expected_property": {"stiffness_proxy": 0.42, "density_proxy": 0.11},
                "hypothesis": "Denser diagonals should improve stiffness.",
                "reason": "Near-miss evidence favors higher connectivity.",
                "imagination": "A diagonal-rich truss with reinforced face links.",
                "failure_reason_guess": "stiffness_insufficient",
                "exploration_strategy": "near_miss_refinement",
                "confidence": 0.73,
            },
        ), patch.object(explorer, "_resolve_model", return_value="gpt-5.3-codex"):
            proposal = explorer.propose(
                target_property={"stiffness_proxy": 0.5, "density_proxy": 0.1},
                failed_candidate=None,
                fem_result=None,
                context={},
                datagen_schema={"default_group": "P222"},
            )

        self.assertEqual(proposal.source, "agent_exploration_llm")
        self.assertEqual(proposal.connectivity_pattern, "higher_diagonal_density")
        self.assertEqual(proposal.max_bars, 12)
        self.assertEqual(proposal.design_search_parameters.max_bars, 12)
        self.assertEqual(proposal.design_search_parameters.rho_target, 0.11)
        self.assertIn("llm_suggestion", proposal.tags)
        self.assertEqual(proposal.failure_analysis["llm_failure_reason_guess"], "stiffness_insufficient")

    def test_agent_explorer_falls_back_when_llm_request_fails(self):
        AgentExplorer._global_llm_disable_reason = ""
        explorer = AgentExplorer(enable_llm=True)
        with patch.object(explorer, "_request_llm_payload", side_effect=RuntimeError("gateway failure")):
            proposal = explorer.propose(
                target_property={"stiffness_proxy": 0.5, "density_proxy": 0.1},
                failed_candidate=None,
                fem_result=None,
                context={},
                datagen_schema={"default_group": "P222"},
            )

        self.assertEqual(proposal.source, "agent_exploration")
        self.assertIn("llm_fallback", proposal.tags)
        self.assertIn("llm_fallback_reason", proposal.failure_analysis)

    @staticmethod
    def _make_sample(structure_id, target_property, evaluated_property, label):
        from src.closed_loop_contracts import KnowledgeSample

        return KnowledgeSample(
            structure_id=structure_id,
            structure_path="/tmp/fake.txt",
            unit_cell_type="u",
            basic_unit_type="b",
            topology_type="t",
            symmetry="P222",
            connectivity_pattern="default",
            parameter_config={"max_bars": 10},
            target_property=target_property,
            evaluated_property=evaluated_property,
            property_error={"stiffness_proxy": 0.05},
            fem_status="success",
            geometry_status="valid",
            label=label,
            source="manual_seed",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={},
        )


if __name__ == "__main__":
    unittest.main()
