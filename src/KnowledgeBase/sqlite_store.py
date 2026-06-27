from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from ..closed_loop_contracts import KnowledgeEvidence, KnowledgeSample


def _property_distance(target: dict[str, float], observed: dict[str, float]) -> float:
    keys = sorted(set(target) & set(observed))
    if not keys:
        return float("inf")
    total = 0.0
    for key in keys:
        t = float(target[key])
        o = float(observed[key])
        scale = abs(t) if abs(t) > 1e-9 else 1.0
        total += abs(o - t) / scale
    return total / len(keys)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _json_loads(payload: str | None, default: Any) -> Any:
    if not payload:
        return default
    return json.loads(payload)


class KnowledgeBase:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self):
        self.conn.close()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                "group" TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                status TEXT NOT NULL,
                target_sample_count INTEGER NOT NULL DEFAULT 0,
                generated_sample_count INTEGER NOT NULL DEFAULT 0,
                summary_path TEXT,
                generated_data_manifest_path TEXT,
                knowledge_base_seed_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                structure_id TEXT PRIMARY KEY,
                structure_path TEXT NOT NULL,
                unit_cell_type TEXT NOT NULL,
                basic_unit_type TEXT NOT NULL,
                topology_type TEXT NOT NULL,
                symmetry TEXT NOT NULL,
                connectivity_pattern TEXT NOT NULL,
                parameter_config TEXT NOT NULL,
                target_property TEXT NOT NULL,
                evaluated_property TEXT NOT NULL,
                property_error TEXT NOT NULL,
                fem_status TEXT NOT NULL,
                geometry_status TEXT NOT NULL,
                label TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT NOT NULL,
                used_for_training INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS datagen_configs (
                config_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                "group" TEXT NOT NULL,
                suggestion_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                agent_reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL,
                run_id TEXT,
                artifact_type TEXT NOT NULL,
                path TEXT NOT NULL,
                description TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_evidence (
                evidence_id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL,
                meta_id TEXT NOT NULL,
                structure_id TEXT NOT NULL,
                label TEXT NOT NULL,
                source TEXT NOT NULL,
                proposal_group TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                intervention_type TEXT NOT NULL,
                meta_summary TEXT NOT NULL,
                structure_features TEXT NOT NULL,
                property_result TEXT NOT NULL,
                error_to_target TEXT NOT NULL,
                intervention_delta TEXT NOT NULL,
                effect_summary TEXT NOT NULL,
                reasoning_tags TEXT NOT NULL,
                provenance TEXT NOT NULL,
                fidelity TEXT NOT NULL,
                confidence REAL NOT NULL,
                used_for_training INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self.conn.commit()

        self._ensure_table_columns(
            "samples",
            {
                "sample_id": "TEXT",
                "run_id": "TEXT",
                '"group"': "TEXT",
                "sample_type": "TEXT",
                "distance_to_target": "REAL",
                "csv_path": "TEXT",
                "abaqus_txt_path": "TEXT",
                "crystal_txt_path": "TEXT",
                "target_property_json": "TEXT",
                "evaluated_property_json": "TEXT",
                "meta_json": "TEXT",
                "structure_json": "TEXT",
            },
        )
        self._ensure_table_columns(
            "runs",
            {
                "generated_data_manifest_path": "TEXT",
                "knowledge_base_seed_path": "TEXT",
                "updated_at": "TEXT NOT NULL DEFAULT ''",
            },
        )
        self._create_indexes()
        self._backfill_normalized_tables()

    def _create_indexes(self):
        cur = self.conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label)")
        cur.execute('CREATE INDEX IF NOT EXISTS idx_samples_group ON samples("group")')
        cur.execute("CREATE INDEX IF NOT EXISTS idx_samples_run_id ON samples(run_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_sample_id ON artifacts(sample_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_configs_run_id ON datagen_configs(run_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_evidence_label ON knowledge_evidence(label)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_evidence_structure_id ON knowledge_evidence(structure_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_evidence_hypothesis_id ON knowledge_evidence(hypothesis_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_evidence_intervention ON knowledge_evidence(intervention_type)")
        self.conn.commit()

    def _ensure_table_columns(self, table_name: str, columns: dict[str, str]):
        existing = self._table_columns(table_name)
        cur = self.conn.cursor()
        for column_name, column_type in columns.items():
            normalized_name = column_name.strip('"')
            if normalized_name in existing:
                continue
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        self.conn.commit()

    def _table_columns(self, table_name: str) -> set[str]:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        return {str(row[1]) for row in cur.fetchall()}

    def _backfill_normalized_tables(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM samples")
        rows = cur.fetchall()
        if not rows:
            return
        for row in rows:
            self._add_sample(self._row_to_sample(row), commit=False)
        self.conn.commit()

    def add_sample(self, sample: KnowledgeSample):
        self._add_sample(sample, commit=True)

    def add_samples(self, samples: Iterable[KnowledgeSample]):
        for sample in samples:
            self._add_sample(sample, commit=False)
        self.conn.commit()

    def add_knowledge_evidence(self, evidence: KnowledgeEvidence):
        self._add_knowledge_evidence(evidence, commit=True)

    def add_knowledge_evidences(self, evidences: Iterable[KnowledgeEvidence]):
        for evidence in evidences:
            self._add_knowledge_evidence(evidence, commit=False)
        self.conn.commit()

    def _add_knowledge_evidence(self, evidence: KnowledgeEvidence, commit: bool):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO knowledge_evidence (
                evidence_id, observation_id, meta_id, structure_id, label, source, proposal_group,
                parent_id, hypothesis_id, hypothesis, intervention_type, meta_summary, structure_features,
                property_result, error_to_target, intervention_delta, effect_summary, reasoning_tags,
                provenance, fidelity, confidence, used_for_training
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT used_for_training FROM knowledge_evidence WHERE evidence_id = ?), 0)
            )
            """,
            (
                evidence.evidence_id,
                evidence.observation_id,
                evidence.meta_id,
                evidence.structure_id,
                evidence.label,
                evidence.source,
                evidence.proposal_group,
                evidence.parent_id,
                evidence.hypothesis_id,
                evidence.hypothesis,
                evidence.intervention_type,
                _json_dumps(evidence.meta_summary),
                _json_dumps(evidence.structure_features),
                _json_dumps(evidence.property_result),
                _json_dumps(evidence.error_to_target),
                _json_dumps(evidence.intervention_delta),
                _json_dumps(evidence.effect_summary),
                _json_dumps(list(evidence.reasoning_tags)),
                _json_dumps(evidence.provenance),
                evidence.fidelity,
                float(evidence.confidence),
                evidence.evidence_id,
            ),
        )
        if commit:
            self.conn.commit()

    def _add_sample(self, sample: KnowledgeSample, commit: bool):
        metadata = dict(sample.metadata or {})
        artifacts = dict(metadata.get("artifacts") or {})
        datagen_config = dict(metadata.get("datagen_config") or {})
        agent_suggestion = dict(metadata.get("agent_suggestion") or {})
        datagen_result = dict(metadata.get("datagen_result") or {})
        run_metadata = dict(metadata.get("run") or {})

        run_dir = (
            artifacts.get("run_dir")
            or datagen_result.get("run_dir")
            or run_metadata.get("run_dir")
            or datagen_config.get("run_dir")
            or ""
        )
        run_id = (
            run_metadata.get("run_id")
            or datagen_result.get("run_id")
            or self._derive_run_id(run_dir, datagen_config, sample)
        )
        group_name = (
            run_metadata.get("group")
            or datagen_result.get("group")
            or datagen_config.get("group")
            or sample.symmetry
        )
        sample_id = str(metadata.get("sample_id") or sample.structure_id)
        sample_type = str(
            metadata.get("sample_type")
            or (metadata.get("bootstrap") or {}).get("mode")
            or datagen_config.get("exploration_strategy")
            or sample.source
        )
        distance_to_target = _property_distance(sample.target_property, sample.evaluated_property)
        csv_path = artifacts.get("csv_path") or ""
        abaqus_txt_path = artifacts.get("abaqus_txt_path") or ""
        crystal_txt_path = artifacts.get("crystal_txt_path") or sample.structure_path
        config_id = str(datagen_config.get("suggestion_id") or sample_id)
        agent_reason = str(
            datagen_config.get("reason")
            or agent_suggestion.get("reason")
            or metadata.get("reason")
            or ""
        )
        explicit_structure = dict(
            getattr(sample, "explicit_structure", {}) or metadata.get("explicit_structure") or {}
        )

        self._upsert_run(
            run_id=run_id,
            group_name=group_name,
            run_dir=run_dir,
            status=str(run_metadata.get("status") or datagen_result.get("status") or "unknown"),
            target_sample_count=int(
                run_metadata.get("target_sample_count")
                or (datagen_result.get("summary") or {}).get("samples_target")
                or 0
            ),
            generated_sample_count=int(
                run_metadata.get("generated_sample_count")
                or (datagen_result.get("summary") or {}).get("crystal_processed")
                or 0
            ),
            summary_path=str(run_metadata.get("summary_path") or datagen_result.get("summary_path") or ""),
            generated_data_manifest_path=str(datagen_result.get("generated_data_manifest_path") or ""),
            knowledge_base_seed_path=str(datagen_result.get("knowledge_base_seed_path") or ""),
            created_at=str(run_metadata.get("created_at") or sample.timestamp),
            updated_at=sample.timestamp,
        )

        self._upsert_datagen_config(
            config_id=config_id,
            run_id=run_id,
            group_name=group_name,
            suggestion_id=str(datagen_config.get("suggestion_id") or config_id),
            config_json=_json_dumps(datagen_config),
            agent_reason=agent_reason,
            created_at=sample.timestamp,
        )

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO samples (
                structure_id, sample_id, run_id, "group", label, sample_type, distance_to_target,
                structure_path, csv_path, abaqus_txt_path, crystal_txt_path,
                unit_cell_type, basic_unit_type, topology_type, symmetry, connectivity_pattern,
                parameter_config, target_property, target_property_json, evaluated_property,
                evaluated_property_json, property_error, fem_status, geometry_status,
                source, timestamp, metadata, meta_json, structure_json, used_for_training
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT used_for_training FROM samples WHERE structure_id = ?), 0)
            )
            """,
            (
                sample.structure_id,
                sample_id,
                run_id,
                group_name,
                sample.label,
                sample_type,
                distance_to_target,
                sample.structure_path,
                csv_path,
                abaqus_txt_path,
                crystal_txt_path,
                sample.unit_cell_type,
                sample.basic_unit_type,
                sample.topology_type,
                sample.symmetry,
                sample.connectivity_pattern,
                _json_dumps(sample.parameter_config),
                _json_dumps(sample.target_property),
                _json_dumps(sample.target_property),
                _json_dumps(sample.evaluated_property),
                _json_dumps(sample.evaluated_property),
                _json_dumps(sample.property_error),
                sample.fem_status,
                sample.geometry_status,
                sample.source,
                sample.timestamp,
                _json_dumps(metadata),
                _json_dumps(metadata),
                _json_dumps(explicit_structure),
                sample.structure_id,
            ),
        )

        artifact_map = {
            "structure_path": sample.structure_path,
            "csv_path": csv_path,
            "abaqus_txt_path": abaqus_txt_path,
            "crystal_txt_path": crystal_txt_path,
            "constraints_path": artifacts.get("constraints_path") or "",
            "summary_path": str(datagen_result.get("summary_path") or ""),
            "generated_data_manifest_path": str(datagen_result.get("generated_data_manifest_path") or ""),
            "knowledge_base_seed_path": str(datagen_result.get("knowledge_base_seed_path") or ""),
        }
        for artifact_type, artifact_path in artifact_map.items():
            if artifact_path:
                self._upsert_artifact(
                    artifact_id=f"{sample_id}:{artifact_type}",
                    sample_id=sample_id,
                    run_id=run_id,
                    artifact_type=artifact_type,
                    path=str(artifact_path),
                    description=f"{artifact_type} for {sample_id}",
                )

        if commit:
            self.conn.commit()

    def _upsert_run(
        self,
        *,
        run_id: str,
        group_name: str,
        run_dir: str,
        status: str,
        target_sample_count: int,
        generated_sample_count: int,
        summary_path: str,
        generated_data_manifest_path: str,
        knowledge_base_seed_path: str,
        created_at: str,
        updated_at: str,
    ):
        if not run_id:
            return
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (
                run_id, "group", run_dir, status, target_sample_count, generated_sample_count,
                summary_path, generated_data_manifest_path, knowledge_base_seed_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                "group" = excluded."group",
                run_dir = excluded.run_dir,
                status = excluded.status,
                target_sample_count = excluded.target_sample_count,
                generated_sample_count = excluded.generated_sample_count,
                summary_path = excluded.summary_path,
                generated_data_manifest_path = excluded.generated_data_manifest_path,
                knowledge_base_seed_path = excluded.knowledge_base_seed_path,
                updated_at = excluded.updated_at
            """,
            (
                run_id,
                group_name,
                run_dir,
                status,
                target_sample_count,
                generated_sample_count,
                summary_path,
                generated_data_manifest_path,
                knowledge_base_seed_path,
                created_at,
                updated_at,
            ),
        )

    def _upsert_datagen_config(
        self,
        *,
        config_id: str,
        run_id: str,
        group_name: str,
        suggestion_id: str,
        config_json: str,
        agent_reason: str,
        created_at: str,
    ):
        if not config_json or config_json == "{}":
            return
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO datagen_configs (
                config_id, run_id, "group", suggestion_id, config_json, agent_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (config_id, run_id, group_name, suggestion_id, config_json, agent_reason, created_at),
        )

    def _upsert_artifact(
        self,
        *,
        artifact_id: str,
        sample_id: str,
        run_id: str,
        artifact_type: str,
        path: str,
        description: str,
    ):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO artifacts (
                artifact_id, sample_id, run_id, artifact_type, path, description
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (artifact_id, sample_id, run_id, artifact_type, path, description),
        )

    def _derive_run_id(self, run_dir: str, datagen_config: dict[str, Any], sample: KnowledgeSample) -> str:
        if run_dir:
            return Path(run_dir).name
        suggestion_id = str(datagen_config.get("suggestion_id") or "")
        if suggestion_id:
            return suggestion_id
        return sample.structure_id.split(":", 2)[0]

    def _query_samples_sql(self, sql: str, params: tuple[Any, ...]) -> list[KnowledgeSample]:
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return [self._row_to_sample(row) for row in cur.fetchall()]

    def _query_by_label(self, label: str, group: str | None = None) -> list[KnowledgeSample]:
        sql = 'SELECT * FROM samples WHERE label = ?'
        params: list[Any] = [label]
        if group:
            sql += ' AND "group" = ?'
            params.append(group)
        sql += " ORDER BY timestamp DESC"
        return self._query_samples_sql(sql, tuple(params))

    def _top_by_distance(
        self,
        label: str,
        target_property: dict[str, float],
        k: int,
        group: str | None = None,
    ) -> list[KnowledgeSample]:
        samples = self._query_by_label(label, group=group)
        ranked = sorted(samples, key=lambda sample: _property_distance(target_property, sample.evaluated_property))
        return ranked[:k]

    def get_success_samples(
        self,
        target_property: dict[str, float] | None = None,
        group: str | None = None,
        top_k: int = 20,
    ) -> list[KnowledgeSample]:
        if target_property:
            return self._top_by_distance("success", target_property, top_k, group=group)
        return self._query_by_label("success", group=group)[:top_k]

    def get_near_miss_samples(
        self,
        target_property: dict[str, float] | None = None,
        group: str | None = None,
        top_k: int = 20,
    ) -> list[KnowledgeSample]:
        if target_property:
            return self._top_by_distance("near_miss", target_property, top_k, group=group)
        return self._query_by_label("near_miss", group=group)[:top_k]

    def get_failure_samples(
        self,
        group: str | None = None,
        reason_type: str | None = None,
        top_k: int = 20,
    ) -> list[KnowledgeSample]:
        failures = self._query_by_label("failure", group=group)
        if reason_type:
            filtered = []
            for sample in failures:
                metadata = sample.metadata or {}
                failure_analysis = metadata.get("agent_suggestion", {}).get("failure_analysis", {})
                if (
                    reason_type == sample.fem_status
                    or reason_type == sample.geometry_status
                    or failure_analysis.get("reason_type") == reason_type
                ):
                    filtered.append(sample)
            failures = filtered
        return failures[:top_k]

    def get_group_statistics(self) -> dict[str, dict[str, int]]:
        cur = self.conn.cursor()
        cur.execute('SELECT "group", label, COUNT(*) AS cnt FROM samples GROUP BY "group", label ORDER BY "group", label')
        stats: dict[str, dict[str, int]] = {}
        for row in cur.fetchall():
            group_name = str(row["group"] or "")
            bucket = stats.setdefault(group_name, {"total": 0})
            bucket[row["label"]] = int(row["cnt"])
            bucket["total"] += int(row["cnt"])
        return stats

    def list_samples(
        self,
        group: str | None = None,
        label: str | None = None,
        top_k: int | None = None,
    ) -> list[KnowledgeSample]:
        sql = "SELECT * FROM samples"
        clauses: list[str] = []
        params: list[Any] = []
        if group:
            clauses.append('"group" = ?')
            params.append(group)
        if label:
            clauses.append("label = ?")
            params.append(label)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC"
        if top_k is not None:
            sql += " LIMIT ?"
            params.append(int(top_k))
        return self._query_samples_sql(sql, tuple(params))

    def get_similar_property_samples(
        self,
        target_property: dict[str, float],
        top_k: int = 20,
        group: str | None = None,
    ) -> list[KnowledgeSample]:
        sql = "SELECT * FROM samples"
        params: tuple[Any, ...] = ()
        if group:
            sql += ' WHERE "group" = ?'
            params = (group,)
        sql += " ORDER BY timestamp DESC"
        samples = self._query_samples_sql(sql, params)
        ranked = sorted(samples, key=lambda sample: _property_distance(target_property, sample.evaluated_property))
        return ranked[:top_k]

    def get_run_provenance(self, run_id: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        run_row = cur.fetchone()
        if not run_row:
            return None
        cur.execute('SELECT * FROM datagen_configs WHERE run_id = ? ORDER BY created_at ASC', (run_id,))
        config_rows = [dict(row) for row in cur.fetchall()]
        cur.execute('SELECT sample_id, structure_id, label, "group", structure_path FROM samples WHERE run_id = ? ORDER BY timestamp ASC', (run_id,))
        sample_rows = [dict(row) for row in cur.fetchall()]
        cur.execute("SELECT artifact_type, path, sample_id FROM artifacts WHERE run_id = ? ORDER BY sample_id, artifact_type", (run_id,))
        artifact_rows = [dict(row) for row in cur.fetchall()]
        return {
            "run": dict(run_row),
            "datagen_configs": config_rows,
            "samples": sample_rows,
            "artifacts": artifact_rows,
        }

    def get_sample_evidence(self, sample_id: str) -> dict[str, Any] | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM samples WHERE sample_id = ? OR structure_id = ? LIMIT 1", (sample_id, sample_id))
        sample_row = cur.fetchone()
        if not sample_row:
            return None
        run_id = sample_row["run_id"] or ""
        cur.execute("SELECT * FROM runs WHERE run_id = ? LIMIT 1", (run_id,))
        run_row = cur.fetchone()
        cur.execute("SELECT * FROM datagen_configs WHERE run_id = ? ORDER BY created_at ASC", (run_id,))
        config_rows = [dict(row) for row in cur.fetchall()]
        cur.execute("SELECT artifact_type, path, description FROM artifacts WHERE sample_id = ? ORDER BY artifact_type", (sample_row["sample_id"],))
        artifact_rows = [dict(row) for row in cur.fetchall()]
        return {
            "sample": dict(sample_row),
            "run": dict(run_row) if run_row else None,
            "datagen_configs": config_rows,
            "artifacts": artifact_rows,
        }

    def query_top_success(self, target_property: dict[str, float], k: int) -> list[KnowledgeSample]:
        return self.get_success_samples(target_property=target_property, top_k=k)

    def query_top_near_miss(self, target_property: dict[str, float], k: int) -> list[KnowledgeSample]:
        return self.get_near_miss_samples(target_property=target_property, top_k=k)

    def query_recent_failures(self, target_property: dict[str, float], k: int) -> list[KnowledgeSample]:
        del target_property
        return self.get_failure_samples(top_k=k)

    def query_diverse_failures(self, target_property: dict[str, float], k: int) -> list[KnowledgeSample]:
        failures = self.get_failure_samples(top_k=max(k * 5, k))
        del target_property
        seen = set()
        diverse = []
        for sample in failures:
            key = (
                sample.symmetry,
                sample.topology_type,
                sample.connectivity_pattern,
                round(float(sample.evaluated_property.get("stiffness_proxy", 0.0)), 3),
            )
            if key in seen:
                continue
            seen.add(key)
            diverse.append(sample)
            if len(diverse) >= k:
                break
        return diverse

    def list_knowledge_evidence(
        self,
        label: str | None = None,
        top_k: int | None = None,
    ) -> list[KnowledgeEvidence]:
        sql = "SELECT * FROM knowledge_evidence"
        params: list[Any] = []
        if label:
            sql += " WHERE label = ?"
            params.append(label)
        sql += " ORDER BY rowid DESC"
        if top_k is not None:
            sql += " LIMIT ?"
            params.append(int(top_k))
        cur = self.conn.cursor()
        cur.execute(sql, tuple(params))
        return [self._row_to_knowledge_evidence(row) for row in cur.fetchall()]

    def get_new_evidence_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM knowledge_evidence WHERE used_for_training = 0")
        return int(cur.fetchone()[0])

    def mark_evidence_used_for_training(self) -> None:
        cur = self.conn.cursor()
        cur.execute("UPDATE knowledge_evidence SET used_for_training = 1 WHERE used_for_training = 0")
        self.conn.commit()

    def get_new_sample_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM samples WHERE used_for_training = 0")
        return int(cur.fetchone()[0])

    def export_training_dataset(self, mark_used: bool = True) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM samples WHERE used_for_training = 0 ORDER BY timestamp ASC")
        rows = cur.fetchall()
        samples = [self._row_to_sample(row).to_dict() for row in rows]
        if mark_used and rows:
            cur.execute("UPDATE samples SET used_for_training = 1 WHERE used_for_training = 0")
            self.conn.commit()
        return samples

    def dataset_statistics(self) -> dict[str, int]:
        cur = self.conn.cursor()
        cur.execute("SELECT label, COUNT(*) AS cnt FROM samples GROUP BY label")
        stats = {row["label"]: int(row["cnt"]) for row in cur.fetchall()}
        stats["total"] = sum(stats.values())
        return stats

    def _row_to_sample(self, row: sqlite3.Row) -> KnowledgeSample:
        return KnowledgeSample(
            structure_id=row["structure_id"],
            structure_path=row["structure_path"],
            unit_cell_type=row["unit_cell_type"],
            basic_unit_type=row["basic_unit_type"],
            topology_type=row["topology_type"],
            symmetry=row["symmetry"],
            connectivity_pattern=row["connectivity_pattern"],
            parameter_config=_json_loads(row["parameter_config"], {}),
            target_property=_json_loads(row["target_property"], {}),
            evaluated_property=_json_loads(row["evaluated_property"], {}),
            property_error=_json_loads(row["property_error"], {}),
            fem_status=row["fem_status"],
            geometry_status=row["geometry_status"],
            label=row["label"],
            source=row["source"],
            timestamp=row["timestamp"],
            metadata=_json_loads(row["metadata"], {}),
            explicit_structure=_json_loads(row["structure_json"] if "structure_json" in row.keys() else None, {}),
        )

    def _row_to_knowledge_evidence(self, row: sqlite3.Row) -> KnowledgeEvidence:
        return KnowledgeEvidence(
            evidence_id=row["evidence_id"],
            observation_id=row["observation_id"],
            meta_id=row["meta_id"],
            structure_id=row["structure_id"],
            meta_summary=_json_loads(row["meta_summary"], {}),
            structure_features=_json_loads(row["structure_features"], {}),
            property_result=_json_loads(row["property_result"], {}),
            error_to_target=_json_loads(row["error_to_target"], {}),
            label=row["label"],
            source=row["source"],
            proposal_group=row["proposal_group"],
            parent_id=row["parent_id"],
            hypothesis_id=row["hypothesis_id"],
            hypothesis=row["hypothesis"],
            intervention_type=row["intervention_type"],
            intervention_delta=_json_loads(row["intervention_delta"], {}),
            effect_summary=_json_loads(row["effect_summary"], {}),
            reasoning_tags=tuple(_json_loads(row["reasoning_tags"], [])),
            provenance=_json_loads(row["provenance"], {}),
            fidelity=row["fidelity"],
            confidence=float(row["confidence"]),
        )
