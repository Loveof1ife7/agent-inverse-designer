from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..closed_loop_contracts import CurveLabelPair, DatasetUpdateSummary


class DatasetManager:
    """Dataset memory for the simplified closed loop.

    This replaces online KnowledgeBase/LoopMemory in the deterministic workflow.
    It stores factual training pairs and keeps label provenance explicit.
    """

    def __init__(
        self,
        root_dir: str | Path = "workspace/datasets",
        *,
        surrogate_inverse_weight: float = 0.25,
        simulation_inverse_weight: float = 1.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.inverse_surrogate_path = self.root_dir / "inverse_surrogate_pairs.jsonl"
        self.inverse_simulation_path = self.root_dir / "inverse_simulation_pairs.jsonl"
        self.forward_simulation_path = self.root_dir / "forward_simulation_pairs.jsonl"
        self.manifest_path = self.root_dir / "dataset_manifest.json"
        self.surrogate_inverse_weight = float(surrogate_inverse_weight)
        self.simulation_inverse_weight = float(simulation_inverse_weight)
        self._inverse_surrogate_train_cursor = 0
        self._inverse_simulation_train_cursor = 0
        self._forward_train_cursor = 0
        self._write_manifest()

    def append_surrogate_pair(self, pair: CurveLabelPair) -> CurveLabelPair:
        if pair.label_source != "surrogate":
            raise ValueError("append_surrogate_pair requires label_source='surrogate'")
        weighted = replace(pair, label_weight=self.surrogate_inverse_weight, model_consumers=("InverseDesigner",))
        self._append_jsonl(self.inverse_surrogate_path, weighted.to_dict())
        self._write_manifest()
        return weighted

    def append_simulation_pair(self, pair: CurveLabelPair) -> CurveLabelPair:
        if pair.label_source != "simulation":
            raise ValueError("append_simulation_pair requires label_source='simulation'")
        weighted = replace(pair, label_weight=self.simulation_inverse_weight, model_consumers=("InverseDesigner", "ForwardSurrogate"))
        self._append_jsonl(self.inverse_simulation_path, weighted.to_dict())
        self._append_jsonl(self.forward_simulation_path, weighted.to_dict())
        self._write_manifest()
        return weighted

    def inverse_training_rows(self) -> list[dict[str, Any]]:
        pairs = self._load_pairs(self.inverse_surrogate_path) + self._load_pairs(self.inverse_simulation_path)
        return [pair.to_inverse_training_row() for pair in pairs]

    def forward_training_rows(self) -> list[dict[str, Any]]:
        pairs = self._load_pairs(self.forward_simulation_path)
        return [pair.to_forward_training_row() for pair in pairs if pair.label_source == "simulation"]

    def new_inverse_training_rows(self, *, consume: bool = True) -> list[dict[str, Any]]:
        surrogate_pairs = self._load_pairs(self.inverse_surrogate_path)
        simulation_pairs = self._load_pairs(self.inverse_simulation_path)
        new_surrogate_pairs = surrogate_pairs[self._inverse_surrogate_train_cursor :]
        new_simulation_pairs = simulation_pairs[self._inverse_simulation_train_cursor :]
        new_rows = [
            pair.to_inverse_training_row()
            for pair in [*new_surrogate_pairs, *new_simulation_pairs]
        ]
        if consume:
            self._inverse_surrogate_train_cursor = len(surrogate_pairs)
            self._inverse_simulation_train_cursor = len(simulation_pairs)
        return new_rows

    def new_forward_training_rows(self, *, consume: bool = True) -> list[dict[str, Any]]:
        rows = self.forward_training_rows()
        new_rows = rows[self._forward_train_cursor :]
        if consume:
            self._forward_train_cursor = len(rows)
        return new_rows

    def select_training_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        max_curve_nmae: float | None = None,
        label_sources: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for row in rows:
            label_source = str(row.get("label_source") or "")
            if label_sources is not None and label_source not in label_sources:
                continue
            if max_curve_nmae is not None:
                provenance = dict(row.get("provenance") or {})
                acceptance = dict(provenance.get("acceptance") or provenance.get("surrogate_acceptance") or {})
                curve_nmae = acceptance.get("curve_nmae")
                if curve_nmae is not None:
                    try:
                        if float(curve_nmae) > float(max_curve_nmae):
                            continue
                    except (TypeError, ValueError):
                        pass
            selected.append(row)
        return selected

    def update_models(
        self,
        *,
        inverse_designer: Any | None = None,
        forward_surrogate: Any | None = None,
        min_inverse_rows: int = 1,
        min_forward_rows: int = 1,
        max_inverse_curve_nmae: float | None = None,
        max_forward_curve_nmae: float | None = None,
    ) -> DatasetUpdateSummary:
        pending_inverse_rows = self.new_inverse_training_rows(consume=False)
        pending_forward_rows = self.new_forward_training_rows(consume=False)
        inverse_rows = self.select_training_rows(
            pending_inverse_rows,
            max_curve_nmae=max_inverse_curve_nmae,
        )
        forward_rows = self.select_training_rows(
            pending_forward_rows,
            max_curve_nmae=max_forward_curve_nmae,
            label_sources={"simulation"},
        )
        inverse_training_weight = self._training_weight(inverse_rows)
        updated_inverse = False
        updated_forward = False

        if inverse_designer is not None and inverse_training_weight >= max(1, int(min_inverse_rows)):
            if hasattr(inverse_designer, "finetune"):
                inverse_designer.finetune(inverse_rows)
            elif hasattr(inverse_designer, "train"):
                inverse_designer.train(self.inverse_training_rows())
            updated_inverse = True
            self._inverse_surrogate_train_cursor = self._count_jsonl(self.inverse_surrogate_path)
            self._inverse_simulation_train_cursor = self._count_jsonl(self.inverse_simulation_path)

        if forward_surrogate is not None and len(forward_rows) >= max(1, int(min_forward_rows)):
            if hasattr(forward_surrogate, "finetune"):
                forward_surrogate.finetune(forward_rows)
            elif hasattr(forward_surrogate, "train"):
                forward_surrogate.train(self.forward_training_rows())
            updated_forward = True
            self._forward_train_cursor = len(self.forward_training_rows())

        counts = self.counts()
        return DatasetUpdateSummary(
            inverse_surrogate_pairs=counts["inverse_surrogate_pairs"],
            inverse_simulation_pairs=counts["inverse_simulation_pairs"],
            forward_simulation_pairs=counts["forward_simulation_pairs"],
            inverse_training_rows=len(inverse_rows),
            inverse_training_weight=inverse_training_weight,
            forward_training_rows=len(forward_rows),
            updated_inverse_designer=updated_inverse,
            updated_forward_surrogate=updated_forward,
        )

    def counts(self) -> dict[str, int]:
        return {
            "inverse_surrogate_pairs": self._count_jsonl(self.inverse_surrogate_path),
            "inverse_simulation_pairs": self._count_jsonl(self.inverse_simulation_path),
            "forward_simulation_pairs": self._count_jsonl(self.forward_simulation_path),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "paths": {
                "inverse_surrogate": str(self.inverse_surrogate_path),
                "inverse_simulation": str(self.inverse_simulation_path),
                "forward_simulation": str(self.forward_simulation_path),
                "manifest": str(self.manifest_path),
            },
            "counts": self.counts(),
            "weights": {
                "surrogate_inverse": self.surrogate_inverse_weight,
                "simulation_inverse": self.simulation_inverse_weight,
            },
        }

    def _write_manifest(self) -> None:
        payload = self.to_dict()
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _count_jsonl(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    @staticmethod
    def _training_weight(rows: list[dict[str, Any]]) -> float:
        total = 0.0
        for row in rows:
            try:
                total += float(row.get("weight", 1.0))
            except (TypeError, ValueError):
                total += 1.0
        return total

    @staticmethod
    def _load_pairs(path: Path) -> list[CurveLabelPair]:
        if not path.exists():
            return []
        pairs: list[CurveLabelPair] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                pairs.append(
                    CurveLabelPair(
                        pair_id=str(payload["pair_id"]),
                        structure=dict(payload.get("structure") or {}),
                        stress_curve=dict(payload.get("stress_curve") or {}),
                        label_source=str(payload.get("label_source") or ""),
                        label_weight=float(payload.get("label_weight", 1.0)),
                        model_consumers=tuple(payload.get("model_consumers") or ()),
                        target_property=dict(payload.get("target_property") or {}),
                        provenance=dict(payload.get("provenance") or {}),
                    )
                )
        return pairs
