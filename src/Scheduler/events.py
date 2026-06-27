from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..closed_loop_contracts import SchedulerEvent


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventStream:
    def __init__(self, task_id: str, events_dir: str | Path, mirror_path: str | Path | None = None):
        self.task_id = task_id
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.events_dir / "events.jsonl"
        self.mirror_path = Path(mirror_path) if mirror_path else None
        if self.mirror_path:
            self.mirror_path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._events: list[SchedulerEvent] = []

    def emit(self, stage: str, status: str, payload: dict | None = None) -> SchedulerEvent:
        self._seq += 1
        event = SchedulerEvent(
            seq=self._seq,
            task_id=self.task_id,
            timestamp=now_iso(),
            stage=stage,
            status=status,
            payload=payload or {},
        )
        self._events.append(event)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        if self.mirror_path:
            with self.mirror_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event

    @property
    def events(self) -> list[SchedulerEvent]:
        return list(self._events)
