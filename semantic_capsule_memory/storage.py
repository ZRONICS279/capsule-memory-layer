"""Storage backends for capsule state."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from contextlib import closing
import json
import sqlite3

from .capsule import Capsule


class CapsuleStore(Protocol):
    def load_all(self) -> dict[str, Capsule]:
        ...

    def save_all(self, capsules: dict[str, Capsule]) -> None:
        ...


class JsonCapsuleStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load_all(self) -> dict[str, Capsule]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            item["capsule_id"]: Capsule.from_json(item)
            for item in data.get("capsules", [])
        }

    def save_all(self, capsules: dict[str, Capsule]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"capsules": [capsule.to_json() for capsule in capsules.values()]}
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


class SQLiteCapsuleStore:
    """SQLite store for local production and desktop agent use."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def load_all(self) -> dict[str, Capsule]:
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute("select capsule_id, payload from capsules").fetchall()
        return {
            capsule_id: Capsule.from_json(json.loads(payload))
            for capsule_id, payload in rows
        }

    def save_all(self, capsules: dict[str, Capsule]) -> None:
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute("begin")
            conn.execute("delete from capsules")
            conn.executemany(
                "insert into capsules(capsule_id, version, updated_at, payload) values (?, ?, ?, ?)",
                [
                    (
                        capsule.capsule_id,
                        capsule.version,
                        capsule.updated_at,
                        json.dumps(capsule.to_json(), sort_keys=True),
                    )
                    for capsule in capsules.values()
                ],
            )
            conn.commit()

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """
                create table if not exists capsules (
                    capsule_id text primary key,
                    version integer not null,
                    updated_at text not null,
                    payload text not null
                )
                """
            )
            conn.execute("create index if not exists idx_capsules_updated_at on capsules(updated_at)")


def store_from_path(path: str | Path) -> CapsuleStore:
    suffix = Path(path).suffix.lower()
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return SQLiteCapsuleStore(path)
    return JsonCapsuleStore(path)
