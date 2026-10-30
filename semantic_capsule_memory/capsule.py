"""Versioned capsule memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .sketch import VectorSketch
from .budget import estimate_tokens
from .symbolic import Fact, active_facts, extract_facts, merge_facts, now_iso, source_hash


@dataclass(frozen=True)
class Event:
    text: str
    source: str
    scope: str = "global"
    kind: str = "event"
    parent: str = ""
    observed_at: str = field(default_factory=now_iso)

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "scope": self.scope,
            "kind": self.kind,
            "parent": self.parent,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Event":
        return cls(
            text=data["text"],
            source=data["source"],
            scope=data.get("scope", "global"),
            kind=data.get("kind", "event"),
            parent=data.get("parent", ""),
            observed_at=data["observed_at"],
        )


@dataclass(frozen=True)
class QueryResult:
    answer: str
    confidence: float
    sources: tuple[str, ...] = ()
    status: str = "answered"


@dataclass(frozen=True)
class Capsule:
    capsule_id: str
    label: str
    version: int = 0
    sketch: VectorSketch = field(default_factory=VectorSketch.empty)
    facts: tuple[Fact, ...] = ()
    events: tuple[Event, ...] = ()
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def update(self, text: str, *, scope: str = "global", chunk_tokens: int = 384) -> "Capsule":
        parent = source_hash(text)
        events = tuple(
            Event(text=chunk, source=source_hash(f"{parent}:{index}:{chunk}"), scope=scope, kind="chunk", parent=parent)
            for index, chunk in enumerate(chunk_text(text, max_tokens=chunk_tokens))
        )
        facts = list(self.facts)
        for event in events:
            facts = merge_facts(facts, extract_facts(event.text, scope=scope, source=event.source))
        return Capsule(
            capsule_id=self.capsule_id,
            label=self.label,
            version=self.version + 1,
            sketch=self.sketch.add_text(text),
            facts=tuple(facts),
            events=self.events + events,
            created_at=self.created_at,
            updated_at=now_iso(),
        )

    def active_facts(self) -> tuple[Fact, ...]:
        return tuple(active_facts(self.facts))

    def render_context(self, *, max_facts: int = 24) -> str:
        """Render the tiny prompt payload for an LLM.

        This is the only text meant to be injected. The event ledger remains
        available for audit, not as default prompt bloat.
        """

        lines = [
            f"CAPSULE {self.label} id={self.capsule_id} version={self.version}",
            f"VECTOR_DIGEST {self.sketch.compact_digest()} shape={self.sketch.slots}x{self.sketch.dims}",
            "EXACT_SYMBOLIC_SPINE:",
        ]
        for fact in self.active_facts()[:max_facts]:
            source = f" source={fact.source}" if fact.source else ""
            lines.append(f"- [{fact.scope}] {fact.key}: {fact.value}{source}")
        if len(self.active_facts()) > max_facts:
            lines.append(f"- ... {len(self.active_facts()) - max_facts} more active facts hidden by budget")
        lines.append("RULE: use exact spine for names, versions, numbers, paths, pins, APIs, and decisions; answer unknown if absent.")
        return "\n".join(lines)

    def query_fact(self, key: str, *, scope: str | None = None) -> QueryResult:
        normalized = key.strip().lower().replace(" ", "_")
        matches = [
            fact for fact in self.active_facts()
            if fact.key == normalized and (scope is None or fact.scope == scope)
        ]
        if not matches:
            return QueryResult(
                answer=f"Unknown: no active fact for {normalized}.",
                confidence=0.0,
                status="unknown",
            )
        if len({fact.value for fact in matches}) > 1:
            values = "; ".join(f"{fact.value} ({fact.source})" for fact in matches)
            return QueryResult(
                answer=f"Conflict for {normalized}: {values}",
                confidence=0.4,
                sources=tuple(fact.source for fact in matches),
                status="conflict",
            )
        fact = matches[-1]
        return QueryResult(answer=fact.value, confidence=fact.confidence, sources=(fact.source,))

    def inspect(self) -> dict:
        return {
            "capsule_id": self.capsule_id,
            "label": self.label,
            "version": self.version,
            "vector_digest": self.sketch.compact_digest(),
            "vector_shape": [self.sketch.slots, self.sketch.dims],
            "active_facts": [fact.to_json() for fact in self.active_facts()],
            "superseded_facts": [fact.to_json() for fact in self.facts if fact.status != "active"],
            "event_count": len(self.events),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_json(self) -> dict:
        return {
            "capsule_id": self.capsule_id,
            "label": self.label,
            "version": self.version,
            "sketch": self.sketch.to_json(),
            "facts": [fact.to_json() for fact in self.facts],
            "events": [event.to_json() for event in self.events],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Capsule":
        return cls(
            capsule_id=data["capsule_id"],
            label=data["label"],
            version=int(data.get("version", 0)),
            sketch=VectorSketch.from_json(data["sketch"]),
            facts=tuple(Fact.from_json(item) for item in data.get("facts", [])),
            events=tuple(Event.from_json(item) for item in data.get("events", [])),
            created_at=data.get("created_at", now_iso()),
            updated_at=data.get("updated_at", now_iso()),
        )


class CapsuleManager:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        from .storage import store_from_path

        self.store = store_from_path(self.path)
        self.capsules: dict[str, Capsule] = {}
        self.load()

    def create(self, capsule_id: str, label: str | None = None) -> Capsule:
        if capsule_id in self.capsules:
            raise ValueError(f"Capsule already exists: {capsule_id}")
        capsule = Capsule(capsule_id=capsule_id, label=label or f"@{capsule_id}")
        self.capsules[capsule_id] = capsule
        self.save()
        return capsule

    def get(self, capsule_id: str) -> Capsule:
        try:
            return self.capsules[capsule_id]
        except KeyError as exc:
            raise KeyError(f"Unknown capsule: {capsule_id}") from exc

    def update(self, capsule_id: str, text: str, *, scope: str = "global", chunk_tokens: int = 384) -> Capsule:
        capsule = self.get(capsule_id).update(text, scope=scope, chunk_tokens=chunk_tokens)
        self.capsules[capsule_id] = capsule
        self.save()
        return capsule

    def similar(self, text: str, *, limit: int = 5) -> list[tuple[str, float]]:
        query = VectorSketch.from_text(text)
        scored = [
            (capsule_id, capsule.sketch.similarity(query))
            for capsule_id, capsule in self.capsules.items()
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def load(self) -> None:
        self.capsules = self.store.load_all()

    def save(self) -> None:
        self.store.save_all(self.capsules)


def chunk_text(text: str, *, max_tokens: int = 384) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        para_tokens = estimate_tokens(paragraph)
        if current and current_tokens + para_tokens > max_tokens:
            pieces.append("\n".join(current))
            current = []
            current_tokens = 0
        if para_tokens > max_tokens:
            words = paragraph.split()
            bucket: list[str] = []
            for word in words:
                bucket.append(word)
                if estimate_tokens(" ".join(bucket)) >= max_tokens:
                    pieces.append(" ".join(bucket))
                    bucket = []
            if bucket:
                current.append(" ".join(bucket))
                current_tokens += estimate_tokens(current[-1])
        else:
            current.append(paragraph)
            current_tokens += para_tokens
    if current:
        pieces.append("\n".join(current))
    return pieces or [text]
