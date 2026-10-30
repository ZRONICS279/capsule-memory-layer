"""Exact symbolic spine extraction and updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import blake2b
import re
from typing import Iterable


BOARD_RE = re.compile(r"\b(ESP32-C3|ESP32-S3|ESP32|Raspberry Pi Pico W|Raspberry Pi Pico|Arduino Uno)\b", re.I)
LANG_RE = re.compile(r"\b(MicroPython|Python(?:\s*3\.\d+)?|Arduino C\+\+|Node(?:\.js)?|TypeScript|JavaScript)\b", re.I)
PIN_EQ_RE = re.compile(r"\b(?:physical\s*)?pin\s*(\d+)\s*(?:=|==|is)\s*(?:BCM\s*)?(?:GPIO\s*)?(\d+)\b", re.I)
PREFER_RE = re.compile(r"\b(?:prefer|prefers|use)\s+([A-Za-z0-9_.+# -]{2,40}?)(?:\.|,|;|$)", re.I)
DO_NOT_RE = re.compile(r"\b(?:do not use|don't use|avoid)\s+([A-Za-z0-9_.+# -]{2,40}?)(?:\.|,|;|$)", re.I)
SWITCH_RE = re.compile(r"\b(?:switch(?:ed)?|moved|changed)\s+from\s+(.+?)\s+to\s+(.+?)(?:\.|,|;|$)", re.I)
KEY_VALUE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_ -]{1,30})\s*(?:=|:)\s*([A-Za-z0-9_.+#/ -]{1,80})(?:\.|,|;|$)")
CONSTANT_ASSIGNMENT_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*=\s*'?([A-Za-z0-9_.@#/-]+)'?")
PATH_RE = re.compile(r"(?<![\w/.-])(?:\.{0,2}/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9_]+")
REPLACE_WITH_RE = re.compile(r"\b(?:replace|replaced|replacing)\s+([A-Za-z0-9_.@#/-]+)\s+with\s+([A-Za-z0-9_.@#/-]+)", re.I)
DECISION_TASK_RE = re.compile(
    r"\b(?:final|revised|accepted|rejected)\s+([A-Za-z][A-Za-z0-9 _/-]{2,48}?)\s+"
    r"(?:decision|config|pin|migration)\b",
    re.I,
)
TOOL_OK_RE = re.compile(r"^TOOL\s+([A-Za-z0-9_-]+)\s+ok=(?:True|true)\b", re.I)
PROJECT_KEYS = {
    "accepted_decision",
    "active_auth_middleware",
    "config_key",
    "created_file",
    "edit_file",
    "endpoint",
    "failing_file",
    "failing_test",
    "feature_flag",
    "final_value",
    "function_name",
    "next_command",
    "on_call_owner",
    "package",
    "pagination_mode",
    "payment_platform_owner",
    "route_file",
    "run_test",
    "schema_version",
    "test_file",
    "tool_evidence",
    "timeout_seconds",
    "version",
}
PROJECT_KEY_VALUE_RE = re.compile(
    r"\b("
    + "|".join(re.escape(key) for key in sorted(PROJECT_KEYS, key=len, reverse=True))
    + r")\s*(?:=|:)\s*([^\n;]+?)(?=\s+(?:"
    + "|".join(re.escape(key) for key in sorted(PROJECT_KEYS, key=len, reverse=True))
    + r")\s*(?:=|:)|[.;]\s*(?:\n|$)|$)",
    re.I,
)
DECISION_RE = re.compile(r"\b(?:final\s+decision|decision|decided|choose|chose|selected|pick(?:ed)?)\s*(?:is|:|to|with|=)?\s*([A-Za-z][A-Za-z0-9_ .+#/-]{2,80})(?:\.|,|;|$)", re.I)
COLOR_RE = re.compile(r"\b(?:color|colour)\s*(?:is|:|=|picked|selected|ended with)?\s*([A-Za-z][A-Za-z -]{2,40})(?:\.|,|;|$)", re.I)


def source_hash(text: str) -> str:
    return blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def clean_value(value: str) -> str:
    value = re.split(r"\.\s+", value, maxsplit=1)[0]
    return " ".join(value.strip().strip("\"'` \t\r\n.,;:").split())


def clean_project_value(value: str) -> str:
    value = clean_value(value)
    value = re.split(
        r"\s+(?:variant|replaces|supersedes|current_value|initial_value|new_value|proposed_value|reason)\s*[=:]?",
        value,
        maxsplit=1,
        flags=re.I,
    )[0]
    value = re.split(r"\s+(?:must|should|will|while|and)\b", value, maxsplit=1, flags=re.I)[0]
    return clean_value(value)


@dataclass(frozen=True)
class Relation:
    kind: str
    target: str
    confidence: float = 1.0
    source: str = ""

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "target": self.target,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Relation":
        return cls(
            kind=data["kind"],
            target=data["target"],
            confidence=float(data.get("confidence", 1.0)),
            source=data.get("source", ""),
        )


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    scope: str = "global"
    status: str = "active"
    confidence: float = 1.0
    source: str = ""
    observed_at: str = field(default_factory=now_iso)
    supersedes: tuple[str, ...] = ()
    relations: tuple[Relation, ...] = ()

    @property
    def identity(self) -> str:
        raw = f"{self.scope}|{self.key}|{self.value}|{self.source}"
        return source_hash(raw)

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "scope": self.scope,
            "status": self.status,
            "confidence": self.confidence,
            "source": self.source,
            "observed_at": self.observed_at,
            "supersedes": list(self.supersedes),
            "relations": [relation.to_json() for relation in self.relations],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Fact":
        return cls(
            key=data["key"],
            value=data["value"],
            scope=data.get("scope", "global"),
            status=data.get("status", "active"),
            confidence=float(data.get("confidence", 1.0)),
            source=data.get("source", ""),
            observed_at=data.get("observed_at", now_iso()),
            supersedes=tuple(data.get("supersedes", ())),
            relations=tuple(Relation.from_json(item) for item in data.get("relations", ())),
        )


def extract_facts(text: str, *, scope: str = "global", source: str | None = None) -> list[Fact]:
    """Extract high-precision facts using conservative rules.

    This deliberately captures fewer facts rather than inventing fuzzy ones.
    Unknowns stay unknown, which is the anti-hallucination contract.
    """

    facts: list[Fact] = []
    src = source or source_hash(text)

    for match in BOARD_RE.finditer(text):
        if _is_negated(text, match.start()):
            facts.append(Fact(key="rejected_or_previous", value=clean_value(match.group(1)), scope=scope, status="superseded", source=src))
            continue
        facts.append(Fact(key="board", value=clean_value(match.group(1)), scope=scope, source=src))

    for match in LANG_RE.finditer(text):
        facts.append(Fact(key="language", value=clean_value(match.group(1)), scope=scope, source=src))

    for match in PIN_EQ_RE.finditer(text):
        physical, bcm = match.groups()
        facts.append(Fact(key=f"physical_pin_{physical}", value=f"BCM GPIO {bcm}", scope=scope, source=src))

    for match in PREFER_RE.finditer(text):
        if _is_negated(text, match.start()):
            continue
        value = clean_value(match.group(1))
        if value.lower() not in {"a", "an", "the", "it"}:
            facts.append(Fact(key="preference", value=value, scope=scope, source=src, confidence=0.8))

    for match in DO_NOT_RE.finditer(text):
        facts.append(Fact(key="do_not_use", value=clean_value(match.group(1)), scope=scope, source=src))

    for match in SWITCH_RE.finditer(text):
        old, new = clean_value(match.group(1)), clean_value(match.group(2))
        facts.append(Fact(key="rejected_or_previous", value=old, scope=scope, status="superseded", source=src))
        facts.append(Fact(key="active_choice", value=new, scope=scope, source=src, supersedes=(old,)))

    for match in PROJECT_KEY_VALUE_RE.finditer(text):
        key, value = normalize_key(match.group(1)), clean_project_value(match.group(2))
        if value:
            facts.append(
                Fact(
                    key=key,
                    value=value,
                    scope=scope,
                    status=_context_status(text, match.start(), match.end()),
                    source=src,
                )
            )

    for match in CONSTANT_ASSIGNMENT_RE.finditer(text):
        key, value = normalize_key(match.group(1)), clean_value(match.group(2))
        if value:
            facts.append(Fact(key=key, value=value, scope=scope, status=_context_status(text, match.start(), match.end()), source=src))

    for match in KEY_VALUE_RE.finditer(text):
        key, raw_value = normalize_key(match.group(1)), match.group(2)
        value = clean_value(raw_value)
        if key in PROJECT_KEYS or any(key.endswith(f"_{project_key}") for project_key in PROJECT_KEYS):
            continue
        if re.fullmatch(r"pin_\d+", key):
            continue
        status = _context_status(text, match.start(), match.start(2) + len(value))
        # Skip generic noise-dump keys — they embed raw entity names contextually
        # and risk leaking stale values (e.g. "ops_note_0 = @alice-chen as primary").
        # Structured facts come from PROJECT_KEY_VALUE_RE or named patterns above.
        if re.search(r"(?:_note|_observation|_report|_finding|_record|_log|_entry)(?:_\d+)?$", key):
            continue
        if key not in {"user", "assistant"} and len(value) > 1:
            facts.append(Fact(key=key, value=value, scope=scope, status=status, source=src))

    for match in COLOR_RE.finditer(text):
        facts.append(Fact(key="color", value=clean_value(match.group(1)), scope=scope, source=src))

    for match in DECISION_RE.finditer(text):
        value = clean_value(match.group(1))
        sentence_prefix = text[max(0, match.start() - 80): match.end()].lower()
        if re.search(r"\b(no selected|no .*decision|undecided|unknown)\b", sentence_prefix):
            continue
        if value and not any(fact.key == "decision" and fact.value == value for fact in facts):
            facts.append(Fact(key="decision", value=value, scope=scope, source=src, confidence=0.85))

    return _dedupe(_attach_event_relations(facts, text, source=src))


def merge_facts(existing: Iterable[Fact], incoming: Iterable[Fact]) -> list[Fact]:
    # Keys where only one value should be active at a time.
    # When a new fact arrives with the same key+scope, the old one is superseded.
    SINGLETON_KEYS = {
        "board", "language", "active_choice",
        "config_key", "active_auth_middleware", "accepted_decision",
        "endpoint", "function_name", "package", "version", "schema_version",
        "on_call_owner", "feature_flag", "final_status", "final_value",
        "pagination_mode", "timeout_seconds",
    }
    merged = list(existing)
    for fact in incoming:
        if any(old.identity == fact.identity for old in merged):
            continue
        if fact.key in SINGLETON_KEYS and fact.status == "active":
            merged = [
                old if old.key != fact.key or old.scope != fact.scope or old.status != "active"
                else Fact(
                    key=old.key,
                    value=old.value,
                    scope=old.scope,
                    status="superseded",
                    confidence=old.confidence,
                    source=old.source,
                    observed_at=old.observed_at,
                    supersedes=old.supersedes,
                    relations=_merge_relations(
                        old.relations,
                        (Relation("superseded_by", fact.value, source=fact.source),),
                    ),
                )
                for old in merged
            ]
        merged.append(fact)
    return merged


def active_facts(facts: Iterable[Fact]) -> list[Fact]:
    return [fact for fact in facts if fact.status == "active"]


def _dedupe(facts: list[Fact]) -> list[Fact]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[Fact] = []
    for fact in facts:
        marker = (fact.scope, fact.key, fact.value.lower(), fact.status)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(fact)
    return result


def _attach_event_relations(facts: list[Fact], text: str, *, source: str) -> list[Fact]:
    if not facts:
        return facts

    event_paths = tuple(dict.fromkeys(_paths(text)))
    replace_pairs = tuple((clean_value(match.group(1)), clean_value(match.group(2))) for match in REPLACE_WITH_RE.finditer(text))
    decision_tasks = tuple(
        normalize_key(match.group(1))
        for match in DECISION_TASK_RE.finditer(text)
        if normalize_key(match.group(1))
    )
    verified_by = _verified_by(text)

    enriched: list[Fact] = []
    for fact in facts:
        relations: list[Relation] = list(fact.relations)
        fact_paths = tuple(dict.fromkeys(_paths(fact.value)))
        if fact.key in {"created_file", "edit_file", "failing_file", "failing_test", "route_file", "test_file"}:
            for path in fact_paths or ((fact.value,) if _looks_like_path(fact.value) else ()):
                relations.append(Relation("belongs_to_file", path, source=source))
        elif len(event_paths) == 1 and _is_projectish_fact(fact):
            relations.append(Relation("belongs_to_file", event_paths[0], confidence=0.9, source=source))
        elif len(fact_paths) == 1:
            relations.append(Relation("belongs_to_file", fact_paths[0], confidence=0.95, source=source))

        for old, new in replace_pairs:
            if _same_symbolish(fact.value, new) or new in fact.value:
                relations.append(Relation("replaces", old, source=source))
                relations.append(Relation("supersedes", old, source=source))
            if _same_symbolish(fact.value, old) or fact.status != "active":
                relations.append(Relation("rejected_option", old, source=source))

        for value in fact.supersedes:
            relations.append(Relation("supersedes", value, source=source))
            relations.append(Relation("rejected_option", value, source=source))

        if fact.status != "active" or fact.key in {"do_not_use", "rejected_or_previous"}:
            relations.append(Relation("rejected_option", fact.value, source=source))

        for task in decision_tasks:
            if _is_projectish_fact(fact):
                relations.append(Relation("decision_for_task", task, confidence=0.85, source=source))

        if verified_by and fact.key in {"run_test", "tool_evidence", "next_command"}:
            relations.append(Relation("verified_by", verified_by, source=source))

        enriched.append(
            Fact(
                key=fact.key,
                value=fact.value,
                scope=fact.scope,
                status=fact.status,
                confidence=fact.confidence,
                source=fact.source,
                observed_at=fact.observed_at,
                supersedes=fact.supersedes,
                relations=_merge_relations((), tuple(relations)),
            )
        )
    return enriched


def _merge_relations(existing: tuple[Relation, ...], incoming: tuple[Relation, ...]) -> tuple[Relation, ...]:
    merged = list(existing)
    seen = {(relation.kind, relation.target, relation.source) for relation in merged}
    for relation in incoming:
        target = clean_value(relation.target)
        if not target:
            continue
        cleaned = Relation(
            kind=relation.kind,
            target=target,
            confidence=relation.confidence,
            source=relation.source,
        )
        marker = (cleaned.kind, cleaned.target, cleaned.source)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(cleaned)
    return tuple(merged)


def _paths(text: str) -> list[str]:
    paths: list[str] = []
    for match in PATH_RE.finditer(text):
        path = match.group(0).strip("`'\".,;:)")
        if re.fullmatch(r"\d+(?:\.\d+)+", path):
            continue
        if path and path not in paths:
            paths.append(path)
    return paths


def _looks_like_path(value: str) -> bool:
    return bool(PATH_RE.fullmatch(value.strip()))


def _same_symbolish(left: str, right: str) -> bool:
    return normalize_key(left) == normalize_key(right)


def _verified_by(text: str) -> str:
    match = TOOL_OK_RE.match(text.strip())
    if match:
        return match.group(1)
    lowered = text.lower()
    if "tests pass" in lowered or "test passed" in lowered or "run_eval.sh" in lowered:
        return "tests_passed"
    return ""


def _is_projectish_fact(fact: Fact) -> bool:
    return (
        fact.key in PROJECT_KEYS
        or fact.key in {"active_choice", "decision", "do_not_use", "rejected_or_previous", "preference"}
        or fact.key.isupper()
        or any(fact.key.endswith(f"_{project_key}") for project_key in PROJECT_KEYS)
    )


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 28):start].lower()
    return bool(re.search(r"\b(not|no|avoid|without|don't use|do not use)\s+(?:regular\s+)?$", prefix))


def _context_status(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    local_start = start - line_start
    local_end = end - line_start
    boundaries = [0]
    boundaries.extend(match.end() for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", line))
    boundaries.append(len(line))
    sentence_start = max(boundary for boundary in boundaries if boundary <= local_start)
    sentence_end = min(boundary for boundary in boundaries if boundary >= local_end)
    sentence = line[sentence_start:sentence_end].lower()
    if re.search(
        r"\b(stale|rejected|deprecated|superseded|replaces|initial default|docs draft|do not use|don't use|not final)\b",
        sentence,
    ):
        return "superseded"
    return "active"
