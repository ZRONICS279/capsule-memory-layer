"""Budgeted memory compiler.

The compiler is the product core for text-compatible agents: it turns a large,
lossless capsule ledger into the smallest useful prompt payload for the current
task. It does not summarize away memory. It selects active facts, attaches
evidence handles, and leaves the full raw events in storage for exact expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .capsule import Capsule
from .budget import estimate_tokens
from .sketch import tokenize
from .symbolic import Fact


PRECISION_KEYS = {
    "active_choice",
    "active_auth_middleware",
    "accepted_decision",
    "board",
    "config_key",
    "device",
    "do_not_use",
    "edit_file",
    "endpoint",
    "failing_file",
    "failing_test",
    "feature_flag",
    "final_value",
    "function_name",
    "language",
    "model",
    "next_command",
    "on_call_owner",
    "pagination_mode",
    "package",
    "path",
    "project",
    "region",
    "retention",
    "retention_days",
    "route_file",
    "schema_version",
    "test_file",
    "timeout",
    "timeout_seconds",
    "tool_evidence",
    "created_file",
    "run_test",
    "version",
}


@dataclass(frozen=True)
class EvidenceHandle:
    source: str
    key: str
    scope: str
    value: str

    def render(self) -> str:
        return f"{self.source}:{self.scope}:{self.key}"


@dataclass(frozen=True)
class CompiledMemory:
    capsule_id: str
    label: str
    version: int
    query: str
    payload: str
    token_estimate: int
    original_token_estimate: int
    evidence: tuple[EvidenceHandle, ...]
    omitted_fact_count: int

    @property
    def compression_ratio(self) -> float:
        if self.token_estimate == 0:
            return float("inf")
        return self.original_token_estimate / self.token_estimate


class MemoryCompiler:
    """Compile task-specific memory with provenance and a hard token budget."""

    def __init__(self, token_budget: int = 256, include_vector_digest: bool = False):
        if token_budget <= 32:
            raise ValueError("token_budget must be greater than 32")
        self.token_budget = token_budget
        self.include_vector_digest = include_vector_digest

    def compile(self, capsule: Capsule, *, query: str = "", original_context: str = "") -> CompiledMemory:
        original_tokens = estimate_tokens(original_context) or sum(estimate_tokens(event.text) for event in capsule.events)
        history_query = _looks_like_history_query(query)
        prompt_facts = capsule.active_facts() if history_query else _prompt_safe_facts(capsule.active_facts())
        # A project query used to keep only the high-priority keys, which threw
        # away exactly the facts a coding task needs: asking to "add a test"
        # dropped "prefer vitest" because preference has no priority score.
        # Priority still decides the order through _rank_facts.
        ranked = self._rank_facts(prompt_facts, query)

        header = [
            f"MEMORY_HANDLE {capsule.label} id={capsule.capsule_id} v={capsule.version}",
            "RULE active facts only; UNKNOWN if absent; CONFLICT if facts disagree.",
        ]
        if self.include_vector_digest:
            header.append(f"STATE_DIGEST {capsule.sketch.compact_digest()} shape={capsule.sketch.slots}x{capsule.sketch.dims}")

        lines = [*header, "ACTIVE_FACTS:"]
        evidence: list[EvidenceHandle] = []
        selected_facts: list[Fact] = []
        rejected_facts: list[Fact] = []
        used = estimate_tokens("\n".join(lines))
        selected = 0

        for fact in ranked:
            handle = EvidenceHandle(source=fact.source, key=fact.key, scope=fact.scope, value=fact.value)
            if _is_rejected_fact(fact):
                rejected_facts.append(fact)
                continue
            line = f"- {fact.scope}.{fact.key}={fact.value} evidence={handle.render()}"
            file_hint = _fact_file_hint(fact)
            if file_hint:
                line += f" file={file_hint}"
            if _query_needs_generic_voltage(query, fact):
                line += f" alias={fact.scope}.voltage={fact.value}"
            line_tokens = estimate_tokens(line)
            # The header alone can exhaust a small budget, which leaves a
            # payload that costs tokens and carries no memory at all. The
            # highest-ranked fact always goes in: paying for the scaffolding
            # and none of the content is the one outcome worth ruling out.
            if used + line_tokens + 12 > self.token_budget and selected_facts:
                continue
            lines.append(line)
            selected_facts.append(fact)
            evidence.append(handle)
            used += line_tokens
            selected += 1

        if rejected_facts and _should_include_rejected_facts(query, selected_facts):
            lines.append("DO_NOT_USE:")
            for fact in rejected_facts[:8]:
                handle = EvidenceHandle(source=fact.source, key=fact.key, scope=fact.scope, value=fact.value)
                lines.append(f"- {fact.value} evidence={handle.render()}; rejected.")
                evidence.append(handle)
                selected += 1
        omitted = max(0, len(ranked) - selected)
        if omitted and original_tokens > self.token_budget * 2:
            lines.append(f"OMITTED_FACTS {omitted}; use memory_retrieve for exact evidence if needed.")
        if original_tokens > self.token_budget * 2:
            lines.append("RESPONSE_RULE use exact active values; do not infer missing names/numbers/versions.")

        payload = "\n".join(lines)
        return CompiledMemory(
            capsule_id=capsule.capsule_id,
            label=capsule.label,
            version=capsule.version,
            query=query,
            payload=payload,
            token_estimate=estimate_tokens(payload),
            original_token_estimate=original_tokens,
            evidence=tuple(evidence),
            omitted_fact_count=omitted,
        )

    def _rank_facts(self, facts: tuple[Fact, ...], query: str) -> list[Fact]:
        """Order every active fact by relevance to the query.

        Ranking decides what comes first; the token budget decides what is left
        out. Dropping a fact only because the query happens not to repeat its
        words meant the caller had to already know the answer to be told it: a
        capsule holding "preference: biome" answered nothing to "set up the
        linter". A handful of project facts costs a few dozen tokens, so the
        budget is the honest limit, not word overlap.
        """

        query_terms = set(tokenize(query.replace("_", " ")))

        def score(fact: Fact) -> tuple[int, float, str]:
            fact_terms = set(tokenize(f"{fact.scope} {fact.key} {fact.key.replace('_', ' ')} {fact.value}"))
            overlap = len(query_terms & fact_terms)
            precision = _fact_priority(fact)
            confidence = fact.confidence
            return (overlap + precision, confidence, fact.observed_at)

        return sorted(facts, key=score, reverse=True)


def _query_needs_generic_voltage(query: str, fact: Fact) -> bool:
    return "voltage" in query.lower() and fact.key.endswith("_voltage") and fact.key != "voltage"


def _is_rejected_fact(fact: Fact) -> bool:
    return fact.key.startswith("rejected") or fact.key in {"do_not_use", "rejected_or_previous"} or fact.status != "active"


def _should_include_rejected_facts(query: str, selected_facts: list[Fact]) -> bool:
    query_terms = set(tokenize(query.replace("_", " ")))
    if query_terms & {"avoid", "rejected", "stale", "previous", "old", "wrong", "outdated"}:
        return True
    return any(_is_project_task_fact(fact) for fact in selected_facts)


def _is_project_task_fact(fact: Fact) -> bool:
    return fact.key in {
        "active_auth_middleware",
        "accepted_decision",
        "config_key",
        "created_file",
        "edit_file",
        "endpoint",
        "failing_file",
        "failing_test",
        "function_name",
        "next_command",
        "package",
        "route_file",
        "run_test",
        "schema_version",
        "test_file",
        "tool_evidence",
        "version",
    }


def _prompt_safe_facts(facts: tuple[Fact, ...]) -> tuple[Fact, ...]:
    active_auth = {fact.value for fact in facts if fact.key == "active_auth_middleware" and fact.status == "active"}
    config_keys = {fact.value for fact in facts if fact.key == "config_key" and fact.status == "active"}
    final_values = {fact.value for fact in facts if fact.key == "final_value" and fact.status == "active"}
    feature_flags = {_normalize_symbol(fact.value) for fact in facts if fact.key == "feature_flag" and fact.status == "active"}
    stale_feature_values = {
        fact.value
        for fact in facts
        if _normalize_symbol(fact.key) in feature_flags and fact.value not in final_values
    }

    safe: list[Fact] = []
    for fact in facts:
        if fact.status != "active":
            safe.append(fact)
            continue
        if active_auth and fact.key in {"auth_middleware", "preference"}:
            symbols = _class_symbols(fact.value)
            if symbols and not symbols.intersection(active_auth):
                continue
        if active_auth and fact.key not in PRECISION_KEYS:
            symbols = _class_symbols(fact.value)
            if symbols and not symbols.issubset(active_auth):
                continue
        if config_keys and fact.key not in PRECISION_KEYS:
            constants = _constant_symbols(fact.value)
            if constants and not constants.intersection(config_keys):
                continue
        if final_values and _normalize_symbol(fact.key) in feature_flags and fact.value not in final_values:
            continue
        if stale_feature_values and fact.key not in PRECISION_KEYS:
            lowered = fact.value.lower()
            if any(value.lower() in lowered for value in stale_feature_values):
                continue
        safe.append(fact)
    return tuple(safe)


def _looks_like_project_query(query: str) -> bool:
    lowered = query.lower()
    return bool(
        re.search(r"(?:^|/)[a-z0-9_.-]+\.[a-z0-9_]+", lowered)
        or any(
            marker in lowered
            for marker in (
                "apply",
                "edit",
                "file",
                "repo",
                "repository",
                "test",
                "route",
                "config",
                "migration",
                "source",
            )
        )
    )


def _looks_like_history_query(query: str) -> bool:
    lowered = query.lower()
    return any(
        marker in lowered
        for marker in (
            "history",
            "historical",
            "why",
            "previous",
            "previously",
            "old",
            "stale",
            "rejected",
            "superseded",
            "deprecated",
            "replace",
            "replaced",
        )
    )


def _class_symbols(value: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", value))


def _constant_symbols(value: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", value))


def _normalize_symbol(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _fact_file_hint(fact: Fact) -> str:
    paths = []
    for relation in fact.relations:
        if relation.kind == "belongs_to_file" and _looks_like_file_path(relation.target):
            if relation.target not in paths:
                paths.append(relation.target)
    return paths[0] if len(paths) == 1 else ""


def _looks_like_file_path(value: str) -> bool:
    return bool(re.search(r"(?:^|/)[A-Za-z0-9_.-]+\.[A-Za-z0-9_]+$", value.strip()))


def _fact_priority(fact: Fact) -> int:
    if fact.key.startswith("pump_") or fact.key.endswith("_voltage") or fact.key.endswith("_current"):
        return 18
    if fact.key in {
        "active_auth_middleware",
        "accepted_decision",
        "config_key",
        "endpoint",
        "feature_flag",
        "final_status",
        "final_value",
        "function_name",
        "on_call_owner",
        "package",
        "pagination_mode",
        "schema_version",
        "timeout_seconds",
        "version",
    }:
        return 16
    if fact.key in {"created_file", "edit_file", "failing_file", "route_file"}:
        return 10
    if fact.key in PRECISION_KEYS or fact.key.startswith(("physical_pin_", "pump_")):
        return 8
    return 0


def _task_plan_hint(facts: list[Fact]) -> list[str]:
    grouped = {
        "decision": [],
        "files": [],
        "commands": [],
        "evidence": [],
    }
    for fact in facts:
        if fact.key in {
            "active_auth_middleware",
            "accepted_decision",
            "config_key",
            "endpoint",
            "feature_flag",
            "final_value",
            "function_name",
            "on_call_owner",
            "package",
            "schema_version",
            "version",
        }:
            grouped["decision"].append(f"{fact.key}: {fact.value}")
        elif fact.key in {"created_file", "edit_file", "failing_file", "failing_test", "route_file", "test_file"}:
            grouped["files"].append(f"{fact.key}: {fact.value}")
        elif fact.key in {"next_command", "run_test"}:
            grouped["commands"].append(f"{fact.key}: {fact.value}")
        elif fact.key in {"tool_evidence"}:
            grouped["evidence"].append(f"{fact.key}: {fact.value}")

    if not any(grouped.values()):
        return []

    lines = ["TASK_PLAN_HINT:"]
    for label in ("decision", "files", "commands", "evidence"):
        if grouped[label]:
            lines.append(f"{label}:")
            for item in grouped[label][:3]:
                lines.append(f"- {item}")
    return lines
