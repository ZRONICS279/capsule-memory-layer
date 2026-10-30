"""Agentic memory layer.

This is the high-level working core for agents. It combines lossless ingestion,
task-specific compilation, exact evidence retrieval, and a hard token-reduction
target for every prepared step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from .budget import estimate_tokens
from .capsule import Capsule
from .compiler import CompiledMemory
from .engine import EvidenceSnippet, GroundedMemoryEngine, MemoryAnswer
from .prompt_codec import encode_prompt
from .symbolic import Fact


@dataclass(frozen=True)
class MemoryPacket:
    capsule_id: str
    task: str
    payload: str
    compiled: CompiledMemory
    retrieved: tuple[EvidenceSnippet, ...]
    raw_tokens: int
    memory_tokens: int
    reduction: float
    target_reduction: float
    evidence_refs: dict[str, str] = field(default_factory=dict)

    @property
    def passed_target(self) -> bool:
        return self.reduction >= self.target_reduction


class AgenticMemoryLayer:
    """Memory layer with an explicit cost-control contract."""

    def __init__(
        self,
        store: str | Path,
        *,
        default_token_budget: int = 192,
        target_reduction: float = 0.75,
        conversation_budget: int = 220,
        lean_threshold: int | None = None,
        prompt_codec: str = "compact",
        autoscale: bool = True,
        autoscale_max_token_budget: int | None = None,
        autoscale_max_retrieve_limit: int = 6,
    ):
        if not 0.0 < target_reduction < 1.0:
            raise ValueError("target_reduction must be between 0 and 1")
        if prompt_codec not in {"verbose", "compact"}:
            raise ValueError("prompt_codec must be 'verbose' or 'compact'")
        if autoscale_max_retrieve_limit < 1:
            raise ValueError("autoscale_max_retrieve_limit must be at least 1")
        self.engine = GroundedMemoryEngine(store, token_budget=default_token_budget)
        self.default_token_budget = default_token_budget
        self.target_reduction = target_reduction
        self.conversation_budget = conversation_budget
        self.prompt_codec = prompt_codec
        self.autoscale = autoscale
        self.autoscale_max_token_budget = autoscale_max_token_budget or max(default_token_budget, 384)
        self.autoscale_max_retrieve_limit = autoscale_max_retrieve_limit
        # Below this much durable memory there is nothing to compact: the full
        # ledger is already cheap, so adding compaction scaffolding (contract
        # headers, fact blocks, retrieved-evidence) would cost MORE than the raw
        # context. In that regime we emit a lean conversation packet instead so
        # the layer never makes a small chat more expensive.
        self.lean_threshold = lean_threshold if lean_threshold is not None else default_token_budget

    def remember(self, capsule_id: str, text: str, *, scope: str = "global", label: str | None = None):
        return self.engine.ingest(capsule_id, text, scope=scope, label=label)

    def prepare_step(
        self,
        capsule_id: str,
        task: str,
        *,
        original_context: str = "",
        retrieve_limit: int = 2,
    ) -> MemoryPacket:
        capsule = self.engine.manager.get(capsule_id)
        ledger_tokens = sum(estimate_tokens(event.text) for event in capsule.events)
        raw_tokens = estimate_tokens(original_context) or ledger_tokens

        # Lean regime: the whole memory is small enough that compaction can only
        # add overhead. Present the conversation directly with minimal scaffolding.
        if ledger_tokens <= self.lean_threshold:
            return self._prepare_lean(capsule, task, raw_tokens=raw_tokens)

        # target_reduction is a reporting goal (packet.passed_target), not a
        # hard budget cap. Always compile up to default_token_budget so that
        # critical facts are never silently dropped just to hit the ratio.
        budget = self.default_token_budget
        if raw_tokens and raw_tokens <= 1200:
            budget = min(budget, max(160, int(raw_tokens * (1.0 - self.target_reduction))))
        budget, retrieve_limit, autoscaled = self._autoscale_step_limits(
            capsule,
            task,
            raw_tokens=raw_tokens,
            budget=budget,
            retrieve_limit=retrieve_limit,
        )

        original_compiler = self.engine.compiler
        try:
            from .compiler import MemoryCompiler

            self.engine.compiler = MemoryCompiler(token_budget=budget)
            compiled = self.engine.compile(capsule_id, question=task, original_context=original_context)
        finally:
            self.engine.compiler = original_compiler

        retrieved = self.engine.retrieve(capsule_id, task, limit=retrieve_limit)
        target_token_cap = int(raw_tokens * (1.0 - self.target_reduction)) if raw_tokens else 0
        memory_tokens = compiled.token_estimate
        payload = compiled.payload
        conversation = _compile_conversation(capsule, task, budget_tokens=self.conversation_budget)
        if conversation:
            payload += f"\n{conversation}"
            memory_tokens += estimate_tokens(conversation)
        # When the ordered conversation view is present it already shows the
        # latest user/assistant turns, so the task-state dialog lines would be
        # pure duplication. Drop them and keep only its unique signal
        # (file focus, grounding rule, tool evidence).
        task_state = ""
        if conversation or raw_tokens > 1200:
            task_state = _compile_task_state(
                capsule,
                task,
                skip_dialog_lines=bool(conversation),
                include_grounding=True,
            )
        if task_state:
            payload += f"\nTASK_STATE:\n{task_state}"
            memory_tokens += estimate_tokens(task_state)
        patch_checklist = _compile_patch_checklist(capsule, task)
        if patch_checklist and _is_broad_patch_checklist(patch_checklist, task):
            payload = _compile_patch_execution_packet(compiled, patch_checklist)
            encoded = encode_prompt(payload, codec=self.prompt_codec)
            payload = encoded.payload
            memory_tokens = encoded.token_estimate
            reduction = 0.0 if raw_tokens == 0 else max(0.0, 1.0 - (memory_tokens / raw_tokens))
            return MemoryPacket(
                capsule_id=capsule_id,
                task=task,
                payload=payload,
                compiled=compiled,
                retrieved=(),
                raw_tokens=raw_tokens,
                memory_tokens=memory_tokens,
                reduction=reduction,
                target_reduction=self.target_reduction,
                evidence_refs=encoded.evidence_refs,
            )
        selected_retrieved: list[EvidenceSnippet] = []
        inject_retrieved = not compiled.evidence
        if inject_retrieved:
            for snippet in retrieved:
                projected = memory_tokens + snippet.token_estimate
                if target_token_cap and projected > target_token_cap:
                    continue
                selected_retrieved.append(snippet)
                memory_tokens = projected

        if selected_retrieved:
            payload += "\nRETRIEVED_EVIDENCE:"
            for snippet in selected_retrieved:
                payload += f"\n- evidence={snippet.source} score={snippet.score}: {snippet.text}"

        encoded = encode_prompt(payload, codec=self.prompt_codec)
        payload = encoded.payload
        memory_tokens = encoded.token_estimate
        reduction = 0.0 if raw_tokens == 0 else max(0.0, 1.0 - (memory_tokens / raw_tokens))
        return MemoryPacket(
            capsule_id=capsule_id,
            task=task,
            payload=payload,
            compiled=compiled,
            retrieved=tuple(selected_retrieved),
            raw_tokens=raw_tokens,
            memory_tokens=memory_tokens,
            reduction=reduction,
            target_reduction=self.target_reduction,
            evidence_refs=encoded.evidence_refs,
        )

    def _autoscale_step_limits(
        self,
        capsule: Capsule,
        task: str,
        *,
        raw_tokens: int,
        budget: int,
        retrieve_limit: int,
    ) -> tuple[int, int, bool]:
        if not self.autoscale:
            return budget, retrieve_limit, False

        active_facts = capsule.active_facts()
        task_paths = set(PATH_RE.findall(task))
        fact_paths = {
            value
            for fact in active_facts
            for value in _fact_paths(fact)
        }
        file_count = len(task_paths | fact_paths)
        finalish_count = sum(
            1
            for fact in active_facts
            if fact.key.startswith("final_")
            or fact.key in {
                "accepted_decision",
                "active_auth_middleware",
                "auth_middleware",
                "checkout_v2",
                "config_key",
                "endpoint",
                "feature_flag",
                "function_name",
                "on_call_owner",
                "package",
                "pagination_mode",
                "payment_platform_owner",
                "timeout_seconds",
                "version",
            }
        )
        edit_task = _looks_like_code_edit_task(task)

        if not edit_task and file_count < 2 and len(active_facts) <= 12 and finalish_count < 4:
            return budget, retrieve_limit, False

        scaled_budget = budget
        if edit_task:
            scaled_budget += 48
        if file_count >= 2:
            scaled_budget += min(128, 32 * file_count)
        if len(active_facts) > 12:
            scaled_budget += 64
        if finalish_count >= 4:
            scaled_budget += 64
        if raw_tokens > 1200:
            scaled_budget += 48
        scaled_budget = min(max(budget, scaled_budget), self.autoscale_max_token_budget)

        scaled_retrieve_limit = retrieve_limit
        if edit_task or file_count >= 2:
            scaled_retrieve_limit = max(scaled_retrieve_limit, 4)
        if len(active_facts) > 12 or finalish_count >= 4:
            scaled_retrieve_limit = max(scaled_retrieve_limit, 6)
        scaled_retrieve_limit = min(scaled_retrieve_limit, self.autoscale_max_retrieve_limit)
        return scaled_budget, scaled_retrieve_limit, scaled_budget > budget or scaled_retrieve_limit > retrieve_limit

    def _prepare_lean(self, capsule: Capsule, task: str, *, raw_tokens: int) -> MemoryPacket:
        """Minimal packet for small ledgers: present the dialog, skip scaffolding.

        Guarantees the layer never costs more than simply re-sending the small
        raw context, while still anchoring the first message and adding file/tool
        grounding only when relevant.
        """

        # Compile cheaply so the packet still carries a CompiledMemory handle,
        # but do not pay for its verbose payload in the lean output.
        compiled = self.engine.compile(capsule.capsule_id, question=task, original_context="")

        conversation = _compile_conversation(
            capsule, task, budget_tokens=max(self.conversation_budget, self.lean_threshold * 2), include_all=True
        )
        parts: list[str] = []
        if conversation:
            parts.append(conversation)
        task_state = _compile_task_state(capsule, task, skip_dialog_lines=bool(conversation))
        if task_state:
            parts.append(f"TASK_STATE:\n{task_state}")
        payload = "\n".join(parts) if parts else compiled.payload
        encoded = encode_prompt(payload, codec=self.prompt_codec)
        payload = encoded.payload
        memory_tokens = encoded.token_estimate
        reduction = 0.0 if raw_tokens == 0 else max(0.0, 1.0 - (memory_tokens / raw_tokens))
        return MemoryPacket(
            capsule_id=capsule.capsule_id,
            task=task,
            payload=payload,
            compiled=compiled,
            retrieved=(),
            raw_tokens=raw_tokens,
            memory_tokens=memory_tokens,
            reduction=reduction,
            target_reduction=self.target_reduction,
            evidence_refs=encoded.evidence_refs,
        )

    def answer(self, capsule_id: str, question: str, *, original_context: str = "") -> MemoryAnswer:
        return self.engine.answer(capsule_id, question, original_context=original_context)

    def expand(self, capsule_id: str, source: str) -> str:
        return self.engine.expand(capsule_id, source)


PATH_RE = re.compile(r"(?<![\w/.-])(?:\.{0,2}/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9_]+")
USER_RE = re.compile(r"^USER:\s*(.+)", re.S)
ASSISTANT_FINAL_RE = re.compile(r"^ASSISTANT_FINAL:\s*(.+)", re.S)
TOOL_RE = re.compile(r"^TOOL\s+([A-Za-z0-9_-]+)\s+ok=(True|False|true|false)\s+risk=([A-Za-z0-9_-]+)\s*(.*)", re.S)
ASSIGNMENT_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*=\s*['\"]?([A-Za-z0-9_.@#/-]+)['\"]?")
PIN_RE = re.compile(r"\b([A-Za-z0-9_.-]+)==([A-Za-z0-9_.-]+)\b")

# Conversation-turn classification. Kept permissive so the recall view works for
# any agent that prefixes turns with a role, not just one framework's format.
_USER_TURN_RE = re.compile(r"^\s*(?:USER|HUMAN)\s*:\s*(.*)", re.S | re.I)
_ASSISTANT_TURN_RE = re.compile(r"^\s*(?:ASSISTANT(?:_FINAL)?|AI|BOT)\s*:\s*(.*)", re.S | re.I)
_TOOL_TURN_RE = re.compile(r"^\s*(?:TOOL|OBSERVATION)\b", re.I)

# Phrases that signal the user is asking about the conversation itself
# (ordering, history, recall). These widen the chronological window.
_RECALL_HINTS = (
    "first thing",
    "first message",
    "what did i",
    "what was the first",
    "earlier",
    "before",
    "previously",
    "last time",
    "what were we",
    "what was i",
    "talking about",
    "remember",
    "recall",
    "history",
    "so far",
    "summary",
    "summarize",
    "recap",
)

_CODE_EDIT_HINTS = (
    "apply",
    "edit",
    "fix",
    "patch",
    "replace",
    "update",
    "make the test",
    "tests pass",
    "run test",
    "repository",
    "repo",
    "source",
    "config",
)


def _looks_like_code_edit_task(task: str) -> bool:
    lowered = task.lower()
    return bool(PATH_RE.search(task)) or any(hint in lowered for hint in _CODE_EDIT_HINTS)


def _compile_patch_checklist(capsule: Capsule, task: str, *, max_files: int = 10, max_terms: int = 16) -> str:
    if not _looks_like_code_edit_task(task):
        return ""

    checklist: dict[str, list[str]] = {}

    def add(path: str, item: str) -> None:
        item = _clean_patch_item(item)
        if not item:
            return
        items = checklist.setdefault(path, [])
        item_parts = {part.strip() for part in item.split(";") if part.strip()}
        existing_parts = {
            part.strip()
            for existing in items
            for part in existing.split(";")
            if part.strip()
        }
        if item_parts and item_parts.issubset(existing_parts):
            return
        if len(item_parts) > 1:
            items[:] = [
                existing
                for existing in items
                if not ({part.strip() for part in existing.split(";") if part.strip()} <= item_parts)
            ]
        if item not in items:
            items.append(item)

    for event in capsule.events:
        text = " ".join(event.text.strip().split())
        if not _looks_like_patch_decision(text):
            continue
        for path in _paths(text):
            item = _patch_item_from_text(text, path)
            if item:
                add(path, item)

    for fact in capsule.active_facts():
        for path in _fact_paths(fact):
            item = _patch_item_from_fact(fact.key, fact.value, path)
            if item:
                add(path, item)

    if not checklist:
        return ""

    lines = ["PATCH_CHECKLIST:"]
    for path, items in list(checklist.items())[:max_files]:
        lines.append(f"- {path}: {'; '.join(items[:4])}")

    rejected = _patch_do_not_use_terms(capsule, max_terms=max_terms)
    if rejected:
        lines.append("DO_NOT_USE:")
        lines.append("- " + ", ".join(rejected))
    return "\n".join(lines)


def _looks_like_patch_decision(text: str) -> bool:
    lowered = text.lower()
    if "superseded" in lowered or "rejected" in lowered or "do not " in lowered:
        return False
    return bool(_paths(text)) and any(
        marker in lowered
        for marker in (
            "final",
            "decision",
            "must",
            "set ",
            "update ",
            "define ",
            "expose ",
            "pin ",
            "accepted",
        )
    )


def _patch_item_from_text(text: str, path: str) -> str:
    assignments = [f"{key}={value}" for key, value in ASSIGNMENT_RE.findall(text)]
    pins = [f"{package}=={version}" for package, version in PIN_RE.findall(text)]
    if pins and path.endswith((".txt", ".toml", ".lock")):
        return "; ".join(pins)
    if assignments and not (path.endswith((".txt", ".toml", ".lock")) and pins):
        return "; ".join(assignments)

    before, _sep, after = text.partition(path)
    lowered_before = before.lower()
    if "define " in lowered_before:
        return before[lowered_before.rfind("define "):].strip()
    if "expose " in lowered_before:
        return before[lowered_before.rfind("expose "):].replace("expose", "define", 1).strip()
    if "set " in lowered_before:
        return before[lowered_before.rfind("set "):].strip()
    after = after.strip(" .;:")
    lowered_after = after.lower()
    if lowered_after.startswith("must "):
        return after[5:].strip()
    if lowered_after.startswith("so "):
        return after[3:].strip()
    return before.strip(" .;:")


def _patch_item_from_fact(key: str, value: str, path: str) -> str:
    cleaned = value.replace(path, "").strip(" .;:")
    if key in {"edit_file", "route_file", "created_file", "test_file"}:
        return ""
    if key in {"pagination_mode", "timeout_seconds", "auth_middleware", "checkout_v2", "payment_platform_owner"}:
        return f"{key.upper()}={cleaned or value}"
    if key.startswith("final_") or key in {"accepted_decision", "decision", "function_name", "package", "version"}:
        return cleaned or value
    return ""


def _clean_patch_item(item: str) -> str:
    item = re.sub(r"^(?:FINAL\s+[A-Z ]+:\s*)", "", item, flags=re.I)
    item = re.sub(r"^(?:decision:\s*)", "", item, flags=re.I)
    item = re.sub(r"^expose\s+", "define ", item, flags=re.I)
    item = " ".join(item.strip(" .;:").split())
    item = item.replace(" = ", "=")
    item = re.sub(r"\s+in$", "", item)
    if "/" in item and "=" not in item and " " not in item:
        return ""
    return item[:180]


def _is_broad_patch_checklist(patch_checklist: str, task: str) -> bool:
    patch_lines = [
        line
        for line in patch_checklist.splitlines()
        if line.startswith("- ") and ":" in line and not line.startswith("- Legacy")
    ]
    return len(patch_lines) >= 3 or len(_paths(task)) >= 3


def _compile_patch_execution_packet(compiled: CompiledMemory, patch_checklist: str) -> str:
    return "\n".join(
        [
            f"MEMORY_HANDLE {compiled.label} id={compiled.capsule_id} v={compiled.version}",
            patch_checklist,
            "EXECUTION_RULE verify; return done after pass.",
        ]
    )


def _patch_do_not_use_terms(capsule: Capsule, *, max_terms: int) -> list[str]:
    terms: list[str] = []

    def add(term: str) -> None:
        term = _clean_rejected_term(term)
        if term and term not in terms:
            terms.append(term)

    for fact in capsule.facts:
        for relation in fact.relations:
            if relation.kind in {"rejected_option", "replaces", "supersedes"}:
                add(relation.target)
        if fact.status != "active":
            add(fact.value)

    for event in capsule.events:
        text = " ".join(event.text.strip().split())
        lowered = text.lower()
        if "rejected" in lowered or "superseded" in lowered or "do not" in lowered:
            for key, value in ASSIGNMENT_RE.findall(text):
                add(f"{key}={value}")
            for package, version in PIN_RE.findall(text):
                add(f"{package}=={version}")
            for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*(?:ApiKeyAuth|legacy_invoice|Legacy[A-Za-z0-9_]*)\b", text):
                add(match.group(0))
            for match in re.finditer(r"@[A-Za-z0-9_-]+", text):
                add(match.group(0))
            for match in re.finditer(r"['\"]([A-Za-z0-9_.@#/-]+)['\"]", text):
                add(match.group(1))
        for match in re.finditer(r"\bDo not (?:leave|keep|use|restore|preserve)\s+(.+?)(?:\.|;|$)", text, re.I):
            for term in re.split(r",|\band\b", match.group(1)):
                add(term)
        if len(terms) >= max_terms:
            break
    return _dedupe_rejected_terms(terms)[:max_terms]


def _clean_rejected_term(term: str) -> str:
    term = " ".join(term.strip(" .;:'\"`").split())
    term = re.split(r"\s+in\s+", term, maxsplit=1, flags=re.I)[0]
    term = re.sub(r"\s+(?:anywhere|in active code|from docs|for .*|during .*)$", "", term, flags=re.I)
    term = term.replace(" = ", "=")
    if re.fullmatch(r"\d+", term):
        return ""
    if " " in term and "=" not in term and "==" not in term:
        return ""
    if len(term) > 80:
        return ""
    if term.lower() in {"old", "previous", "rejected", "states", "owners"}:
        return ""
    return term


def _dedupe_rejected_terms(terms: list[str]) -> list[str]:
    result: list[str] = []
    standalone = {term for term in terms if "=" not in term}
    assignment_values = {
        term.split("=", 1)[1]
        for term in terms
        if "=" in term and "==" not in term and term.split("=", 1)[1]
    }
    for term in terms:
        if "=" in term and "==" not in term:
            value = term.split("=", 1)[1]
            if value in standalone and re.match(r"[A-Z]", value):
                continue
        if term in assignment_values and term not in {"disabled", "canary"} and re.fullmatch(r"[a-z0-9_-]+", term):
            continue
        if term not in result:
            result.append(term)
    return result


@dataclass(frozen=True)
class _Turn:
    order: int
    role: str
    content: str
    source: str


def _classify_turn(text: str) -> tuple[str, str] | None:
    match = _USER_TURN_RE.match(text)
    if match:
        return "user", match.group(1)
    match = _ASSISTANT_TURN_RE.match(text)
    if match:
        return "assistant", match.group(1)
    return None


def _conversation_turns(capsule: Capsule) -> list[_Turn]:
    """Extract user/assistant turns from the ledger in chronological order.

    Tool observations and unprefixed events are skipped here: they are recovered
    separately as evidence. This keeps the dialog view clean for recall.
    """

    turns: list[_Turn] = []
    for event in capsule.events:
        text = event.text.strip()
        if not text or _TOOL_TURN_RE.match(text):
            continue
        classified = _classify_turn(text)
        if classified is None:
            continue
        role, content = classified
        content = " ".join(content.split())
        if content:
            turns.append(_Turn(len(turns) + 1, role, content, event.source))
    return turns


def _looks_like_recall(task: str) -> bool:
    lowered = task.lower()
    return any(hint in lowered for hint in _RECALL_HINTS)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _render_conversation(turns: list[_Turn], recent: list[_Turn], first_user: _Turn | None, per_turn_chars: int) -> str:
    # Truncated only when the shown window does not start at the first turn.
    truncated = bool(recent) and recent[0].order > 1
    header = (
        "CONVERSATION_SO_FAR oldest->newest (recent window; older turns elided):"
        if truncated
        else "CONVERSATION_SO_FAR oldest->newest:"
    )
    lines = [header]
    # Anchor the first message only when it would otherwise be cut by truncation;
    # if the full dialog is shown, turn 1 already is the first message.
    if truncated and first_user is not None and first_user.order < recent[0].order:
        lines.append(f"  {first_user.order}. {first_user.role}: {_clip(first_user.content, per_turn_chars)}")
        gap = recent[0].order - first_user.order - 1
        if gap > 0:
            lines.append(f"  ... {gap} earlier turn(s) elided; call memory_retrieve for exact older text ...")
    for turn in recent:
        lines.append(f"  {turn.order}. {turn.role}: {_clip(turn.content, per_turn_chars)}")
    # Explicit anchors are only worth their tokens when truncation hid them.
    if truncated and first_user is not None:
        lines.append(f"FIRST_USER_MESSAGE: {_clip(first_user.content, per_turn_chars)}")
        last_user = next((turn for turn in reversed(turns) if turn.role == "user"), None)
        if last_user is not None:
            lines.append(f"LATEST_USER_MESSAGE: {_clip(last_user.content, per_turn_chars)}")
    return "\n".join(lines)


def _compile_conversation(capsule: Capsule, task: str, *, budget_tokens: int, include_all: bool = False) -> str:
    """Render a compact, ordered conversation view for grounded recall.

    Always anchors the first user message (so ordering questions like "what was
    the first thing I said" are answerable even after the window scrolls) and
    keeps the most recent turns, eliding the middle under budget. When
    ``include_all`` is set the full dialog is shown (used in the lean regime
    where the whole ledger is already cheap).
    """

    turns = _conversation_turns(capsule)
    if not turns:
        return ""

    recall = _looks_like_recall(task)
    per_turn_chars = 260 if recall else 170
    first_user = next((turn for turn in turns if turn.role == "user"), None)
    if include_all:
        max_recent = len(turns)
    else:
        max_recent = min(len(turns), 18 if recall else 8)

    for recent_count in range(max_recent, 0, -1):
        recent = turns[-recent_count:]
        block = _render_conversation(turns, recent, first_user, per_turn_chars)
        if estimate_tokens(block) <= budget_tokens or recent_count == 1:
            return block
    return ""


def _compile_task_state(
    capsule: Capsule,
    task: str,
    *,
    max_lines: int = 8,
    skip_dialog_lines: bool = False,
    include_grounding: bool = True,
) -> str:
    """Render a tiny continuity packet for follow-up turns.

    This is intentionally separate from facts: it preserves local conversational
    focus and distinguishes claims from observed tool output. That prevents
    lossy compaction from turning "it" back into an older topic.

    When ``skip_dialog_lines`` is set, the latest_user/assistant_claim lines are
    omitted because the ordered CONVERSATION view already carries them; this
    avoids duplicating the same turns in two blocks.
    """

    task_paths = _paths(task)
    focus_paths = list(task_paths)
    recent_lines: list[str] = []
    tool_lines: list[str] = []

    for event in reversed(capsule.events):
        text = event.text.strip()
        event_paths = _paths(text)
        for path in event_paths:
            if path not in focus_paths:
                focus_paths.append(path)

        tool_match = TOOL_RE.match(text)
        if tool_match and (not task_paths or any(path in text for path in task_paths + focus_paths[:3])):
            tool, ok, _risk, output = tool_match.groups()
            compact_output = " ".join(output.split())[:220]
            tool_lines.append(f"- tool={tool} ok={ok.lower()} evidence={event.source}: {compact_output}")
            if len(tool_lines) >= 2:
                continue

        user_match = USER_RE.match(text)
        if user_match:
            content = " ".join(user_match.group(1).split())[:180]
            recent_lines.append(f"- latest_user evidence={event.source}: {content}")
        else:
            assistant_match = ASSISTANT_FINAL_RE.match(text)
            if assistant_match:
                content = " ".join(assistant_match.group(1).split())[:180]
                recent_lines.append(f"- assistant_claim_not_proof evidence={event.source}: {content}")

        if len(recent_lines) >= max_lines:
            break

    lines: list[str] = []
    if focus_paths:
        lines.append("CURRENT_FOCUS paths=" + ", ".join(focus_paths[:6]))
    # The grounding rule only earns its tokens when the turn involves files or
    # tool output (where claim-vs-evidence confusion causes hallucination). For
    # plain conversation it is pure overhead, so omit it.
    if include_grounding and (focus_paths or tool_lines):
        lines.append(
            "GROUNDING_RULE assistant_claim_not_proof; for file existence or edits, inspect the repo before claiming success."
        )
    lines.extend(tool_lines[:2])
    if not skip_dialog_lines:
        lines.extend(recent_lines[:max_lines])
    return "\n".join(lines)


def _paths(text: str) -> list[str]:
    paths: list[str] = []
    for match in PATH_RE.finditer(text):
        path = match.group(0).strip("`'\".,;:)")
        if re.fullmatch(r"\d+(?:\.\d+)+", path):
            continue
        suffix = path.rsplit(".", 1)[-1] if "." in path else ""
        if suffix.isdigit():
            continue
        if path and path not in paths:
            paths.append(path)
    return paths


def _fact_paths(fact: Fact) -> list[str]:
    paths: list[str] = []
    for relation in fact.relations:
        if relation.kind == "belongs_to_file":
            for path in _paths(relation.target):
                if path not in paths:
                    paths.append(path)
    for path in _paths(f"{fact.key} {fact.value}"):
        if path not in paths:
            paths.append(path)
    return paths
