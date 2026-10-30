"""Long-term and general chatbot memory benchmark.

This benchmark is intentionally separate from the coding-agent benchmark. It
tests whether a memory layer retrieves the right personal/chat facts across
sessions, suppresses stale preferences, avoids privacy leakage, respects
forgetting, and handles temporal preference updates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARKS = Path(__file__).resolve().parent
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from agent_memory_adapters import PreparedContext, estimate_tokens, make_adapter
from pricing import estimate_ingest_cost
from scoring_utils import contains_fact
from variant_pools import VariantSelector, CODING_LANGUAGES, TRAVEL_DESTINATIONS, DIET_PREFERENCES, APP_NAMES, NOVEL_ELEMENTS, PHONE_NUMBERS, MEDICAL_TOPICS


FAMILIES = (
    "personal_preference",
    "preference_update",
    "episodic_recall",
    "conversational_continuity",
    "privacy_boundary",
    "deletion_forgetting",
    "temporal_preference",
    "general_chat_mixed",
    "multi_topic_preference",
    "contradiction_resolution",
    "long_term_context",
    "implicit_preference",
    "scope_isolation",
    "correction_chain",
    "partial_update",
    "ambiguous_recall",
)


@dataclass(frozen=True)
class ChatMemoryTask:
    task_id: str
    family: str
    prompt: str
    sessions: tuple[tuple[str, ...], ...]
    required_in_context: tuple[str, ...]
    stale_in_context: tuple[str, ...] = ()
    privacy_forbidden: tuple[str, ...] = ()
    deletion_forbidden: tuple[str, ...] = ()
    expected_unknown: bool = False
    variant: int = 0


@dataclass(frozen=True)
class ChatRecallScore:
    required_found: int
    required_total: int
    stale_found: int
    stale_total: int
    privacy_found: int
    privacy_total: int
    deletion_found: int
    deletion_total: int
    payload_tokens: int
    raw_tokens: int

    @property
    def recall(self) -> float:
        return self.required_found / max(1, self.required_total)

    @property
    def stale_penalty(self) -> float:
        return self.stale_found / max(1, self.stale_total)

    @property
    def privacy_penalty(self) -> float:
        return self.privacy_found / max(1, self.privacy_total)

    @property
    def deletion_penalty(self) -> float:
        return self.deletion_found / max(1, self.deletion_total)

    @property
    def context_success(self) -> bool:
        return (
            self.required_found == self.required_total
            and self.stale_found == 0
            and self.privacy_found == 0
            and self.deletion_found == 0
        )

    @property
    def score(self) -> float:
        return (
            self.recall
            * max(0.0, 1.0 - self.stale_penalty)
            * max(0.0, 1.0 - self.privacy_penalty)
            * max(0.0, 1.0 - self.deletion_penalty)
        )

    @property
    def token_efficiency(self) -> float:
        if self.raw_tokens <= 0:
            return 0.0
        return max(0.0, 1.0 - self.payload_tokens / self.raw_tokens)

    def to_dict(self) -> dict:
        return {
            "recall": round(self.recall, 4),
            "precision_penalty": round(max(self.stale_penalty, self.privacy_penalty, self.deletion_penalty), 4),
            "stale_penalty": round(self.stale_penalty, 4),
            "privacy_penalty": round(self.privacy_penalty, 4),
            "deletion_penalty": round(self.deletion_penalty, 4),
            "score": round(self.score, 4),
            "token_efficiency": round(self.token_efficiency, 4),
            "required_found": self.required_found,
            "required_total": self.required_total,
            "stale_found": self.stale_found,
            "stale_total": self.stale_total,
            "privacy_found": self.privacy_found,
            "privacy_total": self.privacy_total,
            "deletion_found": self.deletion_found,
            "deletion_total": self.deletion_total,
            "payload_tokens": self.payload_tokens,
            "raw_tokens": self.raw_tokens,
        }


def run_chat_memory_harness(
    tasks: list[ChatMemoryTask],
    *,
    adapters: tuple[str, ...] | list[str] = ("mem0_real_async", "capsule", "bm25_rag", "langgraph_real", "no_memory"),
    repeats: int = 1,
    randomize_adapter_order: bool = False,
    seed: int = 1337,
    progress: bool = False,
) -> dict:
    adapter_names = list(adapters)
    total = len(tasks) * len(adapter_names) * max(1, repeats)
    completed = 0
    results: list[dict] = []
    started = time.monotonic()
    rng = random.Random(seed)

    for repeat in range(max(1, repeats)):
        repeat_tasks = list(tasks)
        if randomize_adapter_order:
            rng.shuffle(repeat_tasks)
        for task in repeat_tasks:
            run_adapters = list(adapter_names)
            if randomize_adapter_order:
                rng.shuffle(run_adapters)
            for adapter_name in run_adapters:
                completed += 1
                if progress:
                    _progress(f"[{completed}/{total}] repeat={repeat + 1}/{max(1, repeats)} task={task.task_id} adapter={adapter_name}")

                adapter = make_adapter(adapter_name)
                adapter.reset(task.task_id)
                ingest_start = time.monotonic()
                for session in task.sessions:
                    for event in session:
                        adapter.ingest_event(event, scope="chat")
                flush_ingest = getattr(adapter, "flush_ingest", None)
                if callable(flush_ingest):
                    flush_ingest()
                ingest_duration = time.monotonic() - ingest_start

                with TemporaryDirectory() as tmpdir:
                    prepared = adapter.prepare_context(task.prompt, Path(tmpdir))
                raw_tokens = sum(estimate_tokens(event) for session in task.sessions for event in session)
                score = _eval_context(task, prepared.payload, raw_tokens)
                row = _row(task, adapter.name, prepared, score, raw_tokens, ingest_duration, repeat)
                results.append(row)
                adapter.close()

                if progress:
                    elapsed = time.monotonic() - started
                    _progress(
                        f"[{completed}/{total}] done recall={score.recall:.2f} "
                        f"stale={score.stale_penalty:.2f} privacy={score.privacy_penalty:.2f} "
                        f"delete={score.deletion_penalty:.2f} ctxok={score.context_success} "
                        f"ingest={ingest_duration:.1f}s elapsed={elapsed:.1f}s"
                    )

    return _summarize(results, adapter_names=adapter_names)


def load_tasks(*, families: str | Iterable[str] = "all", cases_per_family: int = 1) -> list[ChatMemoryTask]:
    if isinstance(families, str):
        selected = list(FAMILIES) if families == "all" else [item.strip() for item in families.split(",") if item.strip()]
    else:
        selected = list(families)
    unknown = sorted(set(selected) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown families: {', '.join(unknown)}")
    return [_build_task(family, variant) for family in selected for variant in range(cases_per_family)]


def _build_task(family: str, variant: int) -> ChatMemoryTask:
    builders = {
        "personal_preference": _personal_preference,
        "preference_update": _preference_update,
        "episodic_recall": _episodic_recall,
        "conversational_continuity": _conversational_continuity,
        "privacy_boundary": _privacy_boundary,
        "deletion_forgetting": _deletion_forgetting,
        "temporal_preference": _temporal_preference,
        "general_chat_mixed": _general_chat_mixed,
        "multi_topic_preference": _multi_topic_preference,
        "contradiction_resolution": _contradiction_resolution,
        "long_term_context": _long_term_context,
        "implicit_preference": _implicit_preference,
        "scope_isolation": _scope_isolation,
        "correction_chain": _correction_chain,
        "partial_update": _partial_update,
        "ambiguous_recall": _ambiguous_recall,
    }
    builder = builders.get(family)
    if builder is None:
        raise ValueError(f"unknown family {family!r}")
    return builder(variant)


def _personal_preference(variant: int) -> ChatMemoryTask:
    vs = VariantSelector("personal_preference", variant)
    old_lang, new_lang = vs.pick(CODING_LANGUAGES)
    sessions = (
        (
            "USER: Please remember that I prefer concise answers.",
            "ASSISTANT: I will keep answers concise.",
            f"USER: I prefer {new_lang} examples for frontend topics.",
            "USER: I dislike emojis in technical answers.",
            *vs.chat_noise("preference planning", 3 + variant),
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_personal_preference_{variant:03d}",
        family="personal_preference",
        prompt="Explain debounce in the user's preferred answer style.",
        sessions=sessions,
        required_in_context=("concise", new_lang, "dislike emojis"),
        stale_in_context=(f"{old_lang} examples", "use emojis"),
        variant=variant,
    )


def _preference_update(variant: int) -> ChatMemoryTask:
    vs = VariantSelector("preference_update", variant)
    old_lang, new_lang = vs.pick(CODING_LANGUAGES)
    sessions = (
        (
            f"USER: Remember: I prefer {old_lang} examples.",
            f"ASSISTANT: Noted, {old_lang} examples.",
            *vs.chat_noise("old preference", 2 + variant),
        ),
        (
            f"USER: Update that preference. I now prefer {new_lang} examples, not {old_lang} examples.",
            f"ASSISTANT: Current preference is {new_lang} examples. {old_lang} examples are old.",
            "USER: Also stop being terse; give medium-detail explanations.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_preference_update_{variant:03d}",
        family="preference_update",
        prompt="Explain debounce using the user's current coding-language preference.",
        sessions=sessions,
        required_in_context=(f"{new_lang} examples", "medium-detail"),
        stale_in_context=(f"{old_lang} examples", "terse"),
        variant=variant,
    )


def _episodic_recall(variant: int) -> ChatMemoryTask:
    vs = VariantSelector("episodic_recall", variant)
    city, hotel = vs.pick(TRAVEL_DESTINATIONS)
    sessions = (
        (
            f"USER: I'm planning a trip to {city} in March.",
            f"USER: I chose {hotel} for the stay.",
            f"ASSISTANT: {hotel} saved as the selected hotel.",
            *vs.chat_noise("travel brainstorm", 3 + variant),
        ),
        (
            "USER: I booked a walking tour on April 12.",
            "USER: The restaurant shortlist includes two local favorites.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_episodic_recall_{variant:03d}",
        family="episodic_recall",
        prompt=f"What hotel did the user choose for {city}?",
        sessions=sessions,
        required_in_context=(hotel,),
        stale_in_context=("Park Hyatt", "Ritz-Carlton"),
        variant=variant,
    )


def _conversational_continuity(variant: int) -> ChatMemoryTask:
    vs = VariantSelector("conversational_continuity", variant)
    protagonist, magic, memory_element = vs.pick(NOVEL_ELEMENTS)
    sessions = (
        (
            "USER: I am outlining a fantasy novel.",
            f"USER: The protagonist is named {protagonist}.",
            f"USER: The magic system is based on {magic}.",
            *vs.chat_noise("novel worldbuilding", 4 + variant),
        ),
        (
            f"USER: Chapter 1 ends when {protagonist} realizes each spell costs one {memory_element}.",
            f"ASSISTANT: Continue chapter 2 from {protagonist} and {magic} magic.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_conversation_continuity_{variant:03d}",
        family="conversational_continuity",
        prompt="Continue brainstorming chapter 2 of the user's novel.",
        sessions=sessions,
        required_in_context=(protagonist, magic, memory_element),
        stale_in_context=("protagonist is Rowan", "fire magic"),
        variant=variant,
    )


def _privacy_boundary(variant: int) -> ChatMemoryTask:
    vs = VariantSelector("privacy_boundary", variant)
    sensitive, work_task = vs.pick(MEDICAL_TOPICS)
    sessions = (
        (
            f"USER: I have {sensitive} and prefer not to bring it into unrelated work tasks.",
            f"ASSISTANT: I will not mention {sensitive} unless it is relevant.",
            *vs.chat_noise("sensitive personal note", 3 + variant),
        ),
        (
            f"USER: Help me draft a {work_task}.",
            f"ASSISTANT: The {work_task} should stay workplace-focused.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_privacy_boundary_{variant:03d}",
        family="privacy_boundary",
        prompt=f"Draft a {work_task}.",
        sessions=sessions,
        required_in_context=(work_task,),
        privacy_forbidden=(sensitive,),
        variant=variant,
    )


def _deletion_forgetting(variant: int) -> ChatMemoryTask:
    vs = VariantSelector("deletion_forgetting", variant)
    phone = vs.pick(PHONE_NUMBERS)
    sessions = (
        (
            f"USER: Remember that my phone number is {phone}.",
            f"ASSISTANT: I saved phone number {phone}.",
        ),
        (
            f"USER: Forget my phone number. Do not retain or repeat {phone}.",
            "ASSISTANT: I will treat the phone number as deleted. If asked, answer UNKNOWN.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_deletion_forgetting_{variant:03d}",
        family="deletion_forgetting",
        prompt="What is the user's phone number?",
        sessions=sessions,
        required_in_context=("UNKNOWN",),
        deletion_forbidden=(phone,),
        expected_unknown=True,
        variant=variant,
    )


def _temporal_preference(variant: int) -> ChatMemoryTask:
    vs = VariantSelector("temporal_preference", variant)
    old_diet, new_diet = vs.pick(DIET_PREFERENCES)
    sessions = (
        (
            f"2026-01-10 USER: I prefer {old_diet} recipes.",
            f"2026-02-15 USER: Keep dinner ideas {old_diet} for now.",
            *vs.chat_noise("meal planning", 2 + variant),
        ),
        (
            f"2026-04-01 USER: My diet changed. I am now {new_diet}, not {old_diet}.",
            f"2026-06-01 USER: For current dinner ideas, follow {new_diet}.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_temporal_preference_{variant:03d}",
        family="temporal_preference",
        prompt="Suggest dinner ideas using the user's current diet preference.",
        sessions=sessions,
        required_in_context=(new_diet, "current dinner"),
        stale_in_context=(old_diet,),
        variant=variant,
    )


def _general_chat_mixed(variant: int) -> ChatMemoryTask:
    vs = VariantSelector("general_chat_mixed", variant)
    old_app, new_app = vs.pick(APP_NAMES)
    _, lang = vs.pick(CODING_LANGUAGES)
    sessions = (
        (
            f"USER: I prefer concise answers, {lang} examples, and no emojis.",
            f"USER: I am working on a personal CRM app called {old_app}.",
            *vs.chat_noise("mixed unrelated chat", 5 + variant),
        ),
        (
            f"USER: Correction: the app is now called {new_app}, not {old_app}.",
            f"USER: Keep {lang} examples as my current preference.",
            f"ASSISTANT: Current app name is {new_app}.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_general_mixed_{variant:03d}",
        family="general_chat_mixed",
        prompt="Help me write a short product blurb for my app using my current preferences.",
        sessions=sessions,
        required_in_context=(new_app, f"{lang} examples", "concise", "no emojis"),
        stale_in_context=(old_app,),
        variant=variant,
    )


# ---------------------------------------------------------------------------
# New families (Task D)
# ---------------------------------------------------------------------------

def _multi_topic_preference(variant: int) -> ChatMemoryTask:
    """Preferences across food, code, and communication domains."""
    vs = VariantSelector("multi_topic_preference", variant)
    _, lang = vs.pick(CODING_LANGUAGES)
    _, diet = vs.pick(DIET_PREFERENCES)
    sessions = (
        (
            f"USER: For coding, always use {lang} examples.",
            f"USER: For meal suggestions, I follow a {diet} diet.",
            "USER: I prefer bullet-point answers over paragraphs.",
            *vs.chat_noise("multi-topic chat", 3 + variant),
        ),
        (
            "USER: Also, I communicate asynchronously — no real-time meetings.",
            f"USER: Confirm: {lang} for code, {diet} for food, bullets for format.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_multi_topic_pref_{variant:03d}",
        family="multi_topic_preference",
        prompt="The user wants a meal plan and a code snippet. Use their preferences for both.",
        sessions=sessions,
        required_in_context=(lang, diet, "bullet"),
        stale_in_context=(),
        variant=variant,
    )


def _contradiction_resolution(variant: int) -> ChatMemoryTask:
    """Conflicting info within same session — latest statement wins."""
    vs = VariantSelector("contradiction_resolution", variant)
    old_app, new_app = vs.pick(APP_NAMES)
    sessions = (
        (
            f"USER: The project name is {old_app}.",
            *vs.chat_noise("project discussion", 3),
            f"USER: Actually, scratch that. The project name is {new_app}, not {old_app}.",
            f"ASSISTANT: Updated: project name is {new_app}.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_contradiction_{variant:03d}",
        family="contradiction_resolution",
        prompt="What is the user's project name?",
        sessions=sessions,
        required_in_context=(new_app,),
        stale_in_context=(old_app,),
        variant=variant,
    )


def _long_term_context(variant: int) -> ChatMemoryTask:
    """5+ sessions with gradual project evolution."""
    vs = VariantSelector("long_term_context", variant)
    _, lang = vs.pick(CODING_LANGUAGES)
    _, app = vs.pick(APP_NAMES)
    sessions = (
        (
            f"USER: Starting a new project called {app} in {lang}.",
            *vs.chat_noise("project kickoff", 2),
        ),
        (
            f"USER: {app} now has a REST API with 3 endpoints.",
            *vs.chat_noise("api development", 2),
        ),
        (
            f"USER: Added authentication to {app} using JWT.",
            *vs.chat_noise("auth work", 2),
        ),
        (
            f"USER: {app} deployment is on AWS Lambda.",
            *vs.chat_noise("deployment setup", 2),
        ),
        (
            f"USER: Current status: {app} has REST API, JWT auth, on Lambda, written in {lang}.",
            f"ASSISTANT: {app} summary noted.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_long_term_ctx_{variant:03d}",
        family="long_term_context",
        prompt=f"Summarize the current state of my {app} project.",
        sessions=sessions,
        required_in_context=(app, lang, "JWT", "Lambda"),
        stale_in_context=(),
        variant=variant,
    )


def _implicit_preference(variant: int) -> ChatMemoryTask:
    """Behavior-inferred preferences — not explicitly stated."""
    vs = VariantSelector("implicit_preference", variant)
    _, lang = vs.pick(CODING_LANGUAGES)
    sessions = (
        (
            f"USER: Here's my code:\n```{lang.lower()}\nfunction greet() {{ return 'hello'; }}\n```",
            f"ASSISTANT: I see you're using {lang}.",
            f"USER: Can you add error handling to this {lang} function?",
            *vs.chat_noise("coding session", 3 + variant),
        ),
        (
            f"USER: Another {lang} question — how do I write unit tests?",
            f"ASSISTANT: Continuing with {lang} examples as before.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_implicit_pref_{variant:03d}",
        family="implicit_preference",
        prompt="Write a utility function for the user in their preferred language.",
        sessions=sessions,
        required_in_context=(lang,),
        stale_in_context=(),
        variant=variant,
    )


def _scope_isolation(variant: int) -> ChatMemoryTask:
    """Work vs personal context must not bleed."""
    vs = VariantSelector("scope_isolation", variant)
    city, hotel = vs.pick(TRAVEL_DESTINATIONS)
    _, app = vs.pick(APP_NAMES)
    sessions = (
        (
            f"USER: [PERSONAL] I'm planning a vacation to {city}. Hotel: {hotel}.",
            *vs.chat_noise("personal planning", 2),
        ),
        (
            f"USER: [WORK] Help me with the {app} architecture review.",
            f"ASSISTANT: Focusing on {app} work context.",
            *vs.chat_noise("work discussion", 2),
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_scope_isolation_{variant:03d}",
        family="scope_isolation",
        prompt=f"Help me prepare for the {app} architecture review.",
        sessions=sessions,
        required_in_context=(app,),
        privacy_forbidden=(hotel, city),
        variant=variant,
    )


def _correction_chain(variant: int) -> ChatMemoryTask:
    """A -> B -> C -> D corrections; only D survives."""
    vs = VariantSelector("correction_chain", variant)
    destinations = [vs.pick(TRAVEL_DESTINATIONS, idx=i) for i in range(4)]
    sessions = tuple(
        (
            f"USER: {'Actually, change the' if i > 0 else 'My'} vacation destination "
            f"{'from ' + destinations[i-1][0] + ' ' if i > 0 else ''}to {destinations[i][0]}.",
            f"ASSISTANT: Updated destination to {destinations[i][0]}.",
        )
        for i in range(4)
    )
    final_city = destinations[3][0]
    stale_cities = tuple(destinations[i][0] for i in range(3))
    return ChatMemoryTask(
        task_id=f"chat_correction_chain_{variant:03d}",
        family="correction_chain",
        prompt="What is the user's current vacation destination?",
        sessions=sessions,
        required_in_context=(final_city,),
        stale_in_context=stale_cities,
        variant=variant,
    )


def _partial_update(variant: int) -> ChatMemoryTask:
    """Change one fact; adjacent facts must stay stable."""
    vs = VariantSelector("partial_update", variant)
    _, lang = vs.pick(CODING_LANGUAGES)
    old_app, new_app = vs.pick(APP_NAMES)
    sessions = (
        (
            f"USER: I code in {lang}, my project is {old_app}, and I use VS Code.",
            *vs.chat_noise("setup discussion", 2 + variant),
        ),
        (
            f"USER: Rename the project from {old_app} to {new_app}. Everything else stays the same.",
            f"ASSISTANT: Project renamed to {new_app}. {lang} and VS Code unchanged.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_partial_update_{variant:03d}",
        family="partial_update",
        prompt="Describe the user's current development setup.",
        sessions=sessions,
        required_in_context=(new_app, lang, "VS Code"),
        stale_in_context=(old_app,),
        variant=variant,
    )


def _ambiguous_recall(variant: int) -> ChatMemoryTask:
    """Similar-sounding facts that must not be confused."""
    vs = VariantSelector("ambiguous_recall", variant)
    work_app, personal_app = vs.pick(APP_NAMES)
    sessions = (
        (
            f"USER: My work project is {work_app}. My side project is {personal_app}.",
            f"ASSISTANT: Noted: {work_app} = work, {personal_app} = side project.",
            *vs.chat_noise("project chat", 3 + variant),
        ),
        (
            f"USER: The deadline for {work_app} is next Friday. {personal_app} has no deadline.",
        ),
    )
    return ChatMemoryTask(
        task_id=f"chat_ambiguous_recall_{variant:03d}",
        family="ambiguous_recall",
        prompt=f"What is the deadline for {work_app}?",
        sessions=sessions,
        required_in_context=(work_app, "Friday"),
        stale_in_context=(),
        variant=variant,
    )


def _eval_context(task: ChatMemoryTask, payload: str, raw_tokens: int) -> ChatRecallScore:
    required_found = sum(1 for item in task.required_in_context if contains_fact(payload, item))
    stale_found = sum(1 for item in task.stale_in_context if contains_fact(payload, item))
    privacy_found = sum(1 for item in task.privacy_forbidden if contains_fact(payload, item))
    deletion_found = sum(1 for item in task.deletion_forbidden if contains_fact(payload, item))
    return ChatRecallScore(
        required_found=required_found,
        required_total=len(task.required_in_context),
        stale_found=stale_found,
        stale_total=len(task.stale_in_context),
        privacy_found=privacy_found,
        privacy_total=len(task.privacy_forbidden),
        deletion_found=deletion_found,
        deletion_total=len(task.deletion_forbidden),
        payload_tokens=estimate_tokens(payload),
        raw_tokens=raw_tokens,
    )


def _row(
    task: ChatMemoryTask,
    adapter: str,
    prepared: PreparedContext,
    score: ChatRecallScore,
    raw_tokens: int,
    ingest_duration: float,
    repeat: int,
) -> dict:
    return {
        "task_id": task.task_id,
        "family": task.family,
        "adapter": adapter,
        "mode": "chat_retrieval",
        "repeat": repeat,
        "success": None,
        "context_success": score.context_success,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "memory_payload_tokens": prepared.memory_payload_tokens,
        "raw_context_tokens": raw_tokens,
        "prepared_context_tokens": prepared.token_estimate,
        "token_savings": round(max(0.0, 1.0 - prepared.token_estimate / max(1, raw_tokens)), 3),
        "duration_seconds": 0.0,
        "ingest_duration_seconds": ingest_duration,
        "final_message": "CHAT_RETRIEVAL_ONLY",
        "recall_score": score.to_dict(),
        "sessions": len(task.sessions),
        "session_events": sum(len(session) for session in task.sessions),
        "ingest_cost_usd": round(estimate_ingest_cost(adapter, raw_tokens), 6),
        "agent_cost_usd": 0.0,
        "total_cost_usd": round(estimate_ingest_cost(adapter, raw_tokens), 6),
    }


def _summarize(results: list[dict], *, adapter_names: list[str]) -> dict:
    by_adapter: dict[str, list[dict]] = {name: [] for name in adapter_names}
    for row in results:
        by_adapter.setdefault(row["adapter"], []).append(row)

    scores: dict[str, dict] = {}
    for adapter, items in by_adapter.items():
        n = max(1, len(items))
        scores[adapter] = {
            "tasks": len(items),
            "success_rate": None,
            "context_success_rate": round(sum(1 for item in items if item.get("context_success")) / n, 4),
            "avg_recall": round(sum(item["recall_score"]["recall"] for item in items) / n, 4),
            "avg_stale_penalty": round(sum(item["recall_score"]["stale_penalty"] for item in items) / n, 4),
            "avg_privacy_penalty": round(sum(item["recall_score"]["privacy_penalty"] for item in items) / n, 4),
            "avg_deletion_penalty": round(sum(item["recall_score"]["deletion_penalty"] for item in items) / n, 4),
            "avg_cross_session_score": round(sum(item["recall_score"]["score"] for item in items) / n, 4),
            "avg_token_efficiency": round(sum(item["recall_score"]["token_efficiency"] for item in items) / n, 4),
            "avg_ingest_duration_seconds": round(sum(item.get("ingest_duration_seconds", 0.0) for item in items) / n, 2),
            "avg_total_tokens": 0,
            "avg_raw_context_tokens": round(sum(item.get("raw_context_tokens", 0) for item in items) / n),
            "avg_prepared_context_tokens": round(sum(item.get("prepared_context_tokens", 0) for item in items) / n),
            "avg_token_savings": round(sum(item.get("token_savings", 0.0) for item in items) / n, 3),
            "avg_total_cost_usd": round(sum(item.get("total_cost_usd", 0.0) for item in items) / n, 6),
        }

    ranked = sorted(scores.items(), key=lambda pair: pair[1]["avg_cross_session_score"], reverse=True)
    best_adapter, best_score = ranked[0] if ranked else ("none", {})
    capsule = scores.get("capsule", {})
    mem0_name = "mem0_real" if "mem0_real" in scores else "mem0_real_async" if "mem0_real_async" in scores else None
    mem0 = scores.get(mem0_name) if mem0_name else None
    return {
        "summary": {
            "benchmark": "chat_memory",
            "adapters": adapter_names,
            "adapter_scores": scores,
            "best_adapter": best_adapter,
            "best_cross_session_score": best_score.get("avg_cross_session_score", 0.0),
            "capsule_cross_session_score": capsule.get("avg_cross_session_score"),
            "mem0_baseline": mem0_name,
            "capsule_beats_mem0_real": (
                capsule.get("avg_cross_session_score", 0.0) > mem0.get("avg_cross_session_score", 0.0)
                if mem0 and capsule else None
            ),
        },
        "results": results,
    }


def _write_csv(report: dict, path: Path) -> None:
    scores = report["summary"]["adapter_scores"]
    lines = ["adapter,ctxok,recall,stale,privacy,deletion,score,raw_tokens,prepared_tokens,token_savings,ingest_seconds"]
    for adapter in report["summary"]["adapters"]:
        s = scores.get(adapter, {})
        lines.append(
            f"{adapter},{s.get('context_success_rate', 0):.4f},{s.get('avg_recall', 0):.4f},"
            f"{s.get('avg_stale_penalty', 0):.4f},{s.get('avg_privacy_penalty', 0):.4f},"
            f"{s.get('avg_deletion_penalty', 0):.4f},{s.get('avg_cross_session_score', 0):.4f},"
            f"{s.get('avg_raw_context_tokens', 0):.0f},{s.get('avg_prepared_context_tokens', 0):.0f},"
            f"{s.get('avg_token_savings', 0):.4f},{s.get('avg_ingest_duration_seconds', 0):.2f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(report: dict, path: Path) -> None:
    scores = report["summary"]["adapter_scores"]
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Adapter & CtxOK & Recall & Stale & Privacy & Delete & Score \\\\",
        "\\hline",
    ]
    for adapter in report["summary"]["adapters"]:
        s = scores.get(adapter, {})
        lines.append(
            f"{adapter.replace('_', '\\_')} & {s.get('context_success_rate', 0):.3f} & "
            f"{s.get('avg_recall', 0):.3f} & {s.get('avg_stale_penalty', 0):.3f} & "
            f"{s.get('avg_privacy_penalty', 0):.3f} & {s.get('avg_deletion_penalty', 0):.3f} & "
            f"{s.get('avg_cross_session_score', 0):.3f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters", default="mem0_real_async,capsule,bm25_rag,langgraph_real,no_memory")
    parser.add_argument("--families", default="all", help=f"Comma-separated families or all. Choices: {', '.join(FAMILIES)}")
    parser.add_argument("--cases-per-family", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--randomize-adapter-order", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json-out", default="chat-memory-report.json")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--latex-out", default="")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    tasks = load_tasks(families=args.families, cases_per_family=args.cases_per_family)
    adapters = [item.strip() for item in args.adapters.split(",") if item.strip()]
    report = run_chat_memory_harness(
        tasks,
        adapters=adapters,
        repeats=args.repeats,
        randomize_adapter_order=args.randomize_adapter_order,
        seed=args.seed,
        progress=args.progress,
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        _progress(f"Report written to {args.json_out}")
    else:
        print(json.dumps(report, indent=2))
    if args.csv_out:
        _write_csv(report, Path(args.csv_out))
        _progress(f"CSV table written to {args.csv_out}")
    if args.latex_out:
        _write_latex(report, Path(args.latex_out))
        _progress(f"LaTeX table written to {args.latex_out}")

    print("\n=== Chat Memory Benchmark Results ===")
    print(f"{'Adapter':<20} {'CtxOK':>8} {'Recall':>8} {'Stale':>8} {'Privacy':>8} {'Delete':>8} {'Score':>8} {'Savings':>8}")
    print("-" * 86)
    for adapter in adapters:
        s = report["summary"]["adapter_scores"].get(adapter, {})
        print(
            f"{adapter:<20} {s.get('context_success_rate', 0):>8.3f} "
            f"{s.get('avg_recall', 0):>8.3f} {s.get('avg_stale_penalty', 0):>8.3f} "
            f"{s.get('avg_privacy_penalty', 0):>8.3f} {s.get('avg_deletion_penalty', 0):>8.3f} "
            f"{s.get('avg_cross_session_score', 0):>8.3f} {s.get('avg_token_savings', 0):>8.3f}"
        )
    print(f"\nBest adapter: {report['summary']['best_adapter']} (score={report['summary']['best_cross_session_score']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
