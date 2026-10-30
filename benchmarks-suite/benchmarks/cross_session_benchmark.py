"""Cross-session memory benchmark.

Tests what mem0 is actually designed for: retaining and superseding facts
across *multiple work sessions*, not compressing a single event stream.

The benchmark simulates N past sessions (decisions that evolved over days),
then runs an agent task that must recall the *final* correct facts using only
the adapter's retrieval — no raw event replay.

This directly measures two things non-persistent adapters cannot do:
  1. Fact supersession: old decision is replaced; only the new one should surface.
  2. Cross-session recall: relevant facts from any past session are retrievable.

Adapters
--------
mem0_real           Real Mem0 SDK, infer=True. LLM extracts and merges facts
                    across sessions. The canonical cross-session use-case.
mem0_real_fast      Real Mem0 SDK, infer=False. Plain vector store, no LLM at
                    ingest. Tests whether the LLM cost pays off.
capsule             SemanticCapsuleMemory. Local SQLite, no ingest LLM.
bm25_rag            Sparse retrieval over ALL session events concatenated.
                    Non-persistent upper bound: it sees everything.
no_memory           Lower bound: empty context.

Usage
-----
python3 benchmarks/cross_session_benchmark.py \\
    --adapters mem0_real,mem0_real_fast,capsule,bm25_rag,no_memory \\
    --families all \\
    --json-out cross-session-report.json \\
    --progress

Cross-run persistence (true multi-day simulation)
-------------------------------------------------
Pass --persistent-dir to reuse the same vector store across benchmark runs.
Run once to pre-load sessions, then run again (same dir) to test retrieval
without re-ingesting — this is the closest to real mem0 production usage.

    python3 benchmarks/cross_session_benchmark.py \\
        --adapters mem0_real,capsule \\
        --persistent-dir ./cs-store \\
        --json-out cross-session-report.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
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

from agent_memory_adapters import (
    BM25RagAdapter,
    LangGraphStateAdapter,
    NoMemoryAdapter,
    OpenAIEmbeddingRagAdapter,
    RealLangGraphAdapter,
    RealMem0Adapter,
    RealMem0AsyncAdapter,
    RealMem0FastAdapter,
    SemanticCapsuleAdapter,
    estimate_tokens,
)
from pricing import cost_usd, estimate_ingest_cost
from scoring_utils import contains_fact
from variant_pools import VariantSelector

try:
    from agent_task_harness import (
        OpenAIToolAgent,
        AnthropicToolAgent,
        _create_repo_fixture,
        _make_agent,
        _run_command,
        _score_result,
        _snapshot_files,
        _tool_agent_prompt,
        AgentTask,
    )
except ImportError as exc:
    raise SystemExit(f"Could not import agent_task_harness: {exc}") from exc

# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

FAMILIES = (
    "auth_decision_evolution",
    "config_iteration",
    "route_migration",
    "feature_flag_rollout",
    "service_ownership",
    "export_jobs_config",
    "stale_docs_current_code",
    "dependency_version_pin",
    "unknown_owner_abstention",
    "long_horizon_project_memory",
)


@dataclass(frozen=True)
class CrossSessionTask:
    """A task with pre-ingested session history and a live agent step.

    sessions: past event groups, ingested in order before the agent runs.
              Each inner tuple is one work session (day/PR review/etc).
    required_in_context: strings the adapter's prepared context MUST contain
                         for a correct recall (scored before agent runs).
    stale_in_context: strings that represent superseded/wrong facts.
                      Presence in context is penalised in recall_score.
    """
    task_id: str
    family: str
    prompt: str
    sessions: tuple[tuple[str, ...], ...]
    expected_files: tuple[str, ...]
    test_command: str
    forbidden_strings: tuple[str, ...] = ()
    required_in_context: tuple[str, ...] = ()
    stale_in_context: tuple[str, ...] = ()
    unknown_expected: bool = False
    variant: int = 0

    def as_agent_task(self) -> AgentTask:
        """Convert to AgentTask with no memory_events (retrieval-only run)."""
        return AgentTask(
            task_id=self.task_id,
            family=self.family,
            prompt=self.prompt,
            memory_events=(),         # nothing ingested at run time
            expected_files=self.expected_files,
            test_command=self.test_command,
            forbidden_strings=self.forbidden_strings,
            unknown_expected=self.unknown_expected,
            variant=self.variant,
            repo_files=5,
            repo_noise_lines=10,
        )


# ---------------------------------------------------------------------------
# Adapter factory (cross-session variants with optional persistent storage)
# ---------------------------------------------------------------------------

def make_cross_session_adapter(name: str, store_path: Path | None = None):
    """Return a memory adapter wired for cross-session use."""
    normalized = name.strip().lower().replace("-", "_")
    sp = store_path / normalized if store_path else None
    if normalized == "mem0_real":
        return RealMem0Adapter(top_k=10, infer=True, store_path=sp)
    if normalized == "mem0_real_async":
        return RealMem0AsyncAdapter(top_k=10, infer=True, store_path=sp)
    if normalized == "mem0_real_fast":
        return RealMem0FastAdapter(top_k=10, store_path=sp)
    if normalized == "capsule":
        # Cross-session ledgers are 13–24 events with long evidence hashes;
        # a 280-token budget fits all key PRECISION_KEYS facts comfortably.
        return SemanticCapsuleAdapter(store_path=sp, token_budget=280)
    if normalized == "bm25_rag":
        return BM25RagAdapter(top_k=6)
    if normalized in {"openai_embedding_rag", "embedding_rag", "vector_rag"}:
        adapter = OpenAIEmbeddingRagAdapter(top_k=6)
        adapter.name = normalized
        return adapter
    if normalized == "langgraph_state":
        return LangGraphStateAdapter()
    if normalized == "langgraph_real":
        return RealLangGraphAdapter(top_k=10)
    if normalized == "no_memory":
        return NoMemoryAdapter()
    raise ValueError(
        f"unknown cross-session adapter {name!r}; "
        "choose from: mem0_real, mem0_real_async, mem0_real_fast, capsule, "
        "bm25_rag, openai_embedding_rag, vector_rag, langgraph_state, "
        "langgraph_real, no_memory"
    )


# ---------------------------------------------------------------------------
# Context quality scoring (pre-agent)
# ---------------------------------------------------------------------------

@dataclass
class RecallScore:
    required_found: int
    required_total: int
    stale_found: int
    stale_total: int
    payload_tokens: int
    raw_tokens: int

    @property
    def recall(self) -> float:
        return self.required_found / max(1, self.required_total)

    @property
    def precision_penalty(self) -> float:
        """Fraction of stale strings that leaked into the context."""
        return self.stale_found / max(1, self.stale_total)

    @property
    def score(self) -> float:
        """Combined score: recall minus stale penalty, clipped to [0,1]."""
        return max(0.0, self.recall - self.precision_penalty)

    @property
    def multiplicative_score(self) -> float:
        """Combined score: full recall gets discounted by stale leakage."""
        return self.recall * max(0.0, 1.0 - self.precision_penalty)

    @property
    def token_efficiency(self) -> float:
        if self.raw_tokens <= 0:
            return 0.0
        return max(0.0, 1.0 - self.payload_tokens / self.raw_tokens)

    def to_dict(self, *, score_formula: str = "subtractive") -> dict:
        score = self.multiplicative_score if score_formula == "multiplicative" else self.score
        return {
            "recall": round(self.recall, 4),
            "precision_penalty": round(self.precision_penalty, 4),
            "score": round(score, 4),
            "subtractive_score": round(self.score, 4),
            "multiplicative_score": round(self.multiplicative_score, 4),
            "score_formula": score_formula,
            "token_efficiency": round(self.token_efficiency, 4),
            "required_found": self.required_found,
            "required_total": self.required_total,
            "stale_found": self.stale_found,
            "stale_total": self.stale_total,
            "payload_tokens": self.payload_tokens,
            "raw_tokens": self.raw_tokens,
        }


def _eval_context(task: CrossSessionTask, payload: str, raw_tokens: int) -> RecallScore:
    payload_lower = payload.lower()
    required_found = sum(1 for r in task.required_in_context if contains_fact(payload, r))
    # Only check the ACTIVE_FACTS section for stale detection.
    # Values in TASK_PLAN_HINT, DO_NOT_USE, or RESPONSE_RULE are intentional
    # warnings/context — stale strings there are correct behaviour, not leakage.
    active_parts = payload_lower.split("active_facts:", 1)
    if len(active_parts) > 1:
        active_section = active_parts[1]
        for boundary in ("task_plan_hint:", "do_not_use:", "omitted_facts ", "response_rule "):
            if boundary in active_section:
                active_section = active_section.split(boundary, 1)[0]
                break
    else:
        active_section = payload_lower
    stale_found = sum(1 for s in task.stale_in_context if contains_fact(active_section, s))
    return RecallScore(
        required_found=required_found,
        required_total=len(task.required_in_context),
        stale_found=stale_found,
        stale_total=len(task.stale_in_context),
        payload_tokens=estimate_tokens(payload),
        raw_tokens=raw_tokens,
    )


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_cross_session_harness(
    tasks: list[CrossSessionTask],
    *,
    adapters: tuple[str, ...] | list[str] = ("mem0_real", "capsule", "bm25_rag", "no_memory"),
    agent_name: str = "openai",
    model: str = "gpt-4.1-mini",
    max_steps: int = 10,
    persistent_dir: Path | None = None,
    skip_ingest: bool = False,
    load_only: bool = False,
    retrieval_only: bool = False,
    repeats: int = 1,
    randomize_adapter_order: bool = False,
    score_formula: str = "subtractive",
    seed: int = 1337,
    progress: bool = False,
) -> dict:
    if score_formula not in {"subtractive", "multiplicative"}:
        raise ValueError("score_formula must be 'subtractive' or 'multiplicative'")
    agent = None if load_only or retrieval_only else _make_agent(agent_name, model)
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
            # Build a shared repo fixture reused across adapters for this task.
            with TemporaryDirectory() as tmpdir:
                repo_path = Path(tmpdir) / "repo"
                agent_task = task.as_agent_task()
                _create_repo_fixture(agent_task, repo_path)

                run_adapters = list(adapter_names)
                if randomize_adapter_order:
                    rng.shuffle(run_adapters)
                for adapter_name in run_adapters:
                    completed += 1
                    if progress:
                        _progress(
                            f"[{completed}/{total}] repeat={repeat + 1}/{max(1, repeats)} "
                            f"task={task.task_id} adapter={adapter_name}"
                        )

                    store_path = persistent_dir / task.task_id if persistent_dir else None
                    adapter = make_cross_session_adapter(adapter_name, store_path=store_path)
                    adapter.reset(task.task_id)

                    # --- Phase 1: ingest all sessions sequentially ----------
                    ingest_start = time.monotonic()
                    if skip_ingest:
                        if progress:
                            _progress(f"  skip-ingest: using existing store at {store_path}")
                        ingest_duration = 0.0
                    else:
                        for session_events in task.sessions:
                            for event in session_events:
                                adapter.ingest_event(event, scope="project")
                        flush_ingest = getattr(adapter, "flush_ingest", None)
                        if callable(flush_ingest):
                            flush_ingest()
                        ingest_duration = time.monotonic() - ingest_start

                    if load_only:
                        adapter.close()
                        if progress:
                            _progress(f"  loaded {sum(len(s) for s in task.sessions)} events into store")
                        continue

                    # --- Phase 2: score context quality (pre-agent) ----------
                    repo_copy = Path(tmpdir) / f"repo_{adapter_name}"
                    if repo_copy.exists():
                        shutil.rmtree(repo_copy)
                    shutil.copytree(repo_path, repo_copy)

                    # Raw session tokens = what you'd pay injecting ALL events every step
                    raw_session_tokens = sum(
                        estimate_tokens(event)
                        for session in task.sessions
                        for event in session
                    )
                    prepared = adapter.prepare_context(task.prompt, repo_copy)
                    recall = _eval_context(task, prepared.payload, raw_session_tokens)

                    if retrieval_only:
                        row = _retrieval_only_row(
                            task,
                            adapter.name,
                            prepared,
                            recall,
                            raw_session_tokens,
                            ingest_duration,
                            repeat,
                            score_formula,
                            model=model,
                        )
                        results.append(row)
                        adapter.close()
                        if progress:
                            elapsed = time.monotonic() - started
                            _progress(
                                f"[{completed}/{total}] done "
                                f"recall={recall.recall:.2f} "
                                f"stale_penalty={recall.precision_penalty:.2f} "
                                f"context_success={row['context_success']} "
                                f"ingest={ingest_duration:.1f}s "
                                f"elapsed={elapsed:.1f}s"
                        )
                        continue

                    # --- Phase 3: run agent (retrieval-only, no re-injection) -
                    before = _snapshot_files(repo_copy)
                    run_start = time.monotonic()
                    try:
                        steps, in_tok, out_tok, final_msg, trace = agent.run(
                            agent_task, repo_copy, adapter, max_steps=max_steps
                        )
                    except Exception as exc:
                        steps, in_tok, out_tok, final_msg, trace = 0, 0, 0, f"ERROR: {exc}", ()
                    run_duration = time.monotonic() - run_start

                    tests_passed, test_output = _run_command(task.test_command, repo_copy)
                    after = _snapshot_files(repo_copy)
                    changed = tuple(sorted(p for p in after if before.get(p) != after.get(p)))

                    result = _score_result(
                        agent_task,
                        adapter.name,
                        changed,
                        tests_passed,
                        test_output,
                        steps,
                        in_tok,
                        out_tok,
                        prepared.memory_payload_tokens,
                        prepared.raw_token_estimate,
                        prepared.token_estimate,
                        prepared.token_savings,
                        run_duration,
                        ingest_duration,
                        final_msg,
                        trace,
                        repo_copy,
                    )
                    row = result.to_json()
                    row["mode"] = "agentic"
                    row["repeat"] = repeat
                    row["context_success"] = recall.required_found == recall.required_total and recall.stale_found == 0
                    row["memory_success"] = bool(row["success"] and row["context_success"])
                    row["recall_score"] = recall.to_dict(score_formula=score_formula)
                    row["sessions"] = len(task.sessions)
                    row["session_events"] = sum(len(s) for s in task.sessions)
                    row["family"] = task.family
                    row["raw_context_tokens"] = raw_session_tokens
                    row["token_savings"] = round(
                        max(0.0, 1.0 - prepared.token_estimate / max(1, raw_session_tokens)), 3
                    )
                    ingest_cost = estimate_ingest_cost(adapter.name, raw_session_tokens, model)
                    agent_cost = cost_usd(model, in_tok, out_tok)
                    row["model"] = model
                    row["ingest_cost_usd"] = round(ingest_cost, 6)
                    row["agent_cost_usd"] = round(agent_cost, 6)
                    row["total_cost_usd"] = round(ingest_cost + agent_cost, 6)
                    results.append(row)

                    adapter.close()

                    if progress:
                        elapsed = time.monotonic() - started
                        _progress(
                            f"[{completed}/{total}] done "
                            f"recall={recall.recall:.2f} "
                            f"stale_penalty={recall.precision_penalty:.2f} "
                            f"success={row['success']} "
                            f"ingest={ingest_duration:.1f}s "
                            f"elapsed={elapsed:.1f}s"
                        )

    return _summarize(results, adapter_names=adapter_names, score_formula=score_formula)


def _retrieval_only_row(
    task: CrossSessionTask,
    adapter: str,
    prepared,
    recall: RecallScore,
    raw_session_tokens: int,
    ingest_duration: float,
    repeat: int,
    score_formula: str,
    model: str = "gpt-4.1-mini",
) -> dict:
    context_success = recall.required_found == recall.required_total and recall.stale_found == 0
    ingest_cost = estimate_ingest_cost(adapter, raw_session_tokens, model)
    return {
        "task_id": task.task_id,
        "family": task.family,
        "adapter": adapter,
        "mode": "retrieval_only",
        "repeat": repeat,
        "success": None,
        "context_success": context_success,
        "tests_passed": None,
        "unknown_passed": None,
        "changed_files": [],
        "expected_files_touched": [],
        "missing_expected_files": [],
        "forbidden_found": [],
        "hallucinated_files": [],
        "steps": 0,
        "retries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "memory_payload_tokens": prepared.memory_payload_tokens,
        "raw_context_tokens": raw_session_tokens,
        "prepared_context_tokens": prepared.token_estimate,
        "token_savings": round(
            max(0.0, 1.0 - prepared.token_estimate / max(1, raw_session_tokens)), 3
        ),
        "duration_seconds": 0.0,
        "ingest_duration_seconds": ingest_duration,
        "final_message": "RETRIEVAL_ONLY",
        "action_trace": [],
        "recall_score": recall.to_dict(score_formula=score_formula),
        "sessions": len(task.sessions),
        "session_events": sum(len(s) for s in task.sessions),
        "model": model,
        "ingest_cost_usd": round(ingest_cost, 6),
        "agent_cost_usd": 0.0,
        "total_cost_usd": round(ingest_cost, 6),
    }


def _summarize(results: list[dict], *, adapter_names: list[str], score_formula: str = "subtractive") -> dict:
    by_adapter: dict[str, list[dict]] = {n: [] for n in adapter_names}
    for r in results:
        by_adapter.setdefault(r["adapter"], []).append(r)

    scores: dict[str, dict] = {}
    for adapter, items in by_adapter.items():
        n = max(1, len(items))
        recalls = [item["recall_score"]["recall"] for item in items]
        stale_penalties = [item["recall_score"]["precision_penalty"] for item in items]
        cs_scores = [item["recall_score"]["score"] for item in items]
        successes = [item["success"] for item in items if isinstance(item.get("success"), bool)]
        memory_successes = [
            item["memory_success"]
            for item in items
            if isinstance(item.get("memory_success"), bool)
        ]
        scores[adapter] = {
            "tasks": len(items),
            "success_rate": (
                round(sum(1 for success in successes if success) / len(successes), 4)
                if successes else None
            ),
            "memory_success_rate": (
                round(sum(1 for success in memory_successes if success) / len(memory_successes), 4)
                if memory_successes else None
            ),
            "context_success_rate": round(sum(1 for i in items if i.get("context_success")) / n, 4),
            "avg_recall": round(sum(recalls) / n, 4),
            "avg_stale_penalty": round(sum(stale_penalties) / n, 4),
            "avg_cross_session_score": round(sum(cs_scores) / n, 4),
            "avg_token_efficiency": round(
                sum(i["recall_score"]["token_efficiency"] for i in items) / n, 4
            ),
            "avg_ingest_duration_seconds": round(
                sum(i.get("ingest_duration_seconds", 0.0) for i in items) / n, 2
            ),
            "avg_total_tokens": round(sum(i["total_tokens"] for i in items) / n),
            "avg_raw_context_tokens": round(
                sum(i.get("raw_context_tokens", 0) for i in items) / n
            ),
            "avg_prepared_context_tokens": round(
                sum(i.get("prepared_context_tokens", 0) for i in items) / n
            ),
            "avg_token_savings": round(
                sum(i.get("token_savings", 0.0) for i in items if i.get("raw_context_tokens", 0) > 0) /
                max(1, sum(1 for i in items if i.get("raw_context_tokens", 0) > 0)),
                3
            ),
            "avg_ingest_cost_usd": round(sum(i.get("ingest_cost_usd", 0.0) for i in items) / n, 6),
            "avg_agent_cost_usd": round(sum(i.get("agent_cost_usd", 0.0) for i in items) / n, 6),
            "avg_total_cost_usd": round(sum(i.get("total_cost_usd", 0.0) for i in items) / n, 6),
        }
        scores[adapter]["bootstrap_ci"] = _bootstrap_ci(items)

    # Rank by cross-session score
    ranked = sorted(scores.items(), key=lambda kv: kv[1]["avg_cross_session_score"], reverse=True)
    best_adapter, best_score = ranked[0] if ranked else ("none", {})
    capsule = scores.get("capsule", {})
    mem0_baseline_name = "mem0_real" if "mem0_real" in scores else "mem0_real_async" if "mem0_real_async" in scores else None
    mem0_real = scores.get(mem0_baseline_name) if mem0_baseline_name else None
    capsule_score = capsule.get("avg_cross_session_score")
    mem0_score = mem0_real.get("avg_cross_session_score") if mem0_real else None

    return {
        "summary": {
            "adapters": adapter_names,
            "adapter_scores": scores,
            "best_adapter": best_adapter,
            "best_cross_session_score": best_score.get("avg_cross_session_score", 0.0),
            "capsule_cross_session_score": capsule_score,
            "capsule_beats_mem0_real": (
                capsule_score > mem0_score
                if capsule_score is not None and mem0_score is not None
                else None
            ),
            "mem0_baseline": mem0_baseline_name,
            "capsule_token_savings": capsule.get("avg_token_savings"),
            "mem0_token_savings": mem0_real.get("avg_token_savings") if mem0_real else None,
            "score_formula": score_formula,
            "paired_deltas": (
                _paired_deltas(results, left="capsule", right=mem0_baseline_name)
                if mem0_baseline_name else {}
            ),
        },
        "results": results,
    }


def _bootstrap_ci(items: list[dict], *, samples: int = 1000, seed: int = 2027) -> dict:
    if not items:
        return {}
    rng = random.Random(seed)
    out: dict[str, list[float]] = {
        "cross_session_score": [],
        "recall": [],
        "stale_penalty": [],
        "context_success_rate": [],
    }
    n = len(items)
    for _ in range(samples):
        draw = [items[rng.randrange(n)] for _ in range(n)]
        out["cross_session_score"].append(sum(i["recall_score"]["score"] for i in draw) / n)
        out["recall"].append(sum(i["recall_score"]["recall"] for i in draw) / n)
        out["stale_penalty"].append(sum(i["recall_score"]["precision_penalty"] for i in draw) / n)
        out["context_success_rate"].append(sum(1 for i in draw if i.get("context_success")) / n)
    return {key: _percentile_ci(values) for key, values in out.items()}


def _percentile_ci(values: list[float]) -> dict:
    if not values:
        return {"low": 0.0, "high": 0.0}
    ordered = sorted(values)
    low = ordered[int(0.025 * (len(ordered) - 1))]
    high = ordered[int(0.975 * (len(ordered) - 1))]
    return {"low": round(low, 4), "high": round(high, 4)}


def _paired_deltas(results: list[dict], *, left: str, right: str) -> dict:
    by_key: dict[tuple[str, int, str], dict] = {}
    for row in results:
        by_key[(row["task_id"], row.get("repeat", 0), row["adapter"])] = row
    deltas = []
    for task_id, repeat, adapter in sorted(by_key):
        if adapter != left:
            continue
        left_row = by_key[(task_id, repeat, left)]
        right_row = by_key.get((task_id, repeat, right))
        if not right_row:
            continue
        deltas.append(left_row["recall_score"]["score"] - right_row["recall_score"]["score"])
    if not deltas:
        return {}
    return {
        "left": left,
        "right": right,
        "pairs": len(deltas),
        "avg_cross_session_score_delta": round(sum(deltas) / len(deltas), 4),
        "bootstrap_ci": _percentile_ci(_bootstrap_means(deltas)),
    }


def _bootstrap_means(values: list[float], *, samples: int = 1000, seed: int = 2028) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    return [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def load_tasks(
    *,
    families: str | Iterable[str] = "all",
    cases_per_family: int = 1,
) -> list[CrossSessionTask]:
    if isinstance(families, str):
        selected = list(FAMILIES) if families == "all" else [f.strip() for f in families.split(",") if f.strip()]
    else:
        selected = list(families)
    unknown = sorted(set(selected) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown families: {', '.join(unknown)}")
    tasks = []
    for family in selected:
        for variant in range(cases_per_family):
            tasks.append(_build_task(family, variant))
    return tasks


def _build_task(family: str, variant: int) -> CrossSessionTask:
    if family == "auth_decision_evolution":
        return _auth_decision_evolution(variant)
    if family == "config_iteration":
        return _config_iteration(variant)
    if family == "route_migration":
        return _route_migration(variant)
    if family == "feature_flag_rollout":
        return _feature_flag_rollout(variant)
    if family == "service_ownership":
        return _service_ownership(variant)
    if family == "export_jobs_config":
        return _export_jobs_config(variant)
    if family == "stale_docs_current_code":
        return _stale_docs_current_code(variant)
    if family == "dependency_version_pin":
        return _dependency_version_pin(variant)
    if family == "unknown_owner_abstention":
        return _unknown_owner_abstention(variant)
    if family == "long_horizon_project_memory":
        return _long_horizon_project_memory(variant)
    raise ValueError(f"unknown family {family!r}")


def _auth_decision_evolution(variant: int) -> CrossSessionTask:
    """4 sessions: brainstorm → first decision → security review → revised decision."""
    vs = VariantSelector("auth_decision_evolution", variant)
    sessions = (
        (
            "Team sync: reviewing auth options for admin reports route.",
            "Discussed LegacyApiKeyAuth and session tokens. No final decision.",
            "Security doc: API keys are simpler but session tokens are more secure.",
            *vs.session_noise("auth brainstorm", 4 + variant * 2),
        ),
        (
            "DECISION (superseded): Use LegacyApiKeyAuth for admin reports.",
            "Implementation starts: AUTH_MIDDLEWARE = LegacyApiKeyAuth in src/auth/middleware.py.",
            "route_file=src/api/routes/admin_reports.py edit_file=src/auth/middleware.py",
            *vs.session_noise("auth implementation", 3),
        ),
        (
            "Security review flagged: the current auth class does not validate session expiry.",
            "Recommendation: replace with session-based middleware before launch.",
            "Ticket created: migrate admin_reports route to RequireSessionToken.",
            *vs.session_noise("security findings", 3),
        ),
        (
            "REVISED DECISION: Replace LegacyApiKeyAuth with RequireSessionToken across admin routes.",
            f"active_auth_middleware=RequireSessionToken edit_file=src/auth/middleware.py variant={variant}",
            "route_file=src/api/routes/admin_reports.py test_file=tests/test_admin_reports.py",
            "run_test=python3 -m unittest tests.test_admin_reports",
            "RequireSessionToken must be defined in src/auth/middleware.py and imported in admin_reports.py.",
        ),
    )
    return CrossSessionTask(
        task_id=f"cs_auth_evolution_{variant:03d}",
        family="auth_decision_evolution",
        prompt="Continue the admin reports auth migration and make the targeted test pass.",
        sessions=sessions,
        expected_files=("src/auth/middleware.py", "src/api/routes/admin_reports.py"),
        test_command="python3 -m unittest tests.test_admin_reports",
        forbidden_strings=("LegacyApiKeyAuth",),
        required_in_context=("RequireSessionToken", "src/auth/middleware.py"),
        stale_in_context=("LegacyApiKeyAuth",),
        variant=variant,
    )


def _config_iteration(variant: int) -> CrossSessionTask:
    """3 sessions: wrong config → revised config → finalized config key."""
    vs = VariantSelector("config_iteration", variant)
    sessions = (
        (
            "Billing retry config: using BILLING_RETRY_COUNT=3 as initial default.",
            "CONFIG_KEY=BILLING_RETRY_COUNT set in src/billing/retry_policy.py.",
            *vs.session_noise("config tuning", 3 + variant),
        ),
        (
            "Config review: BILLING_RETRY_COUNT renamed to MAX_RETRIES_3 for clarity.",
            "Still not final. Platform team wants a more specific key name.",
            *vs.session_noise("naming iteration", 3),
        ),
        (
            f"FINAL CONFIG: config_key=PAYMENT_RETRY_LIMIT_0 edit_file=src/billing/retry_policy.py variant={variant}",
            "test_file=tests/test_retry_policy.py run_test=python3 -m unittest tests.test_retry_policy",
            "All previous config names (BILLING_RETRY_COUNT, MAX_RETRIES_3) are rejected and must not be used.",
        ),
    )
    return CrossSessionTask(
        task_id=f"cs_config_iteration_{variant:03d}",
        family="config_iteration",
        prompt="Apply the current billing retry config key and run the retry policy test.",
        sessions=sessions,
        expected_files=("src/billing/retry_policy.py",),
        test_command="python3 -m unittest tests.test_retry_policy",
        forbidden_strings=("BILLING_RETRY_COUNT",),
        required_in_context=("PAYMENT_RETRY_LIMIT_0", "src/billing/retry_policy.py"),
        stale_in_context=("BILLING_RETRY_COUNT", "MAX_RETRIES_3"),
        variant=variant,
    )


def _route_migration(variant: int) -> CrossSessionTask:
    """3 sessions: old route noted → route decided → migration details confirmed."""
    vs = VariantSelector("route_migration", variant)
    sessions = (
        (
            "API review: /api/v1/export flagged as deprecated. Needs a versioned replacement.",
            "EXPORT_ENDPOINT='/api/v1/export' in src/api/routes/report_export.py is stale.",
            *vs.session_noise("deprecation tracking", 3 + variant),
        ),
        (
            "New route decided: /api/v3/reports/0/export replaces /api/v1/export.",
            "route_file=src/api/routes/report_export.py EXPORT_ENDPOINT must be updated.",
            *vs.session_noise("migration planning", 3),
        ),
        (
            f"CONFIRMED: endpoint=/api/v3/reports/0/export edit_file=src/api/routes/report_export.py variant={variant}",
            "test_file=tests/test_report_export.py run_test=python3 -m unittest tests.test_report_export",
            "Do not use /api/v1/export anywhere in the new code.",
        ),
    )
    return CrossSessionTask(
        task_id=f"cs_route_migration_{variant:03d}",
        family="route_migration",
        prompt="Finish the report export route rename and make the route test pass.",
        sessions=sessions,
        expected_files=("src/api/routes/report_export.py",),
        test_command="python3 -m unittest tests.test_report_export",
        forbidden_strings=("/api/v1/export",),
        required_in_context=("/api/v3/reports/0/export", "src/api/routes/report_export.py"),
        stale_in_context=("/api/v1/export",),
        variant=variant,
    )


def _feature_flag_rollout(variant: int) -> CrossSessionTask:
    """5 sessions: planning → beta → canary → revert → final rollout."""
    vs = VariantSelector("feature_flag_rollout", variant)
    sessions = (
        (
            "Team planning: new checkout flow discussed. Feature flag CHECKOUT_V2 proposed.",
            "Decision: CHECKOUT_V2 will start as 'disabled' during development.",
            "feature_flag=CHECKOUT_V2 initial_value=disabled edit_file=src/checkout/flags.py",
            *vs.session_noise("flag planning", 4 + variant * 2),
        ),
        (
            "Beta test report: CHECKOUT_V2='disabled' confirmed in staging.",
            "Load tests passed. Flag still disabled while QA runs.",
            "QA note: all payment flows verified against checkout v2 backend.",
            *vs.session_noise("beta observations", 5),
        ),
        (
            "Ops: enabling CHECKOUT_V2='canary' for 5% of users starting today.",
            "feature_flag=CHECKOUT_V2 current_value=canary edit_file=src/checkout/flags.py",
            "Canary metrics: conversion rate up 2%. No payment errors reported.",
            *vs.session_noise("canary metrics", 5),
        ),
        (
            "INCIDENT: checkout v2 canary caused 500s on mobile Safari. Reverting.",
            "REVERTED: CHECKOUT_V2='disabled'. All users back on v1.",
            "feature_flag=CHECKOUT_V2 current_value=disabled reason=incident_revert",
            "Post-mortem scheduled. Safari WebKit payment API edge case identified.",
            *vs.session_noise("incident notes", 4),
        ),
        (
            "Safari fix merged and verified. Full rollout approved by product.",
            f"FINAL DECISION: feature_flag=CHECKOUT_V2 final_value=enabled variant={variant}",
            "edit_file=src/checkout/flags.py set CHECKOUT_V2 = 'enabled'",
            "test_file=tests/test_checkout_flags.py run_test=python3 -m unittest tests.test_checkout_flags",
            "All previous values (disabled, canary) are superseded. Only 'enabled' is current.",
            *vs.session_noise("rollout confirmation", 3 + variant),
        ),
    )
    return CrossSessionTask(
        task_id=f"cs_feature_flag_{variant:03d}",
        family="feature_flag_rollout",
        prompt="Apply the final checkout feature flag state and make the flag test pass.",
        sessions=sessions,
        expected_files=("src/checkout/flags.py",),
        test_command="python3 -m unittest tests.test_checkout_flags",
        forbidden_strings=("'disabled'", "'canary'"),
        required_in_context=("enabled", "src/checkout/flags.py"),
        stale_in_context=("disabled", "canary"),
        variant=variant,
    )


def _service_ownership(variant: int) -> CrossSessionTask:
    """4 sessions: alice owns → bob takes over → platform-team interim → carol final."""
    vs = VariantSelector("service_ownership", variant)
    sessions = (
        (
            "Service ownership: payments-platform owned by @alice-chen.",
            "on_call_owner=@alice-chen edit_file=src/ops/oncall.py",
            "Weekly runbook updated. @alice-chen handles P1 escalations.",
            *vs.session_noise("on-call rotation", 5 + variant * 2),
        ),
        (
            "Team transition: @alice-chen moving to infra team. @bob-martinez takes payments-platform.",
            "on_call_owner=@bob-martinez edit_file=src/ops/oncall.py",
            "Handoff complete: runbooks reviewed, @bob-martinez has PagerDuty access.",
            "Alert routing updated. @bob-martinez as new primary on-call.",
            *vs.session_noise("ownership handoff", 4),
        ),
        (
            "Reorg: @bob-martinez promoted to EM. Temporary owner: @platform-team.",
            "on_call_owner=@platform-team (interim) edit_file=src/ops/oncall.py",
            "Reorg freeze: no permanent assignment until headcount review complete.",
            *vs.session_noise("reorg coverage", 4),
        ),
        (
            f"Headcount resolved. Permanent owner assigned: @carol-wu variant={variant}",
            "FINAL OWNER: on_call_owner=@carol-wu edit_file=src/ops/oncall.py",
            "test_file=tests/test_oncall.py run_test=python3 -m unittest tests.test_oncall",
            "@carol-wu supersedes all previous owners. Update src/ops/oncall.py now.",
            *vs.session_noise("ownership confirmation", 3 + variant),
        ),
    )
    return CrossSessionTask(
        task_id=f"cs_service_ownership_{variant:03d}",
        family="service_ownership",
        prompt="Update the payments-platform on-call owner to the current owner and make the test pass.",
        sessions=sessions,
        expected_files=("src/ops/oncall.py",),
        test_command="python3 -m unittest tests.test_oncall",
        forbidden_strings=("@alice-chen", "@bob-martinez", "@platform-team"),
        required_in_context=("carol-wu", "src/ops/oncall.py"),
        stale_in_context=("alice-chen", "bob-martinez", "platform-team"),
        variant=variant,
    )


def _export_jobs_config(variant: int) -> CrossSessionTask:
    """4 sessions: offset pagination → cursor decided → timeout raised → both finalized."""
    vs = VariantSelector("export_jobs_config", variant)
    sessions = (
        (
            "Export jobs service review: PAGINATION_MODE='offset' causes memory pressure on large exports.",
            "TIMEOUT_SECONDS=5 is too low; jobs timeout before completion for large datasets.",
            "config_key=PAGINATION_MODE current_value=offset edit_file=src/api/routes/export_jobs.py",
            *vs.session_noise("performance analysis", 4 + variant * 2),
        ),
        (
            "Architecture decision: switch export jobs to cursor-based pagination.",
            "config_key=PAGINATION_MODE new_value=cursor replaces offset.",
            "Cursor pagination eliminates memory spikes on large dataset exports.",
            *vs.session_noise("pagination migration", 5),
        ),
        (
            "SRE review: TIMEOUT_SECONDS must be raised to handle 10k-row exports.",
            "Proposal: TIMEOUT_SECONDS=30 gives adequate headroom without overloading the queue.",
            "config_key=TIMEOUT_SECONDS proposed_value=30 supersedes 5.",
            *vs.session_noise("timeout evaluation", 4),
        ),
        (
            f"FINAL CONFIG: PAGINATION_MODE=cursor AND TIMEOUT_SECONDS=30 variant={variant}",
            "edit_file=src/api/routes/export_jobs.py set both values now.",
            "test_file=tests/test_export_jobs.py run_test=python3 -m unittest tests.test_export_jobs",
            "Both values supersede previous defaults. offset and 5 must not remain.",
            *vs.session_noise("config finalization", 3 + variant),
        ),
    )
    return CrossSessionTask(
        task_id=f"cs_export_jobs_{variant:03d}",
        family="export_jobs_config",
        prompt="Apply the finalized export jobs config (pagination mode and timeout) and make the test pass.",
        sessions=sessions,
        expected_files=("src/api/routes/export_jobs.py",),
        test_command="python3 -m unittest tests.test_export_jobs",
        forbidden_strings=("'offset'", "TIMEOUT_SECONDS = 5"),
        required_in_context=("cursor", "TIMEOUT_SECONDS", "src/api/routes/export_jobs.py"),
        stale_in_context=("offset",),
        variant=variant,
    )


def _stale_docs_current_code(variant: int) -> CrossSessionTask:
    """Docs say one function name, current code/test requires another."""
    vs = VariantSelector("stale_docs_current_code", variant)
    sessions = (
        (
            "Invoice docs draft: use normalize_legacy_invoice for all invoice records.",
            "docs_file=docs/invoices.md rejected_function=normalize_legacy_invoice",
            *vs.session_noise("stale docs review", 4 + variant),
        ),
        (
            "Code review: docs are stale. The active implementation must expose normalize_active_invoice_0.",
            "current_function=normalize_active_invoice_0 edit_file=src/invoices/normalizer.py",
            "The legacy function name is superseded and must not be used.",
        ),
        (
            f"FINAL CODE FACT: function_name=normalize_active_invoice_0 edit_file=src/invoices/normalizer.py variant={variant}",
            "test_file=tests/test_normalizer.py run_test=python3 -m unittest tests.test_normalizer",
            "Do not restore normalize_legacy_invoice from docs.",
        ),
    )
    return CrossSessionTask(
        task_id=f"cs_stale_docs_code_{variant:03d}",
        family="stale_docs_current_code",
        prompt="Use the active invoice normalizer from current code, not stale docs.",
        sessions=sessions,
        expected_files=("src/invoices/normalizer.py",),
        test_command="python3 -m unittest tests.test_normalizer",
        forbidden_strings=("normalize_legacy_invoice",),
        required_in_context=("normalize_active_invoice_0", "src/invoices/normalizer.py"),
        stale_in_context=("normalize_legacy_invoice",),
        variant=variant,
    )


def _dependency_version_pin(variant: int) -> CrossSessionTask:
    """Dependency pin changes through several versions before final approval."""
    vs = VariantSelector("dependency_version_pin", variant)
    sessions = (
        (
            "Dependency audit: pydantic==1.10.0 is the current old baseline.",
            "dependency_file=pyproject.toml package=pydantic current_version=1.10.0",
            *vs.session_noise("dependency audit", 4 + variant),
        ),
        (
            "Upgrade proposal: pydantic==2.6.4 works locally but fails CI on validators.",
            "rejected_pin=pydantic==2.6.4 edit_file=pyproject.toml",
            "Do not use 2.6.4 as the final pin.",
        ),
        (
            f"FINAL DEPENDENCY PIN: package=pydantic version=2.7.0 edit_file=pyproject.toml variant={variant}",
            "test_file=tests/test_dependency_pins.py run_test=python3 -m unittest tests.test_dependency_pins",
            "pydantic==2.7.0 supersedes 1.10.0 and 2.6.4.",
        ),
    )
    return CrossSessionTask(
        task_id=f"cs_dependency_pin_{variant:03d}",
        family="dependency_version_pin",
        prompt="Apply the accepted dependency pin and make the dependency pin test pass.",
        sessions=sessions,
        expected_files=("pyproject.toml",),
        test_command="python3 -m unittest tests.test_dependency_pins",
        forbidden_strings=("pydantic==1.10.0", "pydantic==2.6.4"),
        required_in_context=("pydantic", "2.7.0", "pyproject.toml"),
        stale_in_context=("1.10.0", "2.6.4"),
        variant=variant,
    )


def _unknown_owner_abstention(variant: int) -> CrossSessionTask:
    """Tests abstention when no current owner fact exists."""
    vs = VariantSelector("unknown_owner_abstention", variant)
    sessions = (
        (
            "Observability planning: alert routing owner is still undecided.",
            "No owner was selected. No observability owner decision exists.",
            *vs.session_noise("ownership discussion", 3 + variant),
        ),
        (
            "Follow-up: @platform-team declined ownership pending incident review.",
            "candidate_owner=@platform-team rejected. This is not an active owner decision.",
        ),
        (
            "FINAL STATUS: no selected owner exists for observability alert routing.",
            "If asked to act on an owner, answer UNKNOWN and do not edit files.",
        ),
    )
    return CrossSessionTask(
        task_id=f"cs_unknown_owner_{variant:03d}",
        family="unknown_owner_abstention",
        prompt="Who owns observability alert routing? Only act if a selected owner exists.",
        sessions=sessions,
        expected_files=(),
        test_command="python3 -m unittest tests.test_observability",
        required_in_context=("no selected owner", "UNKNOWN"),
        stale_in_context=("platform-team",),
        unknown_expected=True,
        variant=variant,
    )


def _long_horizon_project_memory(variant: int) -> CrossSessionTask:
    """100+ event project history with multiple facts that must be current."""
    vs = VariantSelector("long_horizon_project_memory", variant)
    sessions: list[tuple[str, ...]] = []
    old_owner_cycle = ("@alice-chen", "@bob-martinez", "@platform-team")
    pagination_cycle = ("offset", "page_number", "batch")
    timeout_cycle = (5, 10, 15)

    for session_index in range(10):
        owner = old_owner_cycle[session_index % len(old_owner_cycle)]
        pagination = pagination_cycle[session_index % len(pagination_cycle)]
        timeout = timeout_cycle[session_index % len(timeout_cycle)]
        events = [
            (
                f"Historical project session {session_index}: export_jobs, checkout, on-call, auth, "
                "billing, invoices, report export routes, and dependency pins were discussed."
            ),
            (
                f"STALE DRAFT session={session_index}: PAGINATION_MODE='{pagination}' "
                f"TIMEOUT_SECONDS={timeout} owner={owner}. These notes are not final."
            ),
            (
                f"Operational note session={session_index}: {owner} reviewed export jobs logs, "
                f"but ownership and export config are still unsettled."
            ),
            (
                f"Route planning session={session_index}: /api/v1/export and /api/v2/export were "
                "mentioned during migration planning but are not the accepted final endpoint."
            ),
            (
                f"Feature flag note session={session_index}: CHECKOUT_V2 moved through disabled, "
                "canary, and incident-revert states during unrelated rollout work."
            ),
        ]
        events.extend(vs.session_noise(f"long-horizon-s{session_index}", 7 + (variant % 3)))
        sessions.append(tuple(events))

    sessions.append(
        (
            "Final architecture review: export jobs must use cursor pagination for all large exports.",
            "FINAL EXPORT CONFIG: PAGINATION_MODE=cursor edit_file=src/api/routes/export_jobs.py",
            "FINAL EXPORT CONFIG: TIMEOUT_SECONDS=30 edit_file=src/api/routes/export_jobs.py",
            "The stale export values offset, page_number, batch, 5, 10, and 15 are rejected.",
            "Final ownership review: payments-platform on-call owner is @carol-wu.",
            "FINAL OWNER: on_call_owner=@carol-wu edit_file=src/ops/oncall.py",
            "Stale owners @alice-chen, @bob-martinez, and @platform-team are superseded.",
            "test_file=tests/test_export_jobs.py run_test=python3 -m unittest tests.test_export_jobs",
            "test_file=tests/test_oncall.py run_test=python3 -m unittest tests.test_oncall",
            f"variant={variant} long_horizon_events={sum(len(session) for session in sessions)}",
        )
    )

    return CrossSessionTask(
        task_id=f"cs_long_horizon_project_{variant:03d}",
        family="long_horizon_project_memory",
        prompt=(
            "Apply the current export jobs config and current payments-platform on-call owner. "
            "Make the export jobs and on-call tests pass."
        ),
        sessions=tuple(sessions),
        expected_files=("src/api/routes/export_jobs.py", "src/ops/oncall.py"),
        test_command="python3 -m unittest tests.test_export_jobs tests.test_oncall",
        forbidden_strings=("'offset'", "TIMEOUT_SECONDS = 5", "@alice-chen", "@bob-martinez", "@platform-team"),
        required_in_context=("cursor", "TIMEOUT_SECONDS", "30", "@carol-wu", "src/api/routes/export_jobs.py", "src/ops/oncall.py"),
        stale_in_context=("offset", "page_number", "batch", "@alice-chen", "@bob-martinez", "@platform-team"),
        variant=variant,
    )



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--adapters",
        default="mem0_real,mem0_real_fast,capsule,bm25_rag,no_memory",
        help="Comma-separated adapter names.",
    )
    parser.add_argument(
        "--families",
        default="all",
        help=f"Comma-separated task families or 'all'. Choices: {', '.join(FAMILIES)}",
    )
    parser.add_argument("--cases-per-family", type=int, default=1)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1, help="Number of paired repeats per task/adapter.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--randomize-adapter-order", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score prepared memory contexts without running the patching agent.",
    )
    parser.add_argument(
        "--score-formula",
        choices=("subtractive", "multiplicative"),
        default="subtractive",
        help="How recall and stale leakage are combined into the headline context score.",
    )
    parser.add_argument(
        "--persistent-dir",
        default="",
        help=(
            "Path to a directory for persistent adapter stores. "
            "Reuse the same path across runs to simulate true cross-run memory: "
            "first run loads sessions, subsequent runs test retrieval without re-ingesting."
        ),
    )
    parser.add_argument("--json-out", default="cross-session-report.json")
    parser.add_argument("--latex-out", default="", help="Optional path for a LaTeX table of adapter scores.")
    parser.add_argument("--csv-out", default="", help="Optional path for a CSV table of adapter scores.")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help=(
            "Skip the ingest phase and use the existing store in --persistent-dir. "
            "Requires --persistent-dir and a prior run (or --load-only run) to have "
            "already built the store. This is the 'retrieval only' half of a two-phase test."
        ),
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help=(
            "Only ingest sessions into the persistent store — do not run the agent. "
            "Use this to pre-load memory on day 1, then run without --load-only on "
            "day 2 with --skip-ingest to test pure retrieval."
        ),
    )
    args = parser.parse_args(argv)

    tasks = load_tasks(families=args.families, cases_per_family=args.cases_per_family)
    adapter_list = [a.strip() for a in args.adapters.split(",") if a.strip()]
    persistent_dir = Path(args.persistent_dir) if args.persistent_dir else None

    if args.skip_ingest and not args.persistent_dir:
        raise SystemExit("--skip-ingest requires --persistent-dir")
    if args.load_only and not args.persistent_dir:
        raise SystemExit("--load-only requires --persistent-dir")

    report = run_cross_session_harness(
        tasks,
        adapters=adapter_list,
        agent_name="openai",
        model=args.model,
        max_steps=args.max_steps,
        persistent_dir=persistent_dir,
        skip_ingest=args.skip_ingest,
        load_only=args.load_only,
        retrieval_only=args.retrieval_only,
        repeats=args.repeats,
        randomize_adapter_order=args.randomize_adapter_order,
        score_formula=args.score_formula,
        seed=args.seed,
        progress=args.progress,
    )

    if args.load_only:
        total_events = sum(sum(len(s) for s in t.sessions) for t in tasks)
        _progress(f"Load-only complete. {total_events} events ingested into {persistent_dir}")
        _progress(f"Re-run without --load-only and with --skip-ingest to test retrieval.")
        return 0

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        _progress(f"Report written to {args.json_out}")
    else:
        print(json.dumps(report, indent=2))

    summary = report["summary"]
    scores = summary["adapter_scores"]
    if args.latex_out:
        Path(args.latex_out).write_text(_latex_table(scores, adapter_list), encoding="utf-8")
        _progress(f"LaTeX table written to {args.latex_out}")
    if args.csv_out:
        Path(args.csv_out).write_text(_csv_table(scores, adapter_list), encoding="utf-8")
        _progress(f"CSV table written to {args.csv_out}")
    print("\n=== Cross-Session Benchmark Results ===")
    print(f"{'Adapter':<20} {'Success':>8} {'MemOK':>8} {'CtxOK':>8} {'Recall':>8} {'Stale%':>8} {'Score':>8} {'Ingest(s)':>10} {'Tokens':>8}")
    print("-" * 91)
    for adapter in adapter_list:
        s = scores.get(adapter, {})
        success = s.get("success_rate")
        memory_success = s.get("memory_success_rate")
        success_text = "n/a" if success is None else f"{success:.3f}"
        memory_success_text = "n/a" if memory_success is None else f"{memory_success:.3f}"
        print(
            f"{adapter:<20} "
            f"{success_text:>8} "
            f"{memory_success_text:>8} "
            f"{s.get('context_success_rate', 0):>8.3f} "
            f"{s.get('avg_recall', 0):>8.3f} "
            f"{s.get('avg_stale_penalty', 0):>8.3f} "
            f"{s.get('avg_cross_session_score', 0):>8.3f} "
            f"{s.get('avg_ingest_duration_seconds', 0):>10.1f} "
            f"{s.get('avg_total_tokens', 0):>8.0f}"
        )
    print(f"\nBest adapter: {summary['best_adapter']} "
          f"(score={summary['best_cross_session_score']:.3f})")
    if summary.get("capsule_beats_mem0_real") is not None:
        verdict = "YES" if summary["capsule_beats_mem0_real"] else "NO"
        print(f"Capsule beats mem0_real: {verdict}")

    return 0


def _latex_table(scores: dict, adapter_list: list[str]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        "Adapter & Success & MemOK & CtxOK & Recall & Stale & Score & Tokens \\\\",
        "\\hline",
    ]
    for adapter in adapter_list:
        s = scores.get(adapter, {})
        success = s.get("success_rate")
        memory_success = s.get("memory_success_rate")
        success_text = "--" if success is None else f"{success:.3f}"
        memory_success_text = "--" if memory_success is None else f"{memory_success:.3f}"
        lines.append(
            f"{adapter.replace('_', '\\_')} & {success_text} & {memory_success_text} & "
            f"{s.get('context_success_rate', 0):.3f} & "
            f"{s.get('avg_recall', 0):.3f} & "
            f"{s.get('avg_stale_penalty', 0):.3f} & "
            f"{s.get('avg_cross_session_score', 0):.3f} & "
            f"{s.get('avg_total_tokens', 0):.0f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _csv_table(scores: dict, adapter_list: list[str]) -> str:
    lines = ["adapter,success_rate,memory_success_rate,context_success_rate,recall,stale_penalty,score,avg_total_tokens"]
    for adapter in adapter_list:
        s = scores.get(adapter, {})
        success = s.get("success_rate")
        memory_success = s.get("memory_success_rate")
        success_text = "" if success is None else f"{success:.4f}"
        memory_success_text = "" if memory_success is None else f"{memory_success:.4f}"
        lines.append(
            f"{adapter},{success_text},{memory_success_text},{s.get('context_success_rate', 0):.4f},"
            f"{s.get('avg_recall', 0):.4f},{s.get('avg_stale_penalty', 0):.4f},"
            f"{s.get('avg_cross_session_score', 0):.4f},{s.get('avg_total_tokens', 0):.0f}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
