"""Project-scale paired quality benchmark for agent memory.

This benchmark exercises real agent-continuity behavior more directly than the
fact-oriented paired benchmark:

- many deterministic case families instead of a few hand-written examples
- parameterized 50-file project fixtures with thousands of lines
- multi-turn user/assistant/tool state
- task-based scoring split by files, decisions, commands, unknowns, and stale facts
- comparison against full context, naive summary, last-N events, keyword context,
  and capsule memory

The default provider is deterministic and local for CI. Use ``--provider openai``
for live model checks with the same scoring contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
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

from paired_quality_benchmark import LLMResult, OpenAIChatLLM
from semantic_capsule_memory.agentic import AgenticMemoryLayer
from semantic_capsule_memory.budget import estimate_tokens, tokenizer_backend


SCALE_PRESETS = {
    "small": {"files": 10, "noise_lines": 30},
    "medium": {"files": 50, "noise_lines": 80},
    "large": {"files": 100, "noise_lines": 140},
}

FAMILIES = (
    "auth_migration",
    "billing_retry",
    "api_route_rename",
    "test_failure_followup",
    "multi_turn_patch_state",
    "unknown_owner",
    "stale_docs_current_code",
    "dependency_version_pin",
    "file_existence_claim",
    "cross_file_contract",
)

KEYS_BY_FAMILY = {
    "auth_migration": ("active_auth_middleware", "edit_file", "route_file", "test_file", "run_test"),
    "billing_retry": ("config_key", "edit_file", "test_file", "run_test"),
    "api_route_rename": ("endpoint", "route_file", "test_file", "run_test"),
    "test_failure_followup": ("accepted_decision", "failing_test", "failing_file", "next_command"),
    "multi_turn_patch_state": ("accepted_decision", "edit_file", "test_file", "run_test"),
    "unknown_owner": (),
    "stale_docs_current_code": ("function_name", "edit_file", "test_file", "run_test"),
    "dependency_version_pin": ("package", "version", "edit_file", "test_file", "run_test"),
    "file_existence_claim": ("created_file", "tool_evidence", "next_command"),
    "cross_file_contract": ("schema_version", "edit_file", "route_file", "test_file", "run_test"),
}

KEY_BOUNDARY = re.compile(
    r"\s+(?:active_auth_middleware|accepted_decision|config_key|created_file|edit_file|endpoint|failing_file|"
    r"failing_test|function_name|next_command|package|route_file|run_test|schema_version|test_file|tool_evidence|"
    r"version|do_not_edit|rejected_auth|rejected_config|rejected_endpoint|rejected_function|rejected_package)="
)


@dataclass(frozen=True)
class ProjectCase:
    case_id: str
    capsule_id: str
    scope: str
    events: tuple[str, ...]
    task: str
    category_items: dict[str, tuple[str, ...]]
    forbidden_items: tuple[str, ...] = ()
    unknown_expected: bool = False
    family: str = "custom"
    scale: str = "custom"
    variant: int = 0

    @property
    def full_context(self) -> str:
        return "\n\n".join(self.events)

    @property
    def required_items(self) -> tuple[str, ...]:
        items: list[str] = []
        for category in ("decision", "file", "command", "evidence"):
            items.extend(self.category_items.get(category, ()))
        return tuple(items)


class ProjectHeuristicLLM:
    name = "heuristic"

    def answer(self, prompt: str, task: str) -> LLMResult:
        answer = _extract_project_answer(prompt, task)
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(answer)
        return LLMResult(answer, input_tokens, output_tokens, input_tokens + output_tokens, "local")


def load_cases(
    *,
    families: str | Iterable[str] = "all",
    cases_per_family: int = 2,
    files: int = 50,
    noise_lines: int = 80,
    scales: str | Iterable[str] = "custom",
) -> list[ProjectCase]:
    selected_families = _select_families(families)
    selected_scales = _select_scales(scales)
    cases: list[ProjectCase] = []
    for scale in selected_scales:
        scale_files = files
        scale_noise_lines = noise_lines
        if scale in SCALE_PRESETS:
            scale_files = SCALE_PRESETS[scale]["files"]
            scale_noise_lines = SCALE_PRESETS[scale]["noise_lines"]
        for family in selected_families:
            for variant in range(cases_per_family):
                cases.append(_build_case(family, variant, files=scale_files, noise_lines=scale_noise_lines, scale=scale))
    return cases


def run_cases(cases: list[ProjectCase], *, provider: str, model: str, target_savings: float, progress: bool = False) -> dict:
    llm = ProjectHeuristicLLM() if provider == "heuristic" else OpenAIChatLLM(model)
    results: list[dict] = []
    total = len(cases)
    started = time.monotonic()

    for index, case in enumerate(cases, start=1):
        if progress:
            _progress(
                f"[{index}/{total}] case={case.case_id} family={case.family} scale={case.scale} "
                f"variant={case.variant}"
            )
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(
                Path(directory) / "memory.sqlite",
                default_token_budget=320,
                target_reduction=target_savings,
                conversation_budget=300,
            )
            for event in case.events:
                memory.remember(case.capsule_id, event, scope=case.scope, label=f"@{case.capsule_id}")
            packet = memory.prepare_step(
                case.capsule_id,
                case.task,
                original_context=case.full_context,
                retrieve_limit=5,
            )

            prompts = {
                "full": _baseline_prompt(case, "FULL_PROJECT_CONTEXT", case.full_context),
                "summary": _baseline_prompt(case, "NAIVE_SUMMARY", _naive_summary(case)),
                "last_n": _baseline_prompt(case, "LAST_N_EVENTS", "\n\n".join(case.events[-3:])),
                "keyword": _baseline_prompt(case, "KEYWORD_CONTEXT", _keyword_context(case)),
                "capsule": _capsule_prompt(case, packet.payload),
            }
            answers = {}
            for mode_index, (name, prompt) in enumerate(prompts.items(), start=1):
                mode_started = time.monotonic()
                if progress:
                    _progress(
                        f"  [{index}/{total} mode {mode_index}/{len(prompts)}] {name} "
                        f"input_estimate={estimate_tokens(prompt)}"
                    )
                answers[name] = llm.answer(prompt, case.task)
                if progress:
                    elapsed = time.monotonic() - mode_started
                    _progress(
                        f"  [{index}/{total} mode {mode_index}/{len(prompts)}] {name} done "
                        f"{elapsed:.1f}s tokens={answers[name].total_tokens}"
                    )

        grades = {name: grade_task_answer(result.text, case) for name, result in answers.items()}
        full_tokens = answers["full"].input_tokens
        token_savings = 0.0 if full_tokens == 0 else (full_tokens - answers["capsule"].input_tokens) / full_tokens
        item = {
            "case_id": case.case_id,
            "family": case.family,
            "scale": case.scale,
            "variant": case.variant,
            "baselines": {
                name: {**grades[name], **_usage_json(answers[name])}
                for name in ("full", "summary", "last_n", "keyword")
            },
            "capsule": {**grades["capsule"], **_usage_json(answers["capsule"]), "memory_payload_tokens": packet.memory_tokens},
            "token_savings": token_savings,
            "task_score_delta_vs_full": grades["capsule"]["task_score"] - grades["full"]["task_score"],
        }
        # Backward-compatible aliases for older tests/consumers.
        item["baseline"] = item["baselines"]["full"]
        item["task_score_delta"] = item["task_score_delta_vs_full"]
        results.append(item)
        if progress:
            elapsed = time.monotonic() - started
            avg_per_case = elapsed / index
            remaining = max(0.0, avg_per_case * (total - index))
            _progress(
                f"[{index}/{total}] complete capsule_score={grades['capsule']['task_score']:.3f} "
                f"savings={token_savings:.3f} elapsed={elapsed:.1f}s eta={remaining:.1f}s"
            )

    return summarize(results, provider=provider, model=model, target_savings=target_savings)


def grade_task_answer(answer: str, case: ProjectCase) -> dict:
    lower = answer.lower()
    required_items = case.required_items
    required_found = [item for item in required_items if item.lower() in lower]
    stale_mentions = [item for item in case.forbidden_items if item.lower() in lower and answer.strip().upper() != "UNKNOWN"]
    forbidden_found = [item for item in stale_mentions if _is_active_stale_use(answer, item)]
    unknown_passed = (not case.unknown_expected) or answer.strip().upper() == "UNKNOWN"
    category_scores = {
        "file_score": _category_score(answer, case.category_items.get("file", ())),
        "decision_score": _category_score(answer, case.category_items.get("decision", ())),
        "command_score": _category_score(answer, case.category_items.get("command", ())),
        "evidence_score": _category_score(answer, case.category_items.get("evidence", ())),
        "unknown_score": 1.0 if unknown_passed else 0.0,
        "stale_fact_avoidance": 0.0 if forbidden_found else 1.0,
    }
    task_score = 1.0 if not required_items else len(required_found) / len(required_items)
    if case.unknown_expected and not unknown_passed:
        task_score = 0.0
    if forbidden_found:
        task_score = max(0.0, task_score - 0.25 * len(forbidden_found))
    return {
        "answer": answer,
        "task_score": task_score,
        **category_scores,
        "required_found": required_found,
        "required_missing": [item for item in required_items if item not in required_found],
        "stale_mentions": stale_mentions,
        "stale_mention_count": len(stale_mentions),
        "forbidden_found": forbidden_found,
        "hallucination_count": len(forbidden_found),
        "unknown_passed": unknown_passed,
    }


def summarize(results: list[dict], *, provider: str, model: str, target_savings: float) -> dict:
    count = max(1, len(results))
    avg_savings = sum(item["token_savings"] for item in results) / count
    full_score = _avg(item["baselines"]["full"]["task_score"] for item in results)
    capsule_score = _avg(item["capsule"]["task_score"] for item in results)
    full_hallucinations = sum(item["baselines"]["full"]["hallucination_count"] for item in results)
    capsule_hallucinations = sum(item["capsule"]["hallucination_count"] for item in results)
    family_scores = _scores_by(results, "family")
    scale_scores = _scores_by(results, "scale")
    unknown_cases = [item for item in results if item["capsule"]["unknown_passed"] is not True or not item["capsule"]["required_found"]]
    actual_unknown_cases = [item for item in results if item["baseline"]["unknown_passed"] and item["baseline"]["answer"].strip().upper() == "UNKNOWN"]
    unknown_pass_rate = 1.0 if not actual_unknown_cases else _avg(1.0 if item["capsule"]["unknown_passed"] else 0.0 for item in actual_unknown_cases)
    min_family_score = min((score["capsule_task_score"] for score in family_scores.values()), default=1.0)

    baseline_scores = {
        "full": full_score,
        "summary": _avg(item["baselines"]["summary"]["task_score"] for item in results),
        "last_n": _avg(item["baselines"]["last_n"]["task_score"] for item in results),
        "keyword": _avg(item["baselines"]["keyword"]["task_score"] for item in results),
    }
    summary = {
        "provider": provider,
        "model": model if provider == "openai" else provider,
        "tokenizer_backend": tokenizer_backend(model if provider == "openai" else None),
        "cases": len(results),
        "families": len(family_scores),
        "avg_token_savings": avg_savings,
        "baseline_task_score": full_score,
        "baseline_scores": baseline_scores,
        "capsule_task_score": capsule_score,
        "task_score_delta": capsule_score - full_score,
        "baseline_hallucination_count": full_hallucinations,
        "capsule_hallucination_count": capsule_hallucinations,
        "unknown_pass_rate": unknown_pass_rate,
        "min_family_capsule_task_score": min_family_score,
        "family_scores": family_scores,
        "scale_scores": scale_scores,
        "category_scores": _category_summary(results),
        "passed": (
            avg_savings >= target_savings
            and capsule_score >= full_score - 0.03
            and capsule_hallucinations <= full_hallucinations
            and unknown_pass_rate >= 0.98
            and min_family_score >= 0.85
        ),
    }
    # Keep linters honest about the intermediate list in case report consumers
    # later need the count; it is intentionally not a pass gate by itself.
    summary["diagnostic_unknown_or_sparse_cases"] = len(unknown_cases)
    return {"summary": summary, "results": results}


def _baseline_prompt(case: ProjectCase, label: str, context: str) -> str:
    return (
        "You are continuing work in a large codebase. Answer only from the provided context. "
        "Return only a concise structured plan with these labels when relevant: decision, files, command, evidence. "
        "Do not include stale/rejected values unless explicitly asked what was rejected. "
        "If the answer is absent, answer exactly UNKNOWN.\n\n"
        f"{label}:\n{context}\n\nTASK:\n{case.task}"
    )


def _capsule_prompt(case: ProjectCase, memory_payload: str) -> str:
    return (
        "You are continuing work in a large codebase. Answer only from CAPSULE_MEMORY. "
        "Use TASK_PLAN_HINT first when present, then ACTIVE_FACTS and retrieved evidence. "
        "Return only a concise structured plan with these labels when relevant: decision, files, command, evidence. "
        "Do not include DO_NOT_USE values unless explicitly asked what was rejected. "
        "Answer UNKNOWN only when no active fact or retrieved evidence answers the task.\n\n"
        f"CAPSULE_MEMORY:\n{memory_payload}\n\nTASK:\n{case.task}"
    )


def _usage_json(result: LLMResult) -> dict:
    return {
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "usage_source": result.usage_source,
    }


def _extract_project_answer(prompt: str, task: str) -> str:
    lower = prompt.lower()
    q = task.lower()
    if _asks_unknown_owner(q) and ("no observability owner decision exists" in lower or "no owner was selected" in lower):
        return "UNKNOWN"

    family = _family_from_task(q)
    keys = KEYS_BY_FAMILY.get(family, ())
    values: list[str] = []
    for key in keys:
        value = _last_key_value(prompt, key)
        if value:
            values.append(f"{key}: {value}")
    return "; ".join(values) if values else "UNKNOWN"


def _last_key_value(text: str, key: str) -> str:
    values = []
    patterns = [
        rf"\b{re.escape(key)}=([^\n;]+)",
        rf"\b{re.escape(key)}:\s*([^\n;]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = match.group(1).split(" evidence=", 1)[0]
            value = KEY_BOUNDARY.split(value, maxsplit=1)[0]
            values.append(_clean_key_value(value))
    return values[-1] if values else ""


def _clean_key_value(value: str) -> str:
    cleaned = value.strip(" `\"',;")
    if cleaned.endswith(".") and not re.search(r"\.[A-Za-z0-9_]+$", cleaned):
        cleaned = cleaned[:-1]
    return cleaned


def _is_active_stale_use(answer: str, forbidden: str) -> bool:
    lowered = " ".join(answer.lower().split())
    target = forbidden.lower()
    for match in re.finditer(re.escape(target), lowered):
        window = lowered[max(0, match.start() - 80): match.end() + 80]
        if any(
            phrase in window
            for phrase in (
                "do not",
                "don't",
                "avoid",
                "rejected",
                "stale",
                "obsolete",
                "deprecated",
                "remove",
                "replace",
                "replacing",
                "old",
                "not restore",
                "clean up",
            )
        ):
            continue
        return True
    return False


def _build_case(family: str, variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    builders = {
        "auth_migration": _auth_migration_case,
        "billing_retry": _billing_retry_case,
        "api_route_rename": _api_route_rename_case,
        "test_failure_followup": _test_failure_followup_case,
        "multi_turn_patch_state": _multi_turn_patch_state_case,
        "unknown_owner": _unknown_owner_case,
        "stale_docs_current_code": _stale_docs_current_code_case,
        "dependency_version_pin": _dependency_version_pin_case,
        "file_existence_claim": _file_existence_claim_case,
        "cross_file_contract": _cross_file_contract_case,
    }
    try:
        return builders[family](variant, files=files, noise_lines=noise_lines, scale=scale)
    except KeyError as exc:
        raise ValueError(f"unknown family: {family}") from exc


def _auth_migration_case(variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    suffix = _suffix(variant)
    middleware = f"RequireSessionToken{suffix}"
    route = f"src/api/routes/admin_reports{suffix}.py"
    test = f"tests/api/test_admin_reports{suffix}.py"
    edit = f"src/auth/middleware{suffix}.py"
    command = f"python3 -m pytest {test} -k session_token"
    files_text = _project_files(
        {
            edit: f"""
# LegacyApiKeyAuth appears below only for stale compatibility notes.
class {middleware}:
    pass
active_auth_middleware={middleware}.
edit_file={edit}.
rejected_auth=LegacyApiKeyAuth.
""",
            route: f"""
from src.auth.middleware{suffix} import {middleware}
route_file={route}.
active_auth_middleware={middleware}.
""",
            test: f"""
test_file={test}.
run_test={command}.
""",
            "docs/security-decisions.md": f"""
Old docs mention LegacyApiKeyAuth for admin reports.
Final admin reports contract: active_auth_middleware={middleware}.
""",
        },
        filler_prefix=f"auth{suffix}",
        files=files,
        noise_lines=noise_lines,
    )
    return _case(
        "auth_migration",
        variant,
        scale,
        (files_text,),
        f"Produce the patch plan for the auth migration on admin reports variant {variant}: middleware, files, test, and command.",
        decision=(middleware,),
        file=(edit, route, test),
        command=(command,),
        forbidden=("LegacyApiKeyAuth",),
    )


def _billing_retry_case(variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    suffix = _suffix(variant)
    key = f"PAYMENT_RETRY_LIMIT_{variant}"
    edit = f"src/billing/retry_policy{suffix}.py"
    test = f"tests/billing/test_retry_policy{suffix}.py"
    command = f"python3 -m pytest {test} -k payment_retry_limit"
    files_text = _project_files(
        {
            edit: f"config_key={key}.\nedit_file={edit}.\nrejected_config=BILLING_RETRY_COUNT.",
            "src/config/settings.py": f"config_key={key}.",
            test: f"test_file={test}.\nrun_test={command}.",
            "docs/billing-decisions.md": f"Do not restore rejected_config=BILLING_RETRY_COUNT. Final config_key={key}.",
        },
        filler_prefix=f"billing{suffix}",
        files=files,
        noise_lines=noise_lines,
    )
    return _case(
        "billing_retry",
        variant,
        scale,
        (files_text,),
        f"Produce the task plan for billing retry config migration variant {variant}: current config key, edit file, test file, and command.",
        decision=(key,),
        file=(edit, test),
        command=(command,),
        forbidden=("BILLING_RETRY_COUNT",),
    )


def _api_route_rename_case(variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    suffix = _suffix(variant)
    endpoint = f"/api/v3/reports/{variant}/export"
    route = f"src/api/routes/report_export{suffix}.py"
    test = f"tests/api/test_report_export{suffix}.py"
    command = f"python3 -m pytest {test} -k route_rename"
    files_text = _project_files(
        {
            route: f"endpoint={endpoint}.\nroute_file={route}.\nrejected_endpoint=/api/v1/export.",
            test: f"test_file={test}.\nrun_test={command}.",
            "docs/api-routes.md": f"Old endpoint /api/v1/export is stale. Current endpoint={endpoint}.",
        },
        filler_prefix=f"routes{suffix}",
        files=files,
        noise_lines=noise_lines,
    )
    return _case(
        "api_route_rename",
        variant,
        scale,
        (files_text,),
        f"What API route rename plan should we follow for report export variant {variant}?",
        decision=(endpoint,),
        file=(route, test),
        command=(command,),
        forbidden=("/api/v1/export",),
    )


def _test_failure_followup_case(variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    suffix = _suffix(variant)
    edit = f"src/api/routes/export_jobs{suffix}.py"
    test = f"tests/api/test_export_jobs{suffix}.py"
    failing = f"{test}::test_export_jobs_cursor_timeout_{variant}"
    command = f"python3 -m pytest {test} -k cursor_timeout"
    files_text = _project_files(
        {
            edit: f"edit_file={edit}.\naccepted_decision=keep cursor pagination.",
            test: f"test_file={test}.",
            "docs/export-jobs.md": "The export jobs endpoint must keep cursor pagination.",
        },
        filler_prefix=f"exports{suffix}",
        files=files,
        noise_lines=noise_lines,
    )
    events = (
        files_text,
        "USER: Fix export jobs timeout but keep cursor pagination. Do not switch back to offset pagination.",
        f"ASSISTANT_FINAL: I inspected {edit} and will preserve accepted_decision=keep cursor pagination.",
        f"TOOL pytest ok=False risk=low failing_test={failing} failing_file={edit} next_command={command}",
        "USER: What is still failing, what decision must stay, and what should I run next?",
    )
    return _case(
        "test_failure_followup",
        variant,
        scale,
        events,
        f"What is still failing in export jobs variant {variant}, what accepted decision must stay, and what command should run next?",
        decision=("keep cursor pagination",),
        file=(failing, edit),
        command=(command,),
        forbidden=("offset pagination",),
    )


def _multi_turn_patch_state_case(variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    suffix = _suffix(variant)
    edit = f"src/search/query_builder{suffix}.py"
    test = f"tests/search/test_query_builder{suffix}.py"
    command = f"python3 -m pytest {test} -k escaped_terms"
    decision = "preserve quoted search terms"
    files_text = _project_files(
        {
            edit: f"edit_file={edit}.\naccepted_decision={decision}.",
            test: f"test_file={test}.\nrun_test={command}.",
            "docs/search.md": "Do not revert to splitting quoted phrases.",
        },
        filler_prefix=f"search{suffix}",
        files=files,
        noise_lines=noise_lines,
    )
    events = (
        files_text,
        "USER: The search parser breaks quoted terms.",
        f"ASSISTANT_FINAL: Patch should stay in {edit}; accepted_decision={decision}.",
        f"USER: Actually avoid changing tokenization globally. Keep the local patch and run {command}.",
    )
    return _case(
        "multi_turn_patch_state",
        variant,
        scale,
        events,
        f"Continue the search patch variant {variant}: decision, file, test, and command.",
        decision=(decision,),
        file=(edit, test),
        command=(command,),
        forbidden=("global tokenization rewrite",),
    )


def _unknown_owner_case(variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    suffix = _suffix(variant)
    files_text = _project_files(
        {
            "docs/observability.md": """
Datadog, Grafana Cloud, Honeycomb, and PagerDuty were discussed as tools.
No owner was selected. No observability owner decision exists.
""",
            f"src/observability/alerts{suffix}.py": "# Alert taxonomy draft only. This file does not assign an owner.",
        },
        filler_prefix=f"observability{suffix}",
        files=files,
        noise_lines=noise_lines,
    )
    return _case(
        "unknown_owner",
        variant,
        scale,
        (files_text,),
        f"Which observability owner did we choose for alert routing variant {variant}?",
        unknown_expected=True,
        forbidden=("Datadog", "Grafana Cloud", "Honeycomb", "PagerDuty", "platform team"),
    )


def _stale_docs_current_code_case(variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    suffix = _suffix(variant)
    function_name = f"normalize_active_invoice_{variant}"
    edit = f"src/invoices/normalizer{suffix}.py"
    test = f"tests/invoices/test_normalizer{suffix}.py"
    command = f"python3 -m pytest {test} -k active_invoice"
    files_text = _project_files(
        {
            edit: f"function_name={function_name}.\nedit_file={edit}.\nrejected_function=normalize_legacy_invoice.",
            test: f"test_file={test}.\nrun_test={command}.",
            "docs/invoices.md": "Stale docs still say normalize_legacy_invoice; current code supersedes that note.",
        },
        filler_prefix=f"invoices{suffix}",
        files=files,
        noise_lines=noise_lines,
    )
    return _case(
        "stale_docs_current_code",
        variant,
        scale,
        (files_text,),
        f"What function/file/test should handle invoice normalization variant {variant} despite stale docs?",
        decision=(function_name,),
        file=(edit, test),
        command=(command,),
        forbidden=("normalize_legacy_invoice",),
    )


def _dependency_version_pin_case(variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    suffix = _suffix(variant)
    package = "pydantic"
    version = f"2.{7 + variant}.0"
    edit = f"pyproject{suffix}.toml"
    test = f"tests/config/test_dependency_pins{suffix}.py"
    command = f"python3 -m pytest {test} -k pydantic_pin"
    files_text = _project_files(
        {
            edit: f"package={package}.\nversion={version}.\nedit_file={edit}.\nrejected_package=pydantic==1.10.",
            test: f"test_file={test}.\nrun_test={command}.",
            "docs/dependencies.md": f"Final dependency pin: package={package}. version={version}.",
        },
        filler_prefix=f"deps{suffix}",
        files=files,
        noise_lines=noise_lines,
    )
    return _case(
        "dependency_version_pin",
        variant,
        scale,
        (files_text,),
        f"What dependency pin plan should we use for variant {variant}?",
        decision=(package, version),
        file=(edit, test),
        command=(command,),
        forbidden=("pydantic==1.10",),
    )


def _file_existence_claim_case(variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    suffix = _suffix(variant)
    created = f"scripts/repair_cache{suffix}.py"
    command = f"python3 {created} --check"
    files_text = _project_files(
        {
            "docs/scripts.md": "Assistant claims are not file-system proof; use tool evidence.",
            f"tests/scripts/test_repair_cache{suffix}.py": "No generated script exists in this test fixture.",
        },
        filler_prefix=f"scripts{suffix}",
        files=files,
        noise_lines=noise_lines,
    )
    events = (
        files_text,
        f"USER: Create {created}.",
        f"ASSISTANT_FINAL: I created {created}.",
        f"TOOL ls ok=True risk=low tool_evidence={created} exists created_file={created} next_command={command}",
        "USER: Does the script exist and what command should I run?",
    )
    return _case(
        "file_existence_claim",
        variant,
        scale,
        events,
        f"Confirm file existence from tool evidence for repair script variant {variant}, and give the next command.",
        file=(created,),
        command=(command,),
        evidence=(f"{created} exists",),
        forbidden=("assistant claim only",),
    )


def _cross_file_contract_case(variant: int, *, files: int, noise_lines: int, scale: str) -> ProjectCase:
    suffix = _suffix(variant)
    schema = f"v{4 + variant}"
    model_file = f"src/contracts/report_schema{suffix}.py"
    route = f"src/api/routes/report_contract{suffix}.py"
    test = f"tests/contracts/test_report_schema{suffix}.py"
    command = f"python3 -m pytest {test} -k schema_contract"
    files_text = _project_files(
        {
            model_file: f"schema_version={schema}.\nedit_file={model_file}.",
            route: f"schema_version={schema}.\nroute_file={route}.",
            test: f"test_file={test}.\nrun_test={command}.",
            "docs/contracts.md": f"Old schema v2 is stale. Final schema_version={schema}.",
        },
        filler_prefix=f"contracts{suffix}",
        files=files,
        noise_lines=noise_lines,
    )
    return _case(
        "cross_file_contract",
        variant,
        scale,
        (files_text,),
        f"What cross-file schema contract update should we make for variant {variant}?",
        decision=(schema,),
        file=(model_file, route, test),
        command=(command,),
        forbidden=("v2",),
    )


def _case(
    family: str,
    variant: int,
    scale: str,
    events: tuple[str, ...],
    task: str,
    *,
    decision: tuple[str, ...] = (),
    file: tuple[str, ...] = (),
    command: tuple[str, ...] = (),
    evidence: tuple[str, ...] = (),
    forbidden: tuple[str, ...] = (),
    unknown_expected: bool = False,
) -> ProjectCase:
    case_id = f"project_{family}_{scale}_{variant:03d}"
    return ProjectCase(
        case_id=case_id,
        capsule_id=case_id,
        scope="project",
        events=events,
        task=task,
        category_items={"decision": decision, "file": file, "command": command, "evidence": evidence},
        forbidden_items=forbidden,
        unknown_expected=unknown_expected,
        family=family,
        scale=scale,
        variant=variant,
    )


def _project_files(special_files: dict[str, str], *, filler_prefix: str, files: int, noise_lines: int) -> str:
    all_files = dict(special_files)
    filler_count = max(0, files - len(all_files))
    for index in range(filler_count):
        path = f"src/{filler_prefix}/generated/filler_{index:03d}.py"
        all_files[path] = _filler_file(filler_prefix, index, noise_lines)

    parts = []
    for path in sorted(all_files):
        parts.append(f"FILE {path}\n{all_files[path].strip()}\nEND_FILE {path}")
    return "\n\n".join(parts)


def _filler_file(prefix: str, index: int, noise_lines: int) -> str:
    lines = [
        f"# {prefix} filler module {index}",
        "def unrelated_helper(value):",
        "    return value",
    ]
    stale_terms = (
        "LegacyApiKeyAuth",
        "BILLING_RETRY_COUNT",
        "/api/v1/export",
        "offset pagination",
        "normalize_legacy_invoice",
        "pydantic==1.10",
        "schema v2",
    )
    for line in range(noise_lines):
        stale = stale_terms[(index + line) % len(stale_terms)]
        lines.append(f"# noise line {line:03d} copied stale note mentions {stale} but has no active decision")
    return "\n".join(lines)


def _naive_summary(case: ProjectCase) -> str:
    snippets = []
    for event in case.events:
        lines = [line for line in event.splitlines() if any(token in line for token in ("Final", "USER:", "ASSISTANT_FINAL:", "TOOL "))]
        snippets.extend(lines[:12])
    return "\n".join(snippets[:36]) or case.events[-1]


def _keyword_context(case: ProjectCase) -> str:
    terms = set(re.findall(r"[A-Za-z0-9_./:-]{4,}", case.task.lower()))
    scored: list[tuple[int, str]] = []
    for event in case.events:
        for paragraph in event.split("\n\n"):
            lower = paragraph.lower()
            score = sum(1 for term in terms if term in lower)
            if score:
                scored.append((score, paragraph))
    selected = [paragraph for _score, paragraph in sorted(scored, key=lambda item: item[0], reverse=True)[:8]]
    return "\n\n".join(selected) if selected else "\n\n".join(case.events[-2:])


def _select_families(families: str | Iterable[str]) -> tuple[str, ...]:
    if families == "all":
        return FAMILIES
    if isinstance(families, str):
        selected = tuple(item.strip() for item in families.split(",") if item.strip())
    else:
        selected = tuple(families)
    unknown = [family for family in selected if family not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families: {', '.join(unknown)}")
    return selected


def _select_scales(scales: str | Iterable[str]) -> tuple[str, ...]:
    if scales == "all":
        return tuple(SCALE_PRESETS)
    if isinstance(scales, str):
        return tuple(item.strip() for item in scales.split(",") if item.strip())
    return tuple(scales)


def _family_from_task(task: str) -> str:
    if "auth" in task:
        return "auth_migration"
    if "billing" in task or "retry config" in task:
        return "billing_retry"
    if "route rename" in task or "api route" in task:
        return "api_route_rename"
    if "still failing" in task or "what command should run next" in task:
        return "test_failure_followup"
    if "search patch" in task:
        return "multi_turn_patch_state"
    if _asks_unknown_owner(task):
        return "unknown_owner"
    if "stale docs" in task or "invoice normalization" in task:
        return "stale_docs_current_code"
    if "dependency pin" in task:
        return "dependency_version_pin"
    if "file existence" in task or "repair script" in task:
        return "file_existence_claim"
    if "schema contract" in task or "cross-file" in task:
        return "cross_file_contract"
    return "unknown"


def _asks_unknown_owner(task: str) -> bool:
    return "owner" in task and "observability" in task


def _category_score(answer: str, expected: tuple[str, ...]) -> float:
    if not expected:
        return 1.0
    lower = answer.lower()
    return sum(1 for item in expected if item.lower() in lower) / len(expected)


def _scores_by(results: list[dict], key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for item in results:
        grouped.setdefault(str(item[key]), []).append(item)
    return {
        group: {
            "cases": len(items),
            "full_task_score": _avg(item["baselines"]["full"]["task_score"] for item in items),
            "summary_task_score": _avg(item["baselines"]["summary"]["task_score"] for item in items),
            "last_n_task_score": _avg(item["baselines"]["last_n"]["task_score"] for item in items),
            "keyword_task_score": _avg(item["baselines"]["keyword"]["task_score"] for item in items),
            "capsule_task_score": _avg(item["capsule"]["task_score"] for item in items),
            "avg_token_savings": _avg(item["token_savings"] for item in items),
        }
        for group, items in sorted(grouped.items())
    }


def _category_summary(results: list[dict]) -> dict[str, float]:
    keys = ("file_score", "decision_score", "command_score", "evidence_score", "unknown_score", "stale_fact_avoidance")
    return {key: _avg(item["capsule"][key] for item in results) for key in keys}


def _avg(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / max(1, len(items))


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _suffix(variant: int) -> str:
    return "" if variant == 0 else f"_{variant}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--target-savings", type=float, default=0.70)
    parser.add_argument("--families", default="all", help="Comma-separated family names or 'all'.")
    parser.add_argument("--cases-per-family", type=int, default=2)
    parser.add_argument("--files", type=int, default=50)
    parser.add_argument("--noise-lines", type=int, default=80)
    parser.add_argument("--scales", default="custom", help="Comma-separated scales, 'all', or 'custom'.")
    parser.add_argument("--progress", action="store_true", help="Print per-case/per-mode progress to stderr.")
    parser.add_argument("--no-progress", action="store_true", help="Disable automatic progress output for OpenAI runs.")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    progress = args.progress or (args.provider == "openai" and not args.no_progress)
    report = run_cases(
        load_cases(
            families=args.families,
            cases_per_family=args.cases_per_family,
            files=args.files,
            noise_lines=args.noise_lines,
            scales=args.scales,
        ),
        provider=args.provider,
        model=args.model,
        target_savings=args.target_savings,
        progress=progress,
    )
    output = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
