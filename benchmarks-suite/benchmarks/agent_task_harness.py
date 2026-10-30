"""Real repo agent-task harness with interchangeable memory adapters.

This benchmark moves beyond answer quality: it creates temporary repositories,
lets an agent edit files, runs the task's test command, retries, and scores the
final repo state. The default agent is deterministic so CI can exercise the
harness without network access; the memory adapter boundary is the stable part
for plugging in live agents later.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Iterable
import urllib.request
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARKS = Path(__file__).resolve().parent
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from agent_memory_adapters import MemoryAdapter, make_adapter
from scoring_utils import contains_fact
from pricing import cost_usd, estimate_ingest_cost
from semantic_capsule_memory.budget import estimate_tokens, tokenizer_backend


FAMILIES = (
    "auth_migration",
    "billing_retry",
    "api_route_rename",
    "test_failure_followup",
    "stale_docs_current_code",
    "dependency_version_pin",
    "unknown_owner",
)


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    family: str
    prompt: str
    memory_events: tuple[str, ...]
    expected_files: tuple[str, ...]
    test_command: str
    forbidden_strings: tuple[str, ...] = ()
    unknown_expected: bool = False
    variant: int = 0
    repo_files: int = 20
    repo_noise_lines: int = 20


@dataclass(frozen=True)
class AgentRunResult:
    task_id: str
    family: str
    adapter: str
    success: bool
    tests_passed: bool
    unknown_passed: bool
    changed_files: tuple[str, ...]
    expected_files_touched: tuple[str, ...]
    missing_expected_files: tuple[str, ...]
    forbidden_found: tuple[str, ...]
    hallucinated_files: tuple[str, ...]
    steps: int
    retries: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    memory_payload_tokens: int
    raw_context_tokens: int
    prepared_context_tokens: int
    token_savings: float
    duration_seconds: float
    ingest_duration_seconds: float
    final_message: str
    action_trace: tuple[str, ...] = ()
    model: str = ""
    ingest_cost_usd: float = 0.0
    agent_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    def to_json(self) -> dict:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "adapter": self.adapter,
            "success": self.success,
            "tests_passed": self.tests_passed,
            "unknown_passed": self.unknown_passed,
            "changed_files": list(self.changed_files),
            "expected_files_touched": list(self.expected_files_touched),
            "missing_expected_files": list(self.missing_expected_files),
            "forbidden_found": list(self.forbidden_found),
            "hallucinated_files": list(self.hallucinated_files),
            "steps": self.steps,
            "retries": self.retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "memory_payload_tokens": self.memory_payload_tokens,
            "raw_context_tokens": self.raw_context_tokens,
            "prepared_context_tokens": self.prepared_context_tokens,
            "token_savings": self.token_savings,
            "duration_seconds": self.duration_seconds,
            "ingest_duration_seconds": self.ingest_duration_seconds,
            "final_message": self.final_message,
            "action_trace": list(self.action_trace),
            "model": self.model,
            "ingest_cost_usd": self.ingest_cost_usd,
            "agent_cost_usd": self.agent_cost_usd,
            "total_cost_usd": self.total_cost_usd,
        }




_SYSTEM_PROMPT = (
    "You are a coding agent in a temporary repository. Reply with exactly one JSON object "
    "for the next tool action. Do not use markdown."
)


class BaseToolAgent:
    """Base class for LLM-backed coding agents. Subclasses implement _complete()."""

    name = "base"

    def __init__(self, model: str, *, max_retries: int = 2) -> None:
        self.model = model
        self.max_retries = max_retries

    def _complete(self, prompt: str) -> tuple[str, int, int]:
        raise NotImplementedError

    def run(self, task: AgentTask, repo_path: Path, adapter: MemoryAdapter, *, max_steps: int = 8) -> tuple[int, int, int, str, tuple[str, ...]]:
        input_tokens = 0
        output_tokens = 0
        observations: list[str] = []
        action_trace: list[str] = []
        final_message = "NO_ACTION"

        initial_ok, initial_output = _run_command(task.test_command, repo_path)
        initial_observation = f"INITIAL_TESTS ok={initial_ok} command={task.test_command}\n{initial_output[-4000:]}"
        observations.append(initial_observation[-2000:])
        action_trace.append(initial_observation[:500])
        adapter.remember_tool_result(initial_observation[-2000:])
        if task.unknown_expected:
            return 1, input_tokens, output_tokens, "UNKNOWN", tuple(action_trace)

        steps = 0
        last_action_key: str | None = None
        consecutive_same: int = 0

        for steps in range(1, max_steps + 1):
            prepared = adapter.prepare_context(task.prompt, repo_path)
            prompt = _tool_agent_prompt(task, prepared.payload, observations)
            action_text, in_tokens, out_tokens = self._complete(prompt)
            input_tokens += in_tokens or estimate_tokens(prompt)
            output_tokens += out_tokens or estimate_tokens(action_text)
            action = _parse_tool_action(action_text)

            action_key = f"{action.get('action')}:{action.get('path', action.get('pattern', ''))}"
            if action_key == last_action_key and action.get("action") in {"read_file", "search_repo"}:
                consecutive_same += 1
            else:
                consecutive_same = 0
            last_action_key = action_key

            observation, done = _execute_tool_action(action, task, repo_path)

            if consecutive_same >= 1 and action.get("action") in {"read_file", "search_repo"}:
                path_hint = action.get("path") or action.get("pattern") or ""
                observation = (
                    observation + f"\n\nSTOP_LOOP: You have now read '{path_hint}' {consecutive_same + 1} times "
                    "in a row with no change. The content is exactly as shown above. "
                    "Do NOT read it again. Choose one of: apply_patch, write_file, or run_tests."
                )

            observations.append(observation[-2000:])
            action_trace.append(f"ACTION {steps}: {action}\nOBSERVATION: {observation[:1000]}")
            adapter.remember_tool_result(observation[-2000:])
            final_message = f"{action.get('action', 'invalid')}: {observation[:160]}"
            if done:
                final_message = action.get("message") or final_message
                break
        return steps, input_tokens, output_tokens, final_message, tuple(action_trace)


class OpenAIToolAgent(BaseToolAgent):
    """OpenAI-backed coding agent."""

    name = "openai"

    def __init__(self, model: str, *, max_retries: int = 2) -> None:
        super().__init__(model, max_retries=max_retries)
        self.api_key = os.environ.get("AYYGENT_AI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise SystemExit("Set AYYGENT_AI_API_KEY or OPENAI_API_KEY to run --agent openai")

    def _complete(self, prompt: str) -> tuple[str, int, int]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = payload["choices"][0]["message"]["content"]
                usage = payload.get("usage") or {}
                return text, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
            except HTTPError as exc:
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise
                last_exc = exc
            except (TimeoutError, URLError) as exc:
                last_exc = exc
            time.sleep(min(6.0, 0.75 * (2**attempt)))
        raise RuntimeError(f"OpenAI tool-agent request failed: {last_exc}")


class AnthropicToolAgent(BaseToolAgent):
    """Anthropic Claude-backed coding agent."""

    name = "anthropic"

    def __init__(self, model: str, *, max_retries: int = 2) -> None:
        super().__init__(model, max_retries=max_retries)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise SystemExit("Set ANTHROPIC_API_KEY to run --agent anthropic / --model claude-*")

    def _complete(self, prompt: str) -> tuple[str, int, int]:
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": 4096,
                "system": _SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = payload["content"][0]["text"]
                usage = payload.get("usage") or {}
                return text, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
            except HTTPError as exc:
                if exc.code not in {408, 409, 429, 500, 502, 503, 504, 529}:
                    raise
                last_exc = exc
            except (TimeoutError, URLError) as exc:
                last_exc = exc
            time.sleep(min(6.0, 0.75 * (2**attempt)))
        raise RuntimeError(f"Anthropic tool-agent request failed: {last_exc}")


class HeuristicPatchAgent:
    """Deterministic patcher used to validate the harness and adapters."""

    name = "heuristic"

    def run(self, task: AgentTask, repo_path: Path, adapter: MemoryAdapter, *, max_steps: int = 3) -> tuple[int, int, int, str, tuple[str, ...]]:
        input_tokens = 0
        output_tokens = 0
        final_message = "NO_ACTION"
        action_trace: list[str] = []
        steps = 0
        for steps in range(1, max_steps + 1):
            prepared = adapter.prepare_context(task.prompt, repo_path)
            input_tokens += estimate_tokens(task.prompt) + prepared.token_estimate
            action = _choose_patch_action(task, prepared.payload, repo_path)
            output_tokens += estimate_tokens(action)
            if action == "UNKNOWN":
                final_message = "UNKNOWN"
                action_trace.append("UNKNOWN")
                break
            if action == "NO_ACTION":
                final_message = "NO_ACTION"
            else:
                _apply_action(action, repo_path)
                final_message = action
                action_trace.append(action)

            ok, output = _run_command(task.test_command, repo_path)
            observation = f"command={task.test_command} ok={ok} output={output[-1200:]}"
            action_trace.append(observation[:500])
            adapter.remember_tool_result(observation)
            if ok:
                break
        return steps, input_tokens, output_tokens, final_message, tuple(action_trace)


def load_tasks(
    *,
    families: str | Iterable[str] = "all",
    cases_per_family: int = 1,
    noise_events: int = 25,
    noise_lines: int = 4,
    stale_events: int = 4,
    repo_files: int = 20,
    repo_noise_lines: int = 20,
) -> list[AgentTask]:
    selected = _select_families(families)
    tasks: list[AgentTask] = []
    for family in selected:
        for variant in range(cases_per_family):
            task = _build_task(family, variant)
            task = _with_noise(task, noise_events=noise_events, noise_lines=noise_lines, stale_events=stale_events)
            tasks.append(_with_repo_shape(task, repo_files=repo_files, repo_noise_lines=repo_noise_lines))
    return tasks


def run_harness(
    tasks: list[AgentTask],
    *,
    adapters: Iterable[str],
    max_steps: int = 3,
    progress: bool = False,
    agent_name: str = "heuristic",
    model: str = "gpt-4.1-mini",
) -> dict:
    results: list[dict] = []
    agent = _make_agent(agent_name, model)
    started = time.monotonic()
    adapter_names = list(adapters)
    total = len(tasks) * len(adapter_names)
    completed = 0

    for task in tasks:
        for adapter_name in adapter_names:
            completed += 1
            if progress:
                _progress(f"[{completed}/{total}] task={task.task_id} adapter={adapter_name}")
            with TemporaryDirectory() as directory:
                repo_path = Path(directory) / "repo"
                _create_repo_fixture(task, repo_path)
                before = _snapshot_files(repo_path)
                adapter = make_adapter(adapter_name)
                adapter.reset(task.task_id)
                try:
                    ingest_started = time.monotonic()
                    for event in task.memory_events:
                        adapter.ingest_event(event, scope="project")
                    ingest_duration = time.monotonic() - ingest_started
                    run_started = time.monotonic()
                    prepared_before = adapter.prepare_context(task.prompt, repo_path)
                    steps, input_tokens, output_tokens, final_message, action_trace = agent.run(task, repo_path, adapter, max_steps=max_steps)
                    tests_passed, test_output = _run_command(task.test_command, repo_path)
                    after = _snapshot_files(repo_path)
                    changed_files = tuple(sorted(path for path in after if before.get(path) != after.get(path)))
                    result = _score_result(
                        task,
                        adapter.name,
                        changed_files,
                        tests_passed,
                        test_output,
                        steps,
                        input_tokens,
                        output_tokens,
                        prepared_before.memory_payload_tokens,
                        prepared_before.raw_token_estimate,
                        prepared_before.token_estimate,
                        prepared_before.token_savings,
                        time.monotonic() - run_started,
                        ingest_duration,
                        final_message,
                        action_trace,
                        repo_path,
                    )
                    results.append(result.to_json())
                finally:
                    adapter.close()
            if progress:
                elapsed = time.monotonic() - started
                _progress(f"[{completed}/{total}] done success={results[-1]['success']} elapsed={elapsed:.1f}s")

    return summarize(results, adapter_names=adapter_names, agent_name=getattr(agent, "name", agent_name))


def summarize(results: list[dict], *, adapter_names: list[str], agent_name: str = "heuristic") -> dict:
    by_adapter: dict[str, list[dict]] = {name: [] for name in adapter_names}
    for result in results:
        by_adapter.setdefault(result["adapter"], []).append(result)

    adapter_scores = {}
    for adapter, items in by_adapter.items():
        count = max(1, len(items))
        successes = sum(1 for item in items if item["success"])
        adapter_scores[adapter] = {
            "tasks": len(items),
            "success_rate": successes / count,
            "tests_pass_rate": sum(1 for item in items if item["tests_passed"]) / count,
            "unknown_pass_rate": sum(1 for item in items if item["unknown_passed"]) / count,
            "avg_token_savings": sum(item["token_savings"] for item in items) / count,
            "avg_raw_context_tokens": sum(item["raw_context_tokens"] for item in items) / count,
            "avg_prepared_context_tokens": sum(item["prepared_context_tokens"] for item in items) / count,
            "avg_memory_payload_tokens": sum(item["memory_payload_tokens"] for item in items) / count,
            "avg_total_tokens": sum(item["total_tokens"] for item in items) / count,
            "avg_steps": sum(item["steps"] for item in items) / count,
            "avg_ingest_duration_seconds": sum(item.get("ingest_duration_seconds", 0.0) for item in items) / count,
            "hallucinated_file_count": sum(len(item["hallucinated_files"]) for item in items),
            "forbidden_found_count": sum(len(item["forbidden_found"]) for item in items),
        }

    best_success = max((score["success_rate"] for score in adapter_scores.values()), default=0.0)
    capsule = adapter_scores.get("capsule", {})
    return {
        "summary": {
            "agent": agent_name,
            "tokenizer_backend": tokenizer_backend(),
            "tasks": len({result["task_id"] for result in results}),
            "runs": len(results),
            "adapters": adapter_names,
            "adapter_scores": adapter_scores,
            "capsule_success_rate": capsule.get("success_rate"),
            "best_success_rate": best_success,
            "capsule_token_savings": capsule.get("avg_token_savings"),
            "passed": bool(capsule) and capsule.get("success_rate", 0.0) >= best_success,
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters", default="no_memory,summary,lexical_overlap,bm25_rag,repo_search,langgraph_state,capsule")
    parser.add_argument("--families", default="all")
    parser.add_argument("--cases-per-family", type=int, default=1)
    parser.add_argument("--noise-events", type=int, default=25)
    parser.add_argument("--noise-lines", type=int, default=4)
    parser.add_argument("--stale-events", type=int, default=4)
    parser.add_argument("--repo-files", type=int, default=20)
    parser.add_argument("--repo-noise-lines", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--agent", choices=("heuristic", "openai"), default="heuristic")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    tasks = load_tasks(
        families=args.families,
        cases_per_family=args.cases_per_family,
        noise_events=args.noise_events,
        noise_lines=args.noise_lines,
        stale_events=args.stale_events,
        repo_files=args.repo_files,
        repo_noise_lines=args.repo_noise_lines,
    )
    report = run_harness(
        tasks,
        adapters=[item.strip() for item in args.adapters.split(",") if item.strip()],
        max_steps=args.max_steps,
        progress=args.progress,
        agent_name=args.agent,
        model=args.model,
    )
    output = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["summary"]["passed"] else 1



def _make_agent(agent_name: str, model: str):
    if agent_name == "heuristic":
        return HeuristicPatchAgent()
    if agent_name == "anthropic":
        return AnthropicToolAgent(model)
    if agent_name in ("openai", "auto"):
        if model.startswith("claude-"):
            return AnthropicToolAgent(model)
        return OpenAIToolAgent(model)
    raise ValueError(f"unknown agent {agent_name!r}")


def _tool_agent_prompt(task: AgentTask, memory_context: str, observations: list[str]) -> str:
    recent = "\n\n".join(observations[-6:])
    return f"""
TASK:
{task.prompt}

EXPECTED TEST COMMAND:
{task.test_command}

MEMORY_CONTEXT:
{memory_context or '(empty)'}

RECENT OBSERVATIONS:
{recent or '(none)'}

Return one JSON object. Available actions:
{{"action":"list_files"}}
{{"action":"read_file","path":"relative/path.py"}}
{{"action":"search_repo","pattern":"text or regex"}}
{{"action":"apply_patch","path":"relative/path.py","old":"exact text to replace","new":"replacement text"}}
{{"action":"write_file","path":"relative/path.py","content":"complete file content"}}
{{"action":"run_tests","command":"{task.test_command}"}}
{{"action":"batch","actions":[{{"action":"apply_patch","path":"relative/path.py","old":"old","new":"new"}},{{"action":"run_tests","command":"{task.test_command}"}}]}}
{{"action":"done","message":"short result"}}

Rules:
- MEMORY_CONTEXT contains project decisions and context, NOT literal file content. Always read_file to get exact text before applying a patch.
- The expected test was already run before your first step; use RECENT OBSERVATIONS.
- A "PATCHED" observation means the edit succeeded. Do NOT retry the same patch — it will always fail because the old text is gone.
- If apply_patch returns "old text not found", read_file to see the current content, then retry with the correct old text. Never retry a failing patch without reading the file first.
- If you read a file and it already shows the correct content, stop reading it. Either edit the next required file, or run tests if all edits are complete.
- If an observation contains STOP_LOOP, you MUST immediately choose apply_patch, write_file, or run_tests. Reading the same file again is forbidden.
- Never read the same file more than twice in a row.
- If the task requires edits to multiple files, address every file before running tests.
- Prefer apply_patch for small edits. Use write_file only when replacement is simpler.
- After editing all required files, run the expected test command.
- If tests pass, return done.
- If the task says no selected owner or no active decision exists, do not edit; return done with message UNKNOWN.
- Return exactly one JSON object and nothing else.
""".strip()


def _parse_tool_action(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        action = json.loads(stripped)
        return action if isinstance(action, dict) else {"action": "invalid", "raw": text}
    except json.JSONDecodeError:
        action = _first_json_object(stripped)
        if action is not None:
            return action
    return {"action": "invalid", "raw": text}


def _first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            action, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(action, dict):
            return action
    return None


def _execute_tool_action(action: dict, task: AgentTask, repo_path: Path) -> tuple[str, bool]:
    name = str(action.get("action", "invalid"))
    if name == "batch":
        observations: list[str] = []
        done = False
        for subaction in action.get("actions", []):
            if not isinstance(subaction, dict):
                observations.append(f"ERROR invalid batch item: {subaction}")
                continue
            observation, sub_done = _execute_tool_action(subaction, task, repo_path)
            observations.append(observation)
            done = done or sub_done
            if sub_done:
                break
        return "\n---\n".join(observations), done
    if name == "list_files":
        files = sorted(_repo_relative(repo_path, path) for path in repo_path.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
        return "FILES:\n" + "\n".join(files[:300]), False
    if name == "read_file":
        path = _safe_repo_path(repo_path, str(action.get("path", "")))
        if path is None or not path.exists() or not path.is_file():
            return f"ERROR read_file invalid path={action.get('path')}", False
        return f"FILE {_repo_relative(repo_path, path)}:\n" + path.read_text(encoding="utf-8")[:6000], False
    if name == "search_repo":
        pattern = str(action.get("pattern", ""))[:200]
        return _search_repo(repo_path, pattern), False
    if name == "apply_patch":
        path = _safe_repo_path(repo_path, str(action.get("path", "")))
        old = action.get("old", "")
        new = action.get("new", "")
        if path is None or not path.exists() or not isinstance(old, str) or not isinstance(new, str):
            return f"ERROR apply_patch invalid path={action.get('path')}", False
        if str(path.relative_to(repo_path.resolve())).startswith("tests/"):
            return "ERROR apply_patch test files are read-only; the test defines the target — fix the source file instead", False
        text = path.read_text(encoding="utf-8")
        if old not in text:
            return f"ERROR apply_patch old text not found in {_repo_relative(repo_path, path)}", False
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"PATCHED {_repo_relative(repo_path, path)} old_bytes={len(old)} new_bytes={len(new)}", False
    if name == "write_file":
        path = _safe_repo_path(repo_path, str(action.get("path", "")))
        content = action.get("content", "")
        if path is None or not isinstance(content, str):
            return f"ERROR write_file invalid path={action.get('path')}", False
        if str(path.relative_to(repo_path.resolve())).startswith("tests/"):
            return "ERROR write_file test files are read-only; the test defines the target — fix the source file instead", False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"WROTE {_repo_relative(repo_path, path)} bytes={len(content)}", False
    if name == "run_tests":
        command = str(action.get("command") or task.test_command)
        if command != task.test_command and not command.startswith("python3 -m unittest"):
            return f"ERROR run_tests command not allowed: {command}", False
        ok, output = _run_command(command, repo_path)
        return f"TESTS ok={ok} command={command}\n{output[-4000:]}", ok
    if name == "done":
        return str(action.get("message", "done")), True
    return f"ERROR invalid action: {action}", False


def _safe_repo_path(repo_path: Path, raw_path: str) -> Path | None:
    if not raw_path or raw_path.startswith("/") or ".." in Path(raw_path).parts:
        return None
    path = (repo_path / raw_path).resolve()
    try:
        path.relative_to(repo_path.resolve())
    except ValueError:
        return None
    return path


def _repo_relative(repo_path: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_path.resolve()))


def _search_repo(repo_path: Path, pattern: str) -> str:
    if not pattern:
        return "ERROR empty search pattern"
    results: list[str] = []
    try:
        regex = re.compile(pattern, re.I)
    except re.error:
        regex = re.compile(re.escape(pattern), re.I)
    for path in sorted(repo_path.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".md", ".toml", ".txt"}:
            continue
        rel = _repo_relative(repo_path, path)
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if regex.search(line):
                results.append(f"{rel}:{lineno}: {line[:240]}")
                if len(results) >= 80:
                    return "SEARCH_RESULTS:\n" + "\n".join(results)
    return "SEARCH_RESULTS:\n" + ("\n".join(results) if results else "(none)")

def _score_result(
    task: AgentTask,
    adapter: str,
    changed_files: tuple[str, ...],
    tests_passed: bool,
    test_output: str,
    steps: int,
    input_tokens: int,
    output_tokens: int,
    memory_payload_tokens: int,
    raw_context_tokens: int,
    prepared_context_tokens: int,
    token_savings: float,
    duration: float,
    ingest_duration: float,
    final_message: str,
    action_trace: tuple[str, ...] = (),
    repo_path: Path | None = None,
    model: str = "",
) -> AgentRunResult:
    changed_set = set(changed_files)
    expected_set = set(task.expected_files)
    missing = tuple(sorted(expected_set - changed_set))
    expected_touched = tuple(sorted(expected_set & changed_set))
    repo_root = repo_path or Path(".")
    forbidden_found = tuple(sorted(item for item in task.forbidden_strings if _repo_contains(repo_root, item, test_output)))
    hallucinated = tuple(path for path in changed_files if path not in expected_set and not task.unknown_expected)
    unknown_passed = (not task.unknown_expected) or (final_message == "UNKNOWN" and not changed_files)
    success = tests_passed and unknown_passed and not forbidden_found and not hallucinated and (task.unknown_expected or not missing)
    total_tokens = input_tokens + output_tokens
    return AgentRunResult(
        task_id=task.task_id,
        family=task.family,
        adapter=adapter,
        success=success,
        tests_passed=tests_passed,
        unknown_passed=unknown_passed,
        changed_files=changed_files,
        expected_files_touched=expected_touched,
        missing_expected_files=missing,
        forbidden_found=forbidden_found,
        hallucinated_files=hallucinated,
        steps=steps,
        retries=max(0, steps - 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        memory_payload_tokens=memory_payload_tokens,
        raw_context_tokens=raw_context_tokens,
        prepared_context_tokens=prepared_context_tokens,
        token_savings=token_savings,
        duration_seconds=duration,
        ingest_duration_seconds=ingest_duration,
        final_message=final_message,
        action_trace=action_trace,
        model=model,
        agent_cost_usd=round(cost_usd(model, input_tokens, output_tokens), 6) if model else 0.0,
    )


def _choose_patch_action(task: AgentTask, context: str, repo_path: Path) -> str:
    if task.unknown_expected:
        return "UNKNOWN" if "owner" in context.lower() or not context else "NO_ACTION"
    lowered = context.lower()
    if task.family == "auth_migration" and "requiresessiontoken" in lowered:
        return "PATCH_AUTH"
    if task.family == "billing_retry" and "payment_retry_limit" in lowered:
        return "PATCH_BILLING"
    if task.family == "api_route_rename" and "/api/v3/reports/" in lowered:
        return "PATCH_ROUTE"
    if task.family == "test_failure_followup" and "cursor pagination" in lowered:
        return "PATCH_EXPORT_JOBS"
    if task.family == "stale_docs_current_code" and "normalize_active_invoice" in lowered:
        return "PATCH_INVOICE"
    if task.family == "dependency_version_pin" and "pydantic" in lowered and "2.7.0" in lowered:
        return "PATCH_DEPENDENCY"
    return "NO_ACTION"


def _apply_action(action: str, repo_path: Path) -> None:
    if action == "PATCH_AUTH":
        _write(repo_path / "src/auth/middleware.py", "class RequireSessionToken:\n    pass\n")
        _write(
            repo_path / "src/api/routes/admin_reports.py",
            "from src.auth.middleware import RequireSessionToken\n\nAUTH_MIDDLEWARE = RequireSessionToken\n",
        )
    elif action == "PATCH_BILLING":
        _write(repo_path / "src/billing/retry_policy.py", "CONFIG_KEY = 'PAYMENT_RETRY_LIMIT_0'\n")
    elif action == "PATCH_ROUTE":
        _write(repo_path / "src/api/routes/report_export.py", "EXPORT_ENDPOINT = '/api/v3/reports/0/export'\n")
    elif action == "PATCH_EXPORT_JOBS":
        _write(repo_path / "src/api/routes/export_jobs.py", "PAGINATION_MODE = 'cursor'\nTIMEOUT_SECONDS = 30\n")
    elif action == "PATCH_INVOICE":
        _write(
            repo_path / "src/invoices/normalizer.py",
            "def normalize_active_invoice_0(invoice):\n    return {'id': invoice['id'], 'status': 'active'}\n",
        )
    elif action == "PATCH_DEPENDENCY":
        _write(repo_path / "pyproject.toml", "[project]\ndependencies = ['pydantic==2.7.0']\n")


def _build_task(family: str, variant: int) -> AgentTask:
    if family == "auth_migration":
        return AgentTask(
            task_id=f"agent_auth_migration_{variant:03d}",
            family=family,
            prompt="Continue the admin reports auth migration and make the targeted test pass.",
            memory_events=(
                "Old auth note rejected_auth=LegacyApiKeyAuth. Do not restore it.",
                "Final decision active_auth_middleware=RequireSessionToken edit_file=src/auth/middleware.py route_file=src/api/routes/admin_reports.py test_file=tests/test_admin_reports.py run_test=python3 -m unittest tests.test_admin_reports.",
            ),
            expected_files=("src/auth/middleware.py", "src/api/routes/admin_reports.py"),
            test_command="python3 -m unittest tests.test_admin_reports",
            forbidden_strings=("LegacyApiKeyAuth",),
            variant=variant,
        )
    if family == "billing_retry":
        return AgentTask(
            task_id=f"agent_billing_retry_{variant:03d}",
            family=family,
            prompt="Apply the current billing retry config key and run the retry policy test.",
            memory_events=(
                "Rejected config rejected_config=BILLING_RETRY_COUNT.",
                "Final config_key=PAYMENT_RETRY_LIMIT_0 edit_file=src/billing/retry_policy.py test_file=tests/test_retry_policy.py run_test=python3 -m unittest tests.test_retry_policy.",
            ),
            expected_files=("src/billing/retry_policy.py",),
            test_command="python3 -m unittest tests.test_retry_policy",
            forbidden_strings=("BILLING_RETRY_COUNT",),
            variant=variant,
        )
    if family == "api_route_rename":
        return AgentTask(
            task_id=f"agent_api_route_rename_{variant:03d}",
            family=family,
            prompt="Finish the report export route rename and make the route test pass.",
            memory_events=(
                "Old endpoint /api/v1/export is stale.",
                "Final endpoint=/api/v3/reports/0/export route_file=src/api/routes/report_export.py test_file=tests/test_report_export.py run_test=python3 -m unittest tests.test_report_export.",
            ),
            expected_files=("src/api/routes/report_export.py",),
            test_command="python3 -m unittest tests.test_report_export",
            forbidden_strings=("/api/v1/export",),
            variant=variant,
        )
    if family == "test_failure_followup":
        return AgentTask(
            task_id=f"agent_test_failure_followup_{variant:03d}",
            family=family,
            prompt="Continue from the failing export jobs test and apply the accepted pagination decision.",
            memory_events=(
                "accepted_decision=keep cursor pagination failing_test=tests/test_export_jobs.py::test_export_jobs_cursor_timeout_0 failing_file=src/api/routes/export_jobs.py next_command=python3 -m unittest tests.test_export_jobs.",
            ),
            expected_files=("src/api/routes/export_jobs.py",),
            test_command="python3 -m unittest tests.test_export_jobs",
            variant=variant,
        )
    if family == "stale_docs_current_code":
        return AgentTask(
            task_id=f"agent_stale_docs_current_code_{variant:03d}",
            family=family,
            prompt="Use the active invoice normalizer from current code, not stale docs.",
            memory_events=(
                "docs/invoices.md is stale and mentions rejected_function=normalize_legacy_invoice.",
                "Current code function_name=normalize_active_invoice_0 edit_file=src/invoices/normalizer.py test_file=tests/test_normalizer.py run_test=python3 -m unittest tests.test_normalizer.",
            ),
            expected_files=("src/invoices/normalizer.py",),
            test_command="python3 -m unittest tests.test_normalizer",
            forbidden_strings=("normalize_legacy_invoice",),
            variant=variant,
        )
    if family == "dependency_version_pin":
        return AgentTask(
            task_id=f"agent_dependency_version_pin_{variant:03d}",
            family=family,
            prompt="Apply the accepted dependency pin and make the dependency pin test pass.",
            memory_events=(
                "Final dependency pin package=pydantic version=2.7.0 edit_file=pyproject.toml test_file=tests/test_dependency_pins.py run_test=python3 -m unittest tests.test_dependency_pins.",
            ),
            expected_files=("pyproject.toml",),
            test_command="python3 -m unittest tests.test_dependency_pins",
            variant=variant,
        )
    if family == "unknown_owner":
        return AgentTask(
            task_id=f"agent_unknown_owner_{variant:03d}",
            family=family,
            prompt="Who owns observability alert routing? Only act if a selected owner exists.",
            memory_events=("No owner was selected. No observability owner decision exists.",),
            expected_files=(),
            test_command="python3 -m unittest tests.test_observability",
            unknown_expected=True,
            variant=variant,
        )
    raise ValueError(f"unknown family {family!r}")


def _with_noise(task: AgentTask, *, noise_events: int, noise_lines: int, stale_events: int) -> AgentTask:
    if noise_events <= 0 and stale_events <= 0:
        return task
    noise = tuple(_noise_event(task, index, lines=noise_lines) for index in range(noise_events))
    stale = tuple(_stale_event(task, index, lines=max(1, noise_lines // 2)) for index in range(stale_events))
    return AgentTask(
        task_id=task.task_id,
        family=task.family,
        prompt=task.prompt,
        memory_events=(*noise, *stale, *task.memory_events),
        expected_files=task.expected_files,
        test_command=task.test_command,
        forbidden_strings=task.forbidden_strings,
        unknown_expected=task.unknown_expected,
        variant=task.variant,
        repo_files=task.repo_files,
        repo_noise_lines=task.repo_noise_lines,
    )


def _with_repo_shape(task: AgentTask, *, repo_files: int, repo_noise_lines: int) -> AgentTask:
    return AgentTask(
        task_id=task.task_id,
        family=task.family,
        prompt=task.prompt,
        memory_events=task.memory_events,
        expected_files=task.expected_files,
        test_command=task.test_command,
        forbidden_strings=task.forbidden_strings,
        unknown_expected=task.unknown_expected,
        variant=task.variant,
        repo_files=repo_files,
        repo_noise_lines=repo_noise_lines,
    )


def _noise_event(task: AgentTask, index: int, *, lines: int) -> str:
    line = (
        f"Noise event {index} for {task.family}. Generated filler module generated/filler_{index:04d}.py "
        f"mentions the task words but has no active decision. It discusses tests, routes, configs, owners, "
        f"dependency pins, invoice normalization, auth middleware, retry policy, and export jobs only as unrelated drafts."
    )
    return "\n".join(line for _ in range(max(1, lines)))


def _stale_event(task: AgentTask, index: int, *, lines: int) -> str:
    stale_by_family = {
        "auth_migration": "Stale auth draft says rejected_auth=LegacyApiKeyAuth for admin reports. It is old and must not be restored.",
        "billing_retry": "Stale billing draft says rejected_config=BILLING_RETRY_COUNT for retry policy. It is old and must not be restored.",
        "api_route_rename": "Stale API draft says rejected_endpoint=/api/v1/export for report export. It is old and must not be restored.",
        "test_failure_followup": "Stale export-jobs draft says use offset pagination. It conflicts with the accepted cursor pagination decision.",
        "stale_docs_current_code": "Stale invoice docs mention rejected_function=normalize_legacy_invoice. Current code supersedes this.",
        "dependency_version_pin": "Stale dependency draft mentions pydantic version 1.10.0. The accepted pin supersedes this.",
        "unknown_owner": "Observability planning note repeats that no owner was selected and no owner decision exists.",
    }
    line = f"Stale event {index}. {stale_by_family.get(task.family, 'Stale unrelated project note.')}"
    return "\n".join(line for _ in range(max(1, lines)))


def _create_repo_fixture(task: AgentTask, repo_path: Path) -> None:
    if repo_path.exists():
        shutil.rmtree(repo_path)
    repo_path.mkdir(parents=True)
    for package in ("src", "src/auth", "src/api", "src/api/routes", "src/billing", "src/invoices", "src/checkout", "src/ops", "tests"):
        _write(repo_path / package / "__init__.py", "")

    _write(repo_path / "src/auth/middleware.py", "class LegacyApiKeyAuth:\n    pass\n")
    _write(repo_path / "src/api/routes/admin_reports.py", "from src.auth.middleware import LegacyApiKeyAuth\n\nAUTH_MIDDLEWARE = LegacyApiKeyAuth\n")
    _write(repo_path / "tests/test_admin_reports.py", "import unittest\nfrom src.api.routes.admin_reports import AUTH_MIDDLEWARE\n\nclass AdminReportsTest(unittest.TestCase):\n    def test_session_token(self):\n        self.assertEqual(AUTH_MIDDLEWARE.__name__, 'RequireSessionToken')\n")

    _write(repo_path / "src/billing/retry_policy.py", "CONFIG_KEY = 'BILLING_RETRY_COUNT'\n")
    _write(repo_path / "tests/test_retry_policy.py", "import unittest\nfrom src.billing.retry_policy import CONFIG_KEY\n\nclass RetryPolicyTest(unittest.TestCase):\n    def test_payment_retry_limit(self):\n        self.assertEqual(CONFIG_KEY, 'PAYMENT_RETRY_LIMIT_0')\n")

    _write(repo_path / "src/api/routes/report_export.py", "EXPORT_ENDPOINT = '/api/v1/export'\n")
    _write(repo_path / "tests/test_report_export.py", "import unittest\nfrom src.api.routes.report_export import EXPORT_ENDPOINT\n\nclass ReportExportTest(unittest.TestCase):\n    def test_route_rename(self):\n        self.assertEqual(EXPORT_ENDPOINT, '/api/v3/reports/0/export')\n")

    _write(repo_path / "src/api/routes/export_jobs.py", "PAGINATION_MODE = 'offset'\nTIMEOUT_SECONDS = 5\n")
    _write(repo_path / "tests/test_export_jobs.py", "import unittest\nfrom src.api.routes.export_jobs import PAGINATION_MODE, TIMEOUT_SECONDS\n\nclass ExportJobsTest(unittest.TestCase):\n    def test_export_jobs_cursor_timeout_0(self):\n        self.assertEqual(PAGINATION_MODE, 'cursor')\n        self.assertGreaterEqual(TIMEOUT_SECONDS, 30)\n")

    _write(repo_path / "src/invoices/normalizer.py", "def normalize_legacy_invoice(invoice):\n    return invoice\n")
    _write(repo_path / "tests/test_normalizer.py", "import unittest\nfrom src.invoices import normalizer\n\nclass NormalizerTest(unittest.TestCase):\n    def test_active_invoice(self):\n        result = normalizer.normalize_active_invoice_0({'id': 'inv_1'})\n        self.assertEqual(result['status'], 'active')\n")

    _write(repo_path / "pyproject.toml", "[project]\ndependencies = ['pydantic==1.10.0']\n")
    _write(repo_path / "tests/test_dependency_pins.py", "import unittest\nfrom pathlib import Path\n\nclass DependencyPinsTest(unittest.TestCase):\n    def test_pydantic_pin(self):\n        self.assertIn('pydantic==2.7.0', Path('pyproject.toml').read_text())\n")

    _write(repo_path / "src/observability.py", "ALERT_OWNER = None\n")
    _write(repo_path / "tests/test_observability.py", "import unittest\n\nclass ObservabilityTest(unittest.TestCase):\n    def test_placeholder(self):\n        self.assertTrue(True)\n")

    # --- Files for cross-session task families ---
    # feature_flag_rollout: CHECKOUT_V2 flag progression
    _write(repo_path / "src/checkout/__init__.py", "")
    _write(repo_path / "src/checkout/flags.py", "CHECKOUT_V2 = 'disabled'\n")
    _write(repo_path / "tests/test_checkout_flags.py", "import unittest\nfrom src.checkout.flags import CHECKOUT_V2\n\nclass CheckoutFlagsTest(unittest.TestCase):\n    def test_checkout_v2_enabled(self):\n        self.assertEqual(CHECKOUT_V2, 'enabled')\n")

    # service_ownership: on-call owner chain
    _write(repo_path / "src/ops/__init__.py", "")
    _write(repo_path / "src/ops/oncall.py", "ON_CALL_OWNER = '@alice-chen'\n")
    _write(repo_path / "tests/test_oncall.py", "import unittest\nfrom src.ops.oncall import ON_CALL_OWNER\n\nclass OncallTest(unittest.TestCase):\n    def test_current_owner(self):\n        self.assertEqual(ON_CALL_OWNER, '@carol-wu')\n")

    for index in range(task.repo_files):
        module = repo_path / "src" / "generated" / f"filler_{index:04d}.py"
        body = "\n".join(
            f"# filler line {line} for generated module {index}: unrelated auth billing route dependency invoice export owner draft"
            for line in range(task.repo_noise_lines)
        )
        _write(module, body + "\nVALUE = 'unrelated'\n")


def _snapshot_files(repo_path: Path) -> dict[str, str]:
    snapshot = {}
    for path in sorted(repo_path.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            snapshot[str(path.relative_to(repo_path))] = path.read_text(encoding="utf-8")
    return snapshot


def _run_command(command: str, cwd: Path) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command.split(),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return completed.returncode == 0, completed.stdout


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repo_contains(root: Path, item: str, extra_text: str = "") -> bool:
    needle = item.casefold()
    if needle in extra_text.casefold() or contains_fact(extra_text, item):
        return True
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".md", ".toml", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if needle in text.casefold() or contains_fact(text, item):
            return True
    return False


def _select_families(families: str | Iterable[str]) -> list[str]:
    if isinstance(families, str):
        if families == "all":
            return list(FAMILIES)
        selected = [item.strip() for item in families.split(",") if item.strip()]
    else:
        selected = list(families)
    unknown = sorted(set(selected) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown families: {', '.join(unknown)}")
    return selected


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
