#!/usr/bin/env python3
"""Run one memory-backed coding agent on a real repo-patch eval workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARKS = Path(__file__).resolve().parent
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from agent_memory_adapters import make_adapter
from agent_task_harness import (
    AgentTask,
    OpenAIToolAgent,
    _repo_contains,
    _run_command,
)


def _read_memory_events(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    events: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            events.append(stripped[2:].strip())
    if not events:
        events = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return tuple(events)


def _git_changed_files(repo: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", "-C", str(repo), "diff", "--name-only"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        return ()
    return tuple(sorted(line.strip() for line in completed.stdout.splitlines() if line.strip()))


def _git_diff_stat(repo: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), "diff", "--stat"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout


def _git_diff(repo: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), "diff", "--", ".", ":(exclude)__pycache__"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout


def _snapshot_text_files(repo: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    allowed_suffixes = {".py", ".md", ".txt", ".toml", ".json", ".sh"}
    for path in sorted(repo.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix and path.suffix not in allowed_suffixes:
            continue
        try:
            snapshot[str(path.relative_to(repo))] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return snapshot


def _load_expected(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_forbidden_from_output(test_output: str) -> tuple[str, ...]:
    found: list[str] = []
    marker = "forbidden pattern remains:"
    for line in test_output.splitlines():
        if marker in line:
            found.append(line.split(marker, 1)[1].strip())
    return tuple(sorted(set(found)))


def _score_repo_patch_result(
    *,
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
    action_trace: tuple[str, ...],
) -> dict:
    changed_set = set(changed_files)
    expected_set = set(task.expected_files)
    forbidden_found = _active_forbidden_from_output(test_output)
    missing = tuple(sorted(expected_set - changed_set))
    success = tests_passed and not forbidden_found and not missing
    return {
        "task_id": task.task_id,
        "family": task.family,
        "adapter": adapter,
        "success": success,
        "tests_passed": tests_passed,
        "unknown_passed": True,
        "changed_files": list(changed_files),
        "expected_files_touched": sorted(expected_set & changed_set),
        "missing_expected_files": list(missing),
        "forbidden_found": list(forbidden_found),
        "hallucinated_files": sorted(path for path in changed_files if path not in expected_set),
        "steps": steps,
        "retries": max(0, steps - 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "memory_payload_tokens": memory_payload_tokens,
        "raw_context_tokens": raw_context_tokens,
        "prepared_context_tokens": prepared_context_tokens,
        "token_savings": token_savings,
        "duration_seconds": duration,
        "ingest_duration_seconds": ingest_duration,
        "final_message": final_message,
        "action_trace": list(action_trace),
    }


def run(
    *,
    eval_dir: Path,
    adapter_name: str,
    model: str,
    max_steps: int,
    out: Path,
    progress: bool,
) -> dict:
    workspace = eval_dir / "workspaces" / adapter_name
    if not workspace.exists():
        raise SystemExit(f"workspace does not exist: {workspace}")

    expected = _load_expected(eval_dir / "expected_outcomes.json")
    task_prompt = (eval_dir / "task_prompt.md").read_text(encoding="utf-8")
    memory_events = _read_memory_events(eval_dir / "memory_sessions.md")
    test_command = "bash run_eval.sh" if (workspace / "run_eval.sh").exists() else expected["test_command"]
    task = AgentTask(
        task_id=f"repo_patch_{adapter_name}",
        family="repo_patch_memory_eval",
        prompt=task_prompt,
        memory_events=memory_events,
        expected_files=tuple(expected["expected_files"]),
        test_command=test_command,
        forbidden_strings=(),
    )

    adapter = make_adapter(adapter_name)
    adapter.reset(task.task_id)
    agent = OpenAIToolAgent(model)
    before = _snapshot_text_files(workspace)

    try:
        ingest_started = time.monotonic()
        for event in memory_events:
            adapter.ingest_event(event, scope="project")
        ingest_duration = time.monotonic() - ingest_started

        prepared_before = adapter.prepare_context(task.prompt, workspace)
        if progress:
            print(
                f"adapter={adapter.name} model={model} workspace={workspace} "
                f"context_tokens={prepared_before.token_estimate} raw_tokens={prepared_before.raw_token_estimate}",
                flush=True,
            )

        run_started = time.monotonic()
        steps, input_tokens, output_tokens, final_message, action_trace = agent.run(
            task,
            workspace,
            adapter,
            max_steps=max_steps,
        )
        duration = time.monotonic() - run_started

        tests_passed, test_output = _run_command(task.test_command, workspace)
        after = _snapshot_text_files(workspace)
        changed_files = _git_changed_files(workspace) or tuple(
            sorted(path for path in after if before.get(path) != after.get(path))
        )
        result = _score_repo_patch_result(
            task=task,
            adapter=adapter.name,
            changed_files=changed_files,
            tests_passed=tests_passed,
            test_output=test_output,
            steps=steps,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            memory_payload_tokens=prepared_before.memory_payload_tokens,
            raw_context_tokens=prepared_before.raw_token_estimate,
            prepared_context_tokens=prepared_before.token_estimate,
            token_savings=prepared_before.token_savings,
            duration=duration,
            ingest_duration=ingest_duration,
            final_message=final_message,
            action_trace=action_trace,
        )

        result["workspace"] = str(workspace)
        result["model"] = model
        result["diff_stat"] = _git_diff_stat(workspace)
        result["diff"] = _git_diff(workspace)
        result["test_output"] = test_output[-6000:]
        result["required_values_found"] = [
            value for value in expected.get("required_values", ()) if _repo_contains(workspace, value, "")
        ]
        result["required_values_missing"] = [
            value for value in expected.get("required_values", ()) if not _repo_contains(workspace, value, "")
        ]
        result["summary"] = {
            "success": result["success"],
            "tests_passed": result["tests_passed"],
            "forbidden_found": result["forbidden_found"],
            "missing_expected_files": result["missing_expected_files"],
            "changed_files": result["changed_files"],
            "total_tokens": result["total_tokens"],
            "steps": result["steps"],
            "duration_seconds": result["duration_seconds"],
        }
    finally:
        adapter.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, help="Repo patch eval directory.")
    parser.add_argument("--adapter", required=True, help="Adapter/workspace name.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--out", help="JSON result path.")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out = Path(args.out) if args.out else eval_dir / "results" / f"{args.adapter}.json"
    result = run(
        eval_dir=eval_dir,
        adapter_name=args.adapter,
        model=args.model,
        max_steps=args.max_steps,
        out=out,
        progress=args.progress,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {out}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
