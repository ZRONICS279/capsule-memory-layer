"""Describe benchmark tasks in a paper-ready way."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from cross_session_benchmark import CrossSessionTask, load_tasks


FAMILY_PURPOSES = {
    "auth_decision_evolution": (
        "Tests whether memory preserves the final auth decision after an earlier auth approach was superseded."
    ),
    "config_iteration": (
        "Tests whether memory keeps the final config key after multiple rejected naming iterations."
    ),
    "route_migration": (
        "Tests whether memory recalls the final API endpoint and file path while suppressing deprecated routes."
    ),
    "feature_flag_rollout": (
        "Tests whether memory handles changing rollout states such as disabled, canary, reverted, and enabled."
    ),
    "service_ownership": (
        "Tests whether memory resolves ownership changes and returns only the current owner."
    ),
    "export_jobs_config": (
        "Tests multi-fact recall: two final config values must be retrieved together without stale defaults."
    ),
    "stale_docs_current_code": (
        "Tests stale-document conflict handling: the agent must follow current code/test facts, not old docs."
    ),
    "dependency_version_pin": (
        "Tests dependency decision memory after older package versions were superseded."
    ),
    "unknown_owner_abstention": (
        "Tests abstention: memory should not invent a fact when no final owner decision exists."
    ),
    "long_horizon_project_memory": (
        "Tests 100+ event project memory under heavy noise, repeated stale facts, and multiple current facts."
    ),
}


def describe_tasks(tasks: list[CrossSessionTask], *, style: str = "markdown") -> str:
    if style == "markdown":
        return _markdown(tasks)
    if style == "plain":
        return _plain(tasks)
    raise ValueError(f"unknown style {style!r}")


def _markdown(tasks: list[CrossSessionTask]) -> str:
    lines = [
        "# Benchmark Suite Description",
        "",
        "This suite evaluates agentic coding memory under evolving project history. Each task pre-ingests prior work sessions into a memory adapter, then asks the agent or retrieval layer to use only retrieved memory, not the raw transcript.",
        "",
        "## Task Families",
        "",
    ]
    for task in tasks:
        lines.extend(_task_markdown(task))
        lines.append("")
    lines.extend(
        [
            "## What The Scores Mean",
            "",
            "- Recall: fraction of required current facts found in retrieved memory.",
            "- Stale: fraction of superseded facts that leaked into active retrieved memory.",
            "- Score: with `--score-formula multiplicative`, `recall * (1 - stale)`.",
            "- CtxOK: strict pass where all required facts are present and no stale facts leaked.",
            "- Success: agentic run only; tests pass, expected files are changed, forbidden patterns are absent.",
            "- Token savings: prepared memory context compared with injecting the full raw session history.",
            "",
            "## What This Proves",
            "",
            "A win on retrieval-only runs proves cleaner memory retrieval: more current facts, fewer stale facts, fewer tokens. A win on agentic runs proves downstream task value: the coding agent solved more tasks or solved them with lower memory/context cost. These results support a claim about cross-session software-engineering memory, not universal memory superiority.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _task_markdown(task: CrossSessionTask) -> list[str]:
    session_events = sum(len(session) for session in task.sessions)
    return [
        f"### `{task.family}`",
        "",
        FAMILY_PURPOSES.get(task.family, "Tests cross-session project memory behavior."),
        "",
        f"- Task id example: `{task.task_id}`",
        f"- Prior sessions: {len(task.sessions)}",
        f"- Memory events: {session_events}",
        f"- Agent prompt: {task.prompt}",
        f"- Expected files: {', '.join(f'`{path}`' for path in task.expected_files) or '`none`'}",
        f"- Test command: `{task.test_command}`",
        f"- Required current facts: {', '.join(f'`{fact}`' for fact in task.required_in_context) or '`none`'}",
        f"- Stale facts penalized: {', '.join(f'`{fact}`' for fact in task.stale_in_context) or '`none`'}",
        f"- Forbidden output/code patterns: {', '.join(f'`{fact}`' for fact in task.forbidden_strings) or '`none`'}",
        f"- Abstention expected: `{task.unknown_expected}`",
    ]


def _plain(tasks: list[CrossSessionTask]) -> str:
    chunks = []
    for task in tasks:
        chunks.append(
            "\n".join(
                [
                    task.family,
                    FAMILY_PURPOSES.get(task.family, "Tests cross-session project memory behavior."),
                    f"Prompt: {task.prompt}",
                    f"Events: {sum(len(session) for session in task.sessions)}",
                    f"Expected files: {', '.join(task.expected_files) or 'none'}",
                    f"Required facts: {', '.join(task.required_in_context) or 'none'}",
                    f"Stale facts: {', '.join(task.stale_in_context) or 'none'}",
                ]
            )
        )
    return "\n\n".join(chunks) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="all")
    parser.add_argument("--cases-per-family", type=int, default=1)
    parser.add_argument("--style", choices=("markdown", "plain"), default="markdown")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    tasks = load_tasks(families=args.families, cases_per_family=args.cases_per_family)
    text = describe_tasks(tasks, style=args.style)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
