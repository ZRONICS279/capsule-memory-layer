"""Real repository patch benchmark scaffold.

This runner is intentionally data-driven. A case file describes a real repo,
base commit, memory sessions, expected files, forbidden patterns, and test
command. The runner can validate cases offline today; clone/install execution
is kept explicit because it requires network and project-specific dependency
setup.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


@dataclass(frozen=True)
class RealRepoCase:
    case_id: str
    repo_url: str
    base_ref: str
    prompt: str
    test_command: str
    memory_sessions: tuple[tuple[str, ...], ...]
    expected_files: tuple[str, ...]
    forbidden_patterns: tuple[str, ...] = ()
    issue_url: str = ""
    gold_pr_url: str = ""


def load_cases(path: Path) -> list[RealRepoCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
    if not isinstance(items, list):
        raise ValueError("case file must be a list or an object with a 'cases' list")
    return [_parse_case(item) for item in items]


def validate_cases(cases: Iterable[RealRepoCase]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            errors.append(f"{case.case_id}: duplicate case_id")
        seen.add(case.case_id)
        for field in ("case_id", "repo_url", "base_ref", "prompt", "test_command"):
            if not getattr(case, field):
                errors.append(f"{case.case_id or '<missing>'}: missing {field}")
        if not case.memory_sessions or not any(case.memory_sessions):
            errors.append(f"{case.case_id}: memory_sessions must contain at least one event")
        if not case.expected_files:
            errors.append(f"{case.case_id}: expected_files must not be empty")
    return errors


def repo_cache_path(cache_dir: Path, case: RealRepoCase) -> Path:
    slug = case.repo_url.rstrip("/").removesuffix(".git").split("/")[-1]
    owner = case.repo_url.rstrip("/").removesuffix(".git").split("/")[-2]
    return cache_dir / f"{owner}__{slug}"


def ensure_repo_cache(case: RealRepoCase, cache_dir: Path) -> Path:
    target = repo_cache_path(cache_dir, case)
    if target.exists():
        _run(("git", "fetch", "--all", "--tags"), cwd=target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        _run(("git", "clone", case.repo_url, str(target)), cwd=cache_dir)
    _run(("git", "checkout", case.base_ref), cwd=target)
    return target


def _parse_case(item: dict) -> RealRepoCase:
    required = ("case_id", "repo_url", "base_ref", "prompt", "test_command", "memory_sessions", "expected_files")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"case missing required keys: {', '.join(missing)}")
    sessions = tuple(tuple(str(event) for event in session) for session in item["memory_sessions"])
    return RealRepoCase(
        case_id=str(item["case_id"]),
        repo_url=str(item["repo_url"]),
        base_ref=str(item["base_ref"]),
        prompt=str(item["prompt"]),
        test_command=str(item["test_command"]),
        memory_sessions=sessions,
        expected_files=tuple(str(path) for path in item["expected_files"]),
        forbidden_patterns=tuple(str(pattern) for pattern in item.get("forbidden_patterns", ())),
        issue_url=str(item.get("issue_url", "")),
        gold_pr_url=str(item.get("gold_pr_url", "")),
    )


def _run(args: tuple[str, ...], *, cwd: Path) -> None:
    completed = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed in {cwd}:\n{completed.stdout}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="Path to a JSON case file.")
    parser.add_argument("--cache-dir", default=".real-repo-cache")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--prepare-cache",
        action="store_true",
        help="Clone/fetch repositories and checkout base refs. Requires network for uncached repos.",
    )
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases))
    errors = validate_cases(cases)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2

    if args.prepare_cache:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            path = ensure_repo_cache(case, cache_dir)
            print(f"{case.case_id}: cached at {path}")
    elif args.validate_only:
        print(f"validated {len(cases)} real-repo cases")
    else:
        print("case validation passed; use --prepare-cache to clone/fetch repositories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
