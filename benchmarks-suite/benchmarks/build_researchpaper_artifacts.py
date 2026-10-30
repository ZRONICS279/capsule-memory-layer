"""Build a researchpaper/ artifact folder from benchmark reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


REPORTS = (
    ("long-horizon-retrieval", "reports-long-horizon-retrieval.json", "1. Long-Horizon Retrieval-Only Results"),
    ("long-horizon-agentic", "reports-long-horizon-agentic.json", "2. Long-Horizon Agentic Solving Results"),
    ("all-retrieval", "reports-all-retrieval.json", "3. Full Cross-Session Retrieval-Only Results"),
    ("all-agentic", "reports-all-agentic.json", "4. Full Cross-Session Agentic Benchmark Results"),
    ("persistent-retrieval", "reports-persistent-retrieval.json", "5. Persistent Two-Phase Memory Results"),
    ("chat-memory", "reports-chat-memory.json", "6. Long-Term / General Chat Memory Results"),
)


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="researchpaper")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _run(
        "describe benchmark suite",
        [
            sys.executable,
            "benchmarks/describe_benchmark_suite.py",
            "--families",
            "all",
            "--out",
            str(out_dir / "paper-benchmark-methods.md"),
        ],
    )

    appendix_parts = ["# Paper Benchmark Appendix\n\n"]
    appendix_parts.append((out_dir / "paper-benchmark-methods.md").read_text(encoding="utf-8"))
    post_parts = ["Agentic memory benchmark results\n"]

    for stem, report_name, title in REPORTS:
        report_path = ROOT / report_name
        if not report_path.exists():
            message = f"missing {report_name}; run the benchmark first"
            if args.allow_missing:
                print(message, file=sys.stderr)
                continue
            raise SystemExit(message)

        summary_path = out_dir / f"{stem}-summary.md"
        post_path = out_dir / f"{stem}-post.txt"
        _run(
            f"summarize {report_name}",
            [
                sys.executable,
                "benchmarks/summarize_benchmark_report.py",
                report_name,
                "--style",
                "paper",
                "--out",
                str(summary_path),
            ],
        )
        _run(
            f"post {report_name}",
            [
                sys.executable,
                "benchmarks/summarize_benchmark_report.py",
                report_name,
                "--style",
                "twitter",
                "--out",
                str(post_path),
            ],
        )
        appendix_parts.append(f"\n\n# {title}\n\n")
        appendix_parts.append(summary_path.read_text(encoding="utf-8"))
        post_parts.append("\n")
        post_parts.append(post_path.read_text(encoding="utf-8"))

        _copy_if_exists(report_path, out_dir / report_name)
        for suffix in (".csv", ".tex"):
            sibling = report_path.with_suffix(suffix)
            if sibling.exists():
                _copy_if_exists(sibling, out_dir / sibling.name)

    (out_dir / "paper-benchmark-appendix.md").write_text("".join(appendix_parts), encoding="utf-8")
    (out_dir / "benchmark-post-draft.txt").write_text("".join(post_parts), encoding="utf-8")

    expanded_report = ROOT / "reports-all-retrieval-expanded.json"
    if expanded_report.exists():
        _copy_if_exists(expanded_report, out_dir / expanded_report.name)
        for suffix in (".csv", ".tex"):
            sibling = expanded_report.with_suffix(suffix)
            if sibling.exists():
                _copy_if_exists(sibling, out_dir / sibling.name)

    _run(
        "render benchmark scorecard",
        [
            sys.executable,
            "benchmarks/render_benchmark_scorecard.py",
            "--out",
            str(out_dir / "benchmark-scorecard.png"),
            "--markdown-out",
            str(out_dir / "benchmark-scorecard.md"),
        ],
    )
    print(f"research artifacts written to {out_dir}")
    return 0


def _run(label: str, cmd: list[str]) -> None:
    completed = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{completed.stdout}")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


if __name__ == "__main__":
    raise SystemExit(main())
