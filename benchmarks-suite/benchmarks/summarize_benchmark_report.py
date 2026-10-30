"""Summarize benchmark JSON reports for papers, READMEs, and short posts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_report(report: dict, *, style: str = "paper") -> str:
    summary = report.get("summary", {})
    scores = summary.get("adapter_scores", {})
    adapters = summary.get("adapters") or list(scores)
    rows = [_adapter_row(adapter, scores.get(adapter, {})) for adapter in adapters]

    if style == "twitter":
        return _twitter_summary(summary, rows)
    if style == "markdown":
        return _markdown_summary(summary, rows, report)
    if style == "paper":
        return _paper_summary(summary, rows, report)
    raise ValueError(f"unknown style {style!r}")


def _adapter_row(adapter: str, score: dict) -> dict:
    return {
        "adapter": adapter,
        "success": score.get("success_rate"),
        "memory_success": score.get("memory_success_rate"),
        "ctxok": score.get("context_success_rate", 0.0),
        "recall": score.get("avg_recall", 0.0),
        "stale": score.get("avg_stale_penalty", 0.0),
        "privacy": score.get("avg_privacy_penalty", 0.0),
        "deletion": score.get("avg_deletion_penalty", 0.0),
        "score": score.get("avg_cross_session_score", 0.0),
        "raw_tokens": score.get("avg_raw_context_tokens", 0),
        "prepared_tokens": score.get("avg_prepared_context_tokens", 0),
        "token_savings": score.get("avg_token_savings", 0.0),
    }


def _paper_summary(summary: dict, rows: list[dict], report: dict) -> str:
    lines = [
        "# Benchmark Summary",
        "",
        _claim_line(summary, rows),
        "",
        "## What Was Tested",
        "",
        _workload_line(report),
        "",
        "## Results",
        "",
        _markdown_table(rows, include_chat_penalties=summary.get("benchmark") == "chat_memory"),
        "",
        "## Plain-English Comparison",
        "",
        _plain_english_comparison(rows),
        "",
        "## Interpretation",
        "",
        _interpretation(rows),
    ]
    caveat = _caveat(report, rows)
    if caveat:
        lines.extend(["", "## Caveat", "", caveat])
    return "\n".join(lines).rstrip() + "\n"


def _markdown_summary(summary: dict, rows: list[dict], report: dict) -> str:
    lines = [
        "## Benchmark Results",
        "",
        _workload_line(report),
        "",
        _markdown_table(rows, include_chat_penalties=summary.get("benchmark") == "chat_memory"),
        "",
        _claim_line(summary, rows),
        "",
        _plain_english_comparison(rows),
    ]
    caveat = _caveat(report, rows)
    if caveat:
        lines.extend(["", f"Caveat: {caveat}"])
    return "\n".join(lines).rstrip() + "\n"


def _twitter_summary(summary: dict, rows: list[dict]) -> str:
    best = max(rows, key=lambda row: row["score"], default=None)
    capsule = next((row for row in rows if row["adapter"] == "capsule"), None)
    mem0 = _mem0_row(rows)
    lines = ["Agentic memory benchmark:"]
    if capsule and mem0:
        stale_drop = mem0["stale"] - capsule["stale"]
        score_gain = capsule["score"] - mem0["score"]
        lines.append(
            f"capsule vs {mem0['adapter']}: {_score_lift_text(capsule['score'], mem0['score'])}, "
            f"{stale_drop:.1%} less stale leakage."
        )
    elif best:
        lines.append(f"best adapter: {best['adapter']} score={best['score']:.3f}.")
    if capsule:
        lines.append(
            f"capsule recall={capsule['recall']:.3f}, stale={capsule['stale']:.3f}, "
            f"token savings={capsule['token_savings']:.1%}."
        )
    lines.append("Workload: cross-session software-engineering memory with stale facts and long project history.")
    return "\n".join(lines) + "\n"


def _markdown_table(rows: list[dict], *, include_chat_penalties: bool = False) -> str:
    if include_chat_penalties:
        lines = [
            "| Adapter | Success | MemSuccess | CtxOK | Recall | Stale | Privacy | Delete | Score | Raw Tok | Prep Tok | Savings |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "| Adapter | Success | MemSuccess | CtxOK | Recall | Stale | Score | Raw Tok | Prep Tok | Savings |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    for row in rows:
        success = "n/a" if row["success"] is None else f"{row['success']:.3f}"
        memory_success = "n/a" if row["memory_success"] is None else f"{row['memory_success']:.3f}"
        if include_chat_penalties:
            lines.append(
                f"| {row['adapter']} | {success} | {memory_success} | {row['ctxok']:.3f} | "
                f"{row['recall']:.3f} | {row['stale']:.3f} | {row['privacy']:.3f} | "
                f"{row['deletion']:.3f} | {row['score']:.3f} | "
                f"{row['raw_tokens']:.0f} | {row['prepared_tokens']:.0f} | {row['token_savings']:.1%} |"
            )
        else:
            lines.append(
                f"| {row['adapter']} | {success} | {memory_success} | {row['ctxok']:.3f} | "
                f"{row['recall']:.3f} | {row['stale']:.3f} | {row['score']:.3f} | "
                f"{row['raw_tokens']:.0f} | {row['prepared_tokens']:.0f} | {row['token_savings']:.1%} |"
            )
    return "\n".join(lines)


def _claim_line(summary: dict, rows: list[dict]) -> str:
    paired = summary.get("paired_deltas") or {}
    if paired:
        delta = paired.get("avg_cross_session_score_delta", 0.0)
        ci = paired.get("bootstrap_ci", {})
        capsule = next((row for row in rows if row["adapter"] == "capsule"), None)
        mem0 = _mem0_row(rows)
        lift = ""
        if capsule and mem0:
            lift = f"; {_score_lift_text(capsule['score'], mem0['score'])}"
        return (
            f"Capsule beat {paired.get('right')} by {delta:+.3f} average cross-session score "
            f"over {paired.get('pairs')} paired runs"
            f"{lift}"
            f"{_ci_text(ci)}."
        )
    best = summary.get("best_adapter", "unknown")
    score = summary.get("best_cross_session_score", 0.0)
    return f"Best adapter: {best} with cross-session score {score:.3f}."


def _ci_text(ci: dict) -> str:
    if not ci:
        return ""
    return f" (95% bootstrap CI {ci.get('low', 0):+.3f} to {ci.get('high', 0):+.3f})"


def _score_lift_text(left_score: float, right_score: float) -> str:
    delta = left_score - right_score
    points = delta * 100
    if right_score <= 0:
        return f"{points:+.1f} percentage points higher score"
    multiple = left_score / right_score
    relative = (multiple - 1.0) * 100
    return f"{points:+.1f} percentage points higher score ({multiple:.1f}x, {relative:+.0f}% relative)"


def _plain_english_comparison(rows: list[dict]) -> str:
    capsule = next((row for row in rows if row["adapter"] == "capsule"), None)
    if not capsule:
        return "No capsule row is available in this report."

    baselines = [row for row in rows if row["adapter"] != "capsule" and row.get("adapter")]
    if not baselines:
        return "No baseline rows are available in this report."

    lines = ["Capsule compared with every baseline:"]
    for baseline in baselines:
        lines.append("")
        lines.append(f"Compared with `{baseline['adapter']}`, capsule:")
        if capsule["success"] is not None and baseline["success"] is not None:
            lines.append(f"- Agent task success: {_higher_metric(capsule['success'], baseline['success'])}.")
        if capsule["memory_success"] is not None and baseline["memory_success"] is not None:
            lines.append(
                f"- Memory-grounded task success (`MemSuccess`): "
                f"{_higher_metric(capsule['memory_success'], baseline['memory_success'])}."
            )
        lines.extend(
            [
                f"- Strict clean-context pass rate (`CtxOK`): {_higher_metric(capsule['ctxok'], baseline['ctxok'])}.",
                f"- Current-fact recall: {_higher_metric(capsule['recall'], baseline['recall'])}.",
                f"- Stale fact leakage: {_lower_metric(capsule['stale'], baseline['stale'])}.",
                f"- Cross-session score: {_higher_metric(capsule['score'], baseline['score'])}.",
                f"- Token savings: {_higher_metric(capsule['token_savings'], baseline['token_savings'])}.",
            ]
        )
        if baseline.get("privacy", 0.0) or capsule.get("privacy", 0.0):
            lines.append(f"- Privacy leakage: {_lower_metric(capsule['privacy'], baseline['privacy'])}.")
        if baseline.get("deletion", 0.0) or capsule.get("deletion", 0.0):
            lines.append(f"- Deletion/forgetting leakage: {_lower_metric(capsule['deletion'], baseline['deletion'])}.")
    return "\n".join(lines)


def _higher_metric(left: float, right: float) -> str:
    delta = left - right
    points = delta * 100
    if right <= 0:
        return f"{left:.1%} vs {right:.1%} ({points:+.1f} percentage points)"
    multiple = left / right
    relative = (multiple - 1.0) * 100
    return f"{left:.1%} vs {right:.1%} ({points:+.1f} points, {multiple:.1f}x, {relative:+.0f}% relative)"


def _lower_metric(left: float, right: float) -> str:
    delta = right - left
    points = delta * 100
    if right <= 0:
        return f"{left:.1%} vs {right:.1%} ({points:+.1f} percentage points lower)"
    reduction = (1.0 - left / right) * 100
    return f"{left:.1%} vs {right:.1%} ({points:.1f} points lower, {reduction:.0f}% reduction)"


def _workload_line(report: dict) -> str:
    results = report.get("results", [])
    families = sorted({row.get("family", "unknown") for row in results})
    tasks = len({row.get("task_id") for row in results})
    runs = len(results)
    modes = sorted({row.get("mode", "agentic") for row in results})
    return (
        f"{tasks} task(s), {runs} adapter run(s), mode(s): {', '.join(modes)}. "
        f"Families: {', '.join(families)}."
    )


def _caveat(report: dict, rows: list[dict]) -> str:
    modes = {row.get("mode", "agentic") for row in report.get("results", [])}
    non_persistent_empty = [
        row["adapter"]
        for row in rows
        if row["adapter"] in {"bm25_rag", "no_memory"}
        and row["recall"] == 0
        and row["prepared_tokens"] == 0
        and "agentic" in modes
    ]
    if non_persistent_empty:
        names = ", ".join(non_persistent_empty)
        return (
            f"{names} have no persistent store in skip-ingest/two-phase runs, so their retrieval rows are "
            "expected to be empty. Use these rows as lower-bound/product-realism baselines, not as a fresh-ingest RAG comparison."
        )
    return ""


def _interpretation(rows: list[dict]) -> str:
    capsule = next((row for row in rows if row["adapter"] == "capsule"), None)
    mem0 = _mem0_row(rows)
    if capsule and mem0:
        return (
            "The result supports a narrow claim about cross-session agentic coding memory: "
            "current-fact recall, stale-fact suppression, and token efficiency under evolving project history. "
            "It does not by itself prove global superiority across every memory workload."
        )
    return (
        "Use this table to compare current-fact recall, stale leakage, context success, and token savings. "
        "Include mem0_real or mem0_real_async in the run for a direct capsule-vs-mem0 claim."
    )


def _mem0_row(rows: list[dict]) -> dict | None:
    return (
        next((row for row in rows if row["adapter"] == "mem0_real"), None)
        or next((row for row in rows if row["adapter"] == "mem0_real_async"), None)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to benchmark JSON report.")
    parser.add_argument("--style", choices=("paper", "markdown", "twitter"), default="paper")
    parser.add_argument("--out", default="", help="Optional output file.")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    text = summarize_report(report, style=args.style)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
