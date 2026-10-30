"""Fail if the working memory core misses the minimum benchmark gate."""

from __future__ import annotations

from agentic_cost_benchmark import run


if __name__ == "__main__":
    result = run()
    if result["accuracy"] < 1.0:
        raise SystemExit(f"accuracy gate failed: {result['accuracy']}")
    if result["average_reduction"] < 0.60:
        raise SystemExit(f"reduction gate failed: {result['average_reduction']}")
    print(
        f"passed: accuracy={result['accuracy']:.3f} "
        f"average_reduction={result['average_reduction']:.3f}"
    )

