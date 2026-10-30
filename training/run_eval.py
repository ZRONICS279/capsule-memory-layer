"""Run local compression/precision evals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_capsule_memory.eval import load_cases, run_symbolic_eval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evals/symbolic_cases.jsonl")
    args = parser.parse_args()

    results = run_symbolic_eval(load_cases(args.cases))
    totals = {
        "cases": len(results),
        "questions": sum(result.total_questions for result in results),
        "correct": sum(result.correct for result in results),
        "unknown": sum(result.unknown for result in results),
        "conflicts": sum(result.conflicts for result in results),
    }
    totals["accuracy"] = totals["correct"] / max(totals["questions"], 1)
    totals["mean_compression_ratio"] = sum(result.compression_ratio for result in results) / max(len(results), 1)

    print(json.dumps({"summary": totals, "results": [result.to_json() for result in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
