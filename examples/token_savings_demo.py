"""Estimate repeated prompt-token and input-cost savings for capsule memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_capsule_memory.agentic import AgenticMemoryLayer
from semantic_capsule_memory.savings import estimate_input_savings


DEMO_IMPORTANT = (
    "Capsule Memory demo record. We are testing repeated prompt-token savings for agent memory. "
    "Final decision: use capsule_id codex_savings_demo for this test. "
    "The important project fact is that memory_prepare_step reports raw_tokens, memory_tokens, "
    "reduction, target_reduction, and passed_target. "
    "Cost savings should compare the original full context input cost against the compiled memory packet input cost. "
    "The demo should expose estimated tokens saved, percent reduction, baseline input cost, memory input cost, "
    "and estimated dollar savings."
)


def build_demo_context(repetitions: int) -> str:
    noise = (
        "Repeated transcript noise about terminal output, planning notes, old tool results, "
        "and irrelevant implementation chatter. "
    )
    return (noise * repetitions) + DEMO_IMPORTANT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", help="SQLite store path. Defaults to a temporary store.")
    parser.add_argument("--capsule-id", default="codex_savings_demo")
    parser.add_argument("--scope", default="demo")
    parser.add_argument("--task", default="How should we test token and cost savings for repeated memory prompts?")
    parser.add_argument("--context-file", help="Use this file as the full original context.")
    parser.add_argument("--demo-repetitions", type=int, default=500, help="Noise repetitions for generated demo context.")
    parser.add_argument("--turns", type=int, default=100, help="Repeated turns to estimate.")
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=1.0,
        help="Input-token price per 1M tokens. Use current model pricing here.",
    )
    parser.add_argument("--retrieve-limit", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def run(store: Path, args: argparse.Namespace) -> dict:
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")
    else:
        context = build_demo_context(args.demo_repetitions)

    memory = AgenticMemoryLayer(store)
    memory.remember(args.capsule_id, context, scope=args.scope, label=f"@{args.capsule_id}")
    packet = memory.prepare_step(
        args.capsule_id,
        args.task,
        original_context=context,
        retrieve_limit=args.retrieve_limit,
    )
    savings = estimate_input_savings(packet, input_price_per_1m=args.input_price_per_1m, turns=args.turns)

    return {
        "capsule_id": args.capsule_id,
        "task": args.task,
        "passed_target": packet.passed_target,
        "target_reduction": packet.target_reduction,
        "savings": savings.to_json(),
        "payload": packet.payload,
    }


def print_human(result: dict) -> None:
    savings = result["savings"]
    print(f"Capsule: {result['capsule_id']}")
    print(f"Task: {result['task']}")
    print(f"Target passed: {result['passed_target']} (target={result['target_reduction']:.0%})")
    print()
    print(f"Raw tokens/turn: {savings['raw_tokens_per_turn']}")
    print(f"Memory tokens/turn: {savings['memory_tokens_per_turn']}")
    print(f"Tokens saved/turn: {savings['tokens_saved_per_turn']}")
    print(f"Reduction: {savings['token_reduction']:.1%}")
    print()
    print(f"Turns: {savings['turns']}")
    print(f"Input price: ${savings['input_price_per_1m']:.6f} per 1M tokens")
    print(f"Baseline input cost: ${savings['baseline_input_cost']:.6f}")
    print(f"Memory input cost: ${savings['memory_input_cost']:.6f}")
    print(f"Estimated input savings: ${savings['input_cost_saved']:.6f}")


def main() -> int:
    args = parse_args()
    if args.store:
        result = run(Path(args.store), args)
    else:
        with TemporaryDirectory() as directory:
            result = run(Path(directory) / "memory.sqlite", args)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
