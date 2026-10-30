"""Deterministic benchmark for the working agentic memory core."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_capsule_memory.agentic import AgenticMemoryLayer


QUESTIONS = {
    "What board should the greenhouse target?": "ESP32-C3",
    "What language should code use?": "MicroPython",
    "What pump voltage should I use?": "12V",
    "What pump current should I plan for?": "2A",
    "What should I avoid using?": "Arduino C++",
}


def build_context() -> str:
    noise = []
    for index in range(800):
        noise.append(
            f"Turn {index}: agent reviewed logs, discussed options, and noted no durable decision for greenhouse control."
        )
    facts = [
        "Earlier draft used ESP32, but correction: use ESP32-C3, not regular ESP32.",
        "Final decision: language: MicroPython.",
        "Pump voltage: 12V.",
        "Pump current: 2A.",
        "Do not use Arduino C++.",
        "Pin 12 = BCM GPIO 18.",
    ]
    return "\n".join(noise + facts)


def run() -> dict:
    context = build_context()
    with TemporaryDirectory() as directory:
        memory = AgenticMemoryLayer(Path(directory) / "memory.sqlite", default_token_budget=192, target_reduction=0.60)
        memory.remember("greenhouse", context, scope="greenhouse", label="@greenhouse")

        results = []
        correct = 0
        for question, expected in QUESTIONS.items():
            packet = memory.prepare_step("greenhouse", question, original_context=context)
            answer = memory.answer("greenhouse", question, original_context=context)
            if answer.answer == expected:
                correct += 1
            results.append(
                {
                    "question": question,
                    "expected": expected,
                    "answer": answer.answer,
                    "status": answer.status,
                    "raw_tokens": packet.raw_tokens,
                    "memory_tokens": packet.memory_tokens,
                    "reduction": round(packet.reduction, 4),
                    "passed_target": packet.passed_target,
                }
            )

    avg_reduction = sum(item["reduction"] for item in results) / len(results)
    return {
        "questions": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "average_reduction": round(avg_reduction, 4),
        "passed_60_percent_reduction": avg_reduction >= 0.60,
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

