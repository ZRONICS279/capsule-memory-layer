"""Evaluation harness for memory compression quality."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .capsule import Capsule
from .budget import estimate_tokens
from .runtime import AgentRuntime, TextSpineRuntime


@dataclass(frozen=True)
class EvalQuestion:
    key: str
    expected: str
    scope: str = "global"


@dataclass(frozen=True)
class EvalCase:
    capsule_id: str
    context: str
    questions: tuple[EvalQuestion, ...]
    scope: str = "global"


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    total_questions: int
    correct: int
    unknown: int
    conflicts: int
    original_tokens: int
    memory_tokens: int
    compression_ratio: float

    @property
    def accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.correct / self.total_questions

    def to_json(self) -> dict:
        return {
            "case_id": self.case_id,
            "total_questions": self.total_questions,
            "correct": self.correct,
            "unknown": self.unknown,
            "conflicts": self.conflicts,
            "accuracy": self.accuracy,
            "original_tokens": self.original_tokens,
            "memory_tokens": self.memory_tokens,
            "compression_ratio": self.compression_ratio,
        }


def load_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            questions = tuple(EvalQuestion(**item) for item in raw["questions"])
            cases.append(
                EvalCase(
                    capsule_id=raw["capsule_id"],
                    context=raw["context"],
                    questions=questions,
                    scope=raw.get("scope", "global"),
                )
            )
    return cases


def run_symbolic_eval(cases: list[EvalCase], runtime: AgentRuntime | None = None) -> list[EvalResult]:
    runtime = runtime or TextSpineRuntime()
    results: list[EvalResult] = []

    for case in cases:
        capsule = Capsule(case.capsule_id, f"@{case.capsule_id}").update(case.context, scope=case.scope)
        original_tokens = estimate_tokens(case.context)
        prepared = runtime.prepare_memory(capsule, original_token_estimate=original_tokens)

        correct = 0
        unknown = 0
        conflicts = 0
        for question in case.questions:
            answer = capsule.query_fact(question.key, scope=question.scope)
            if answer.status == "unknown":
                unknown += 1
            elif answer.status == "conflict":
                conflicts += 1
            elif answer.answer == question.expected:
                correct += 1

        memory_tokens = prepared.estimated_text_tokens or prepared.soft_token_count
        compression = float("inf") if memory_tokens == 0 else original_tokens / memory_tokens
        results.append(
            EvalResult(
                case_id=case.capsule_id,
                total_questions=len(case.questions),
                correct=correct,
                unknown=unknown,
                conflicts=conflicts,
                original_tokens=original_tokens,
                memory_tokens=memory_tokens,
                compression_ratio=compression,
            )
        )

    return results
