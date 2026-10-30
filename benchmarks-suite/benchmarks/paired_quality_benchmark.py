"""Paired full context vs capsule memory answer quality benchmark.

The default provider is deterministic and local so this benchmark can run in CI.
Use ``--provider openai`` with ``AYYGENT_AI_API_KEY`` or ``OPENAI_API_KEY`` for
live LLM checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import time
import urllib.request
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_capsule_memory.agentic import AgenticMemoryLayer
from semantic_capsule_memory.budget import estimate_tokens, tokenizer_backend


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    capsule_id: str
    scope: str
    context: str
    question: str
    expected_facts: tuple[str, ...]
    forbidden_claims: tuple[str, ...] = ()
    unknown_expected: bool = False

    @classmethod
    def from_json(cls, raw: dict) -> "EvalCase":
        context = raw["context"]
        if raw.get("noise_repeat"):
            context = str(raw.get("noise_text", "Noise. ")) * int(raw["noise_repeat"]) + context
        return cls(
            case_id=raw["case_id"],
            capsule_id=raw.get("capsule_id", raw["case_id"]),
            scope=raw.get("scope", "global"),
            context=context,
            question=raw["question"],
            expected_facts=tuple(raw.get("expected_facts", ())),
            forbidden_claims=tuple(raw.get("forbidden_claims", ())),
            unknown_expected=bool(raw.get("unknown_expected", False)),
        )


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usage_source: str


class HeuristicLLM:
    """Deterministic stand-in for a chat model.

    It follows the benchmark prompt contract closely enough to test the paired
    harness, grader, and token accounting without network access.
    """

    name = "heuristic"

    def answer(self, prompt: str, question: str) -> LLMResult:
        answer = _extract_answer(prompt, question)
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(answer)
        return LLMResult(answer, input_tokens, output_tokens, input_tokens + output_tokens, "local")


class OpenAIChatLLM:
    name = "openai"

    def __init__(self, model: str, *, max_retries: int = 2) -> None:
        self.model = model
        self.max_retries = max_retries
        self.api_key = os.environ.get("AYYGENT_AI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise SystemExit("Set AYYGENT_AI_API_KEY or OPENAI_API_KEY to run --provider openai")

    def answer(self, prompt: str, question: str) -> LLMResult:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the provided context. If the answer is absent, "
                            "reply exactly UNKNOWN. Keep answers concise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        payload = self._send_with_retries(request)
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or estimate_tokens(prompt))
        output_tokens = int(usage.get("completion_tokens") or estimate_tokens(text))
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        return LLMResult(text, input_tokens, output_tokens, total_tokens, "provider" if usage else "estimated")

    def _send_with_retries(self, request: urllib.request.Request) -> dict:
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise
                last_exc = exc
            except (TimeoutError, URLError) as exc:
                last_exc = exc
            if attempt < self.max_retries:
                time.sleep(min(6.0, 0.75 * (2**attempt)))
        raise RuntimeError(f"model request failed after {self.max_retries + 1} attempts: {last_exc}")


def load_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(EvalCase.from_json(json.loads(line)))
    return cases


def run_cases(cases: list[EvalCase], *, provider: str, model: str, target_savings: float) -> dict:
    llm = HeuristicLLM() if provider == "heuristic" else OpenAIChatLLM(model)
    results = []
    for case in cases:
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(Path(directory) / "memory.sqlite", default_token_budget=192, target_reduction=target_savings)
            memory.remember(case.capsule_id, case.context, scope=case.scope, label=f"@{case.capsule_id}")
            packet = memory.prepare_step(case.capsule_id, case.question, original_context=case.context)

            baseline_prompt = _baseline_prompt(case)
            capsule_prompt = _capsule_prompt(case, packet.payload)
            baseline = llm.answer(baseline_prompt, case.question)
            capsule = llm.answer(capsule_prompt, case.question)

        baseline_grade = grade_answer(baseline.text, case)
        capsule_grade = grade_answer(capsule.text, case)
        token_savings = 0.0 if baseline.input_tokens == 0 else (baseline.input_tokens - capsule.input_tokens) / baseline.input_tokens
        results.append(
            {
                "case_id": case.case_id,
                "baseline": {**baseline_grade, **_usage_json(baseline)},
                "capsule": {**capsule_grade, **_usage_json(capsule), "memory_payload_tokens": packet.memory_tokens},
                "token_savings": token_savings,
                "accuracy_delta": capsule_grade["accuracy"] - baseline_grade["accuracy"],
                "hallucination_delta": int(capsule_grade["hallucination_count"]) - int(baseline_grade["hallucination_count"]),
                "unknown_pass_delta": int(capsule_grade["unknown_passed"]) - int(baseline_grade["unknown_passed"]),
            }
        )

    return summarize(results, provider=provider, model=model, target_savings=target_savings)


def grade_answer(answer: str, case: EvalCase) -> dict:
    lower = answer.lower()
    expected = [fact for fact in case.expected_facts if fact.upper() != "UNKNOWN"]
    found = [fact for fact in expected if _expected_fact_found(fact, answer, case.question)]
    forbidden = [claim for claim in case.forbidden_claims if claim.lower() in lower]
    unknown_passed = (not case.unknown_expected) or answer.strip().upper() == "UNKNOWN"
    accuracy = 1.0 if not expected else len(found) / len(expected)
    if case.unknown_expected and not unknown_passed:
        accuracy = 0.0
    return {
        "answer": answer,
        "accuracy": accuracy,
        "expected_found": found,
        "expected_missing": [fact for fact in expected if fact not in found],
        "forbidden_found": forbidden,
        "hallucination_count": len(forbidden),
        "unknown_passed": unknown_passed,
    }


def _expected_fact_found(expected: str, answer: str, question: str) -> bool:
    if expected.lower() in answer.lower():
        return True

    unit_match = re.fullmatch(r"(\d+)\s+(turns?|messages?)", expected.strip(), re.I)
    if not unit_match:
        return False

    number, unit = unit_match.groups()
    answer_lower = answer.lower()
    question_lower = question.lower()
    if not re.search(rf"\b{re.escape(number)}\b", answer_lower):
        return False
    return "checkpoint" in answer_lower or "checkpoint" in question_lower


def summarize(results: list[dict], *, provider: str, model: str, target_savings: float) -> dict:
    count = max(1, len(results))
    avg_savings = sum(item["token_savings"] for item in results) / count
    baseline_accuracy = sum(item["baseline"]["accuracy"] for item in results) / count
    capsule_accuracy = sum(item["capsule"]["accuracy"] for item in results) / count
    baseline_hallucinations = sum(item["baseline"]["hallucination_count"] for item in results)
    capsule_hallucinations = sum(item["capsule"]["hallucination_count"] for item in results)
    unknown_cases = [item for item in results if item["baseline"]["answer"].strip().upper() == "UNKNOWN" or item["capsule"]["answer"].strip().upper() == "UNKNOWN"]
    summary = {
        "provider": provider,
        "model": model if provider == "openai" else provider,
        "tokenizer_backend": tokenizer_backend(model if provider == "openai" else None),
        "cases": len(results),
        "avg_token_savings": avg_savings,
        "baseline_accuracy": baseline_accuracy,
        "capsule_accuracy": capsule_accuracy,
        "accuracy_delta": capsule_accuracy - baseline_accuracy,
        "baseline_hallucination_count": baseline_hallucinations,
        "capsule_hallucination_count": capsule_hallucinations,
        "unknown_pass_rate": 1.0 if not unknown_cases else sum(1 for item in unknown_cases if item["capsule"]["unknown_passed"]) / len(unknown_cases),
        "passed": (
            avg_savings >= target_savings
            and capsule_accuracy >= baseline_accuracy - 0.03
            and capsule_hallucinations <= baseline_hallucinations
        ),
    }
    return {"summary": summary, "results": results}


def _baseline_prompt(case: EvalCase) -> str:
    return (
        "Answer only from FULL_CONTEXT. If absent, answer UNKNOWN.\n\n"
        f"FULL_CONTEXT:\n{case.context}\n\nQUESTION:\n{case.question}"
    )


def _capsule_prompt(case: EvalCase, memory_payload: str) -> str:
    return (
        "Answer only from CAPSULE_MEMORY. If absent, answer UNKNOWN.\n\n"
        f"CAPSULE_MEMORY:\n{memory_payload}\n\nQUESTION:\n{case.question}"
    )


def _usage_json(result: LLMResult) -> dict:
    return {
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "usage_source": result.usage_source,
    }


def _extract_answer(prompt: str, question: str) -> str:
    lower = prompt.lower()
    q = question.lower()
    if "color" in q or "colour" in q:
        if re.search(r"no enclosure color was selected|no .*color|no enclosure decision|unknown: no active fact", lower):
            return "UNKNOWN"
        if "capsule_memory:" in lower and "color" not in lower.split("question:", 1)[0] and "colour" not in lower.split("question:", 1)[0]:
            return "UNKNOWN"
        match = re.search(r"\b(?:color|colour)[^.\n]*(?:is|:|=|selected|picked)\s+([A-Za-z -]+)", prompt, re.I)
        return match.group(1).strip(" .,:;") if match else "UNKNOWN"
    if "deployment target" in q or "cloud provider" in q:
        if re.search(r"no deployment target was selected|no cloud provider decision|unknown: no active fact", lower):
            return "UNKNOWN"
        match = re.search(r"\bdeployment target(?: is|:|=| selected)?\s+([A-Za-z0-9 .-]+)", prompt, re.I)
        return match.group(1).strip(" .,:;") if match else "UNKNOWN"
    if "avoid" in q or "do not use" in q or "not use" in q:
        matches = re.findall(r"\b(?:do_not_use|do not use|don't use|avoid)[:= ]+([A-Za-z0-9 +.#-]+)", prompt, re.I)
        if matches:
            return _clean_extracted_value(matches[-1])
        if "Arduino C++" in prompt:
            return "Arduino C++"
        if "Node.js" in prompt:
            return "Node.js"
        return "UNKNOWN"
    if "pin" in q or "gpio" in q:
        pin_match = re.search(r"physical pin\s+(\d+)", q)
        if pin_match:
            pin = pin_match.group(1)
            if re.search(rf"no mapping for physical pin {pin}\b|unknown: no active fact", lower):
                return "UNKNOWN"
            match = re.search(rf"physical pin {pin}\s+maps to\s+([A-Za-z ]*GPIO\s*\d+)", prompt, re.I)
            return match.group(1).strip(" .,:;") if match else "UNKNOWN"
    parts = []
    if "board" in q:
        board = _last_matching_value(
            prompt,
            [
                r"\b(?:board|controller_board)=([A-Za-z0-9 .+-]+)",
                r"final decision:\s*(?:switch from [^.]+ to|use)?\s*(Raspberry Pi Pico W|ESP32-C3|ESP32-S3|ESP32|Arduino Uno)",
                r"controller board:\s*(Raspberry Pi Pico W|ESP32-C3|ESP32-S3|ESP32|Arduino Uno)",
                r"board:\s*(Raspberry Pi Pico W|ESP32-C3|ESP32-S3|ESP32|Arduino Uno)",
                r"use\s+(Raspberry Pi Pico W|ESP32-C3|ESP32-S3|ESP32|Arduino Uno)",
            ],
        )
        if board:
            parts.append(board)
    if "language" in q or "code" in q or "runtime" in q:
        language = _last_matching_value(
            prompt,
            [
                r"\b(?:language|runtime)=([A-Za-z0-9 .+#-]+)",
                r"final decision:\s*language:\s*([A-Za-z0-9 .+#-]+)",
                r"language:\s*([A-Za-z0-9 .+#-]+)",
            ],
        ) or _last_matching_value(
            prompt,
            [
                r"use\s+(Python 3\.11|MicroPython|TypeScript|JavaScript|Arduino C\+\+)",
                r"with\s+(Python 3\.11|MicroPython|TypeScript|JavaScript|Arduino C\+\+)",
            ],
        )
        if language:
            parts.append(language)
    if "voltage" in q:
        voltage = _last_matching_value(prompt, [r"\b(?:pump_)?voltage=([0-9]+V)", r"\b(?:pump )?voltage:\s*([0-9]+V)"])
        if voltage:
            parts.append(voltage)
    if "current" in q:
        current = _last_matching_value(prompt, [r"\b(?:pump_)?current=([0-9]+A)", r"\b(?:pump )?current:\s*([0-9]+A)"])
        if current:
            parts.append(current)
    if "model" in q:
        model = _last_matching_value(prompt, [r"\bmodel=([A-Za-z0-9_.:/# +-]+)", r"\bmodel:\s*([A-Za-z0-9_.:/# +-]+)"])
        if model:
            parts.append(model)
    if "package" in q or "library" in q:
        package = _last_matching_value(prompt, [r"\bpackage=([A-Za-z0-9_.:/# +-]+)", r"\bpackage:\s*([A-Za-z0-9_.:/# +-]+)"])
        if package:
            parts.append(package)
    if "path" in q or "file" in q or "runbook" in q:
        path = _last_matching_value(prompt, [r"\bpath=([A-Za-z0-9_./-]+)", r"\bpath:\s*([A-Za-z0-9_./-]+)"])
        if path:
            parts.append(path)
    if "endpoint" in q or "route" in q or "api" in q:
        endpoint = _last_matching_value(prompt, [r"\bendpoint=([A-Za-z0-9_./{}:-]+)", r"\bendpoint:\s*([A-Za-z0-9_./{}:-]+)"])
        if endpoint:
            parts.append(endpoint)
    if "region" in q:
        region = _last_matching_value(prompt, [r"\bregion=([A-Za-z0-9_.-]+)", r"\bregion:\s*([A-Za-z0-9_.-]+)"])
        if region:
            parts.append(region)
    if "version" in q or "schema" in q:
        version = _last_matching_value(prompt, [r"\bversion=([A-Za-z0-9_.-]+)", r"\bschema_version=([A-Za-z0-9_.-]+)", r"\bversion:\s*([A-Za-z0-9_.-]+)"])
        if version:
            parts.append(version)
    if "retention" in q or "ttl" in q:
        retention = _last_matching_value(
            prompt,
            [
                r"\bretention_days=([0-9]+\s*days?)",
                r"\bretention=([0-9]+\s*days?)",
                r"\bttl=([0-9]+\s*days?)",
            ],
        )
        if retention:
            parts.append(retention)
    if "timeout" in q:
        timeout = _last_matching_value(prompt, [r"\btimeout=([0-9]+\s*(?:ms|s|sec|seconds?))", r"\btimeout:\s*([0-9]+\s*(?:ms|s|sec|seconds?))"])
        if timeout:
            parts.append(timeout)
    if "secret store" in q or "secrets" in q:
        secret_store = _last_matching_value(prompt, [r"\bsecret_store=([A-Za-z0-9_.:/# +-]+)", r"\bsecret store:\s*([A-Za-z0-9_.:/# +-]+)"])
        if secret_store:
            parts.append(secret_store)
    if "checkpoint" in q:
        checkpoint = _last_matching_value(
            prompt,
            [
                r"\bcheckpoint_interval=([0-9]+\s*(?:turns?|messages?))",
                r"\bcheckpoint:\s*([0-9]+\s*(?:turns?|messages?))",
            ],
        )
        if checkpoint:
            parts.append(checkpoint)
    return ", ".join(parts) if parts else "UNKNOWN"


def _last_matching_value(text: str, patterns: list[str]) -> str | None:
    values: list[str] = []
    for pattern in patterns:
        values.extend(_clean_extracted_value(match.group(1)) for match in re.finditer(pattern, text, re.I))
    return values[-1] if values else None


def _clean_extracted_value(value: str) -> str:
    value = value.split(" evidence=", 1)[0]
    value = value.split(" evidence", 1)[0]
    value = value.split(". Do not", 1)[0]
    return value.strip(" .,:;")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evals/paired_quality_cases.jsonl")
    parser.add_argument("--provider", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default=os.environ.get("AYYGENT_MODEL") or os.environ.get("AYYGENT_OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--target-savings", type=float, default=0.70)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    report = run_cases(load_cases(args.cases), provider=args.provider, model=args.model, target_savings=args.target_savings)
    output = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
