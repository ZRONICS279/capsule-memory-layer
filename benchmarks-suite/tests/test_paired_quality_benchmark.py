from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "paired_quality_benchmark.py"
spec = importlib.util.spec_from_file_location("paired_quality_benchmark", BENCHMARK)
paired_quality_benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["paired_quality_benchmark"] = paired_quality_benchmark
spec.loader.exec_module(paired_quality_benchmark)


class PairedQualityBenchmarkTests(unittest.TestCase):
    def test_paired_quality_benchmark_passes_quality_cases(self):
        cases = paired_quality_benchmark.load_cases(ROOT / "evals" / "paired_quality_cases.jsonl")

        report = paired_quality_benchmark.run_cases(
            cases,
            provider="heuristic",
            model="heuristic",
            target_savings=0.70,
        )

        self.assertTrue(report["summary"]["passed"])
        self.assertGreaterEqual(report["summary"]["cases"], 10)
        self.assertGreaterEqual(report["summary"]["capsule_accuracy"], report["summary"]["baseline_accuracy"] - 0.03)
        self.assertLessEqual(
            report["summary"]["capsule_hallucination_count"],
            report["summary"]["baseline_hallucination_count"],
        )

    def test_grader_counts_expected_forbidden_and_unknown(self):
        case = paired_quality_benchmark.EvalCase(
            case_id="grade",
            capsule_id="grade",
            scope="test",
            context="",
            question="What color?",
            expected_facts=("UNKNOWN",),
            forbidden_claims=("blue",),
            unknown_expected=True,
        )

        good = paired_quality_benchmark.grade_answer("UNKNOWN", case)
        bad = paired_quality_benchmark.grade_answer("blue", case)

        self.assertTrue(good["unknown_passed"])
        self.assertEqual(good["hallucination_count"], 0)
        self.assertFalse(bad["unknown_passed"])
        self.assertEqual(bad["hallucination_count"], 1)

    def test_grader_accepts_checkpoint_interval_number_when_unit_is_contextual(self):
        case = paired_quality_benchmark.EvalCase(
            case_id="checkpoint",
            capsule_id="checkpoint",
            scope="agent_state",
            context="",
            question="What checkpoint interval should the agent use?",
            expected_facts=("12 turns",),
        )

        grade = paired_quality_benchmark.grade_answer("Use a checkpoint interval of 12.", case)

        self.assertEqual(grade["accuracy"], 1.0)
        self.assertEqual(grade["expected_missing"], [])


if __name__ == "__main__":
    unittest.main()
