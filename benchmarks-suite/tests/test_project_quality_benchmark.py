from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "project_quality_benchmark.py"
spec = importlib.util.spec_from_file_location("project_quality_benchmark", BENCHMARK)
project_quality_benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["project_quality_benchmark"] = project_quality_benchmark
spec.loader.exec_module(project_quality_benchmark)


class ProjectQualityBenchmarkTests(unittest.TestCase):
    def test_project_quality_benchmark_passes_task_cases(self):
        report = project_quality_benchmark.run_cases(
            project_quality_benchmark.load_cases(),
            provider="heuristic",
            model="heuristic",
            target_savings=0.70,
        )

        self.assertTrue(report["summary"]["passed"])
        self.assertGreaterEqual(report["summary"]["cases"], 20)
        self.assertGreaterEqual(report["summary"]["families"], 10)
        self.assertGreaterEqual(report["summary"]["avg_token_savings"], 0.70)
        self.assertGreaterEqual(
            report["summary"]["capsule_task_score"],
            report["summary"]["baseline_task_score"] - 0.03,
        )
        self.assertLessEqual(
            report["summary"]["capsule_hallucination_count"],
            report["summary"]["baseline_hallucination_count"],
        )
        self.assertIn("summary", report["summary"]["baseline_scores"])
        self.assertIn("file_score", report["summary"]["category_scores"])
        self.assertGreaterEqual(report["summary"]["min_family_capsule_task_score"], 0.85)

    def test_project_quality_case_generator_supports_scale_and_family_selection(self):
        cases = project_quality_benchmark.load_cases(
            families="auth_migration,billing_retry",
            cases_per_family=1,
            scales="small,medium",
        )

        self.assertEqual(len(cases), 4)
        self.assertEqual({case.family for case in cases}, {"auth_migration", "billing_retry"})
        self.assertEqual({case.scale for case in cases}, {"small", "medium"})

    def test_project_task_grader_scores_required_forbidden_and_unknown(self):
        case = project_quality_benchmark.ProjectCase(
            case_id="grade",
            capsule_id="grade",
            scope="project",
            events=(),
            task="Who owns it?",
            category_items={"file": ("src/app.py",)},
            forbidden_items=("LegacyOwner",),
            unknown_expected=False,
        )

        good = project_quality_benchmark.grade_task_answer("edit_file: src/app.py", case)
        bad = project_quality_benchmark.grade_task_answer("edit_file: src/app.py owner=LegacyOwner", case)

        self.assertEqual(good["task_score"], 1.0)
        self.assertEqual(good["hallucination_count"], 0)
        self.assertLess(bad["task_score"], good["task_score"])
        self.assertEqual(bad["hallucination_count"], 1)

        unknown_case = project_quality_benchmark.ProjectCase(
            case_id="unknown",
            capsule_id="unknown",
            scope="project",
            events=(),
            task="Who owns it?",
            category_items={},
            unknown_expected=True,
        )
        self.assertTrue(project_quality_benchmark.grade_task_answer("UNKNOWN", unknown_case)["unknown_passed"])
        self.assertFalse(project_quality_benchmark.grade_task_answer("platform team", unknown_case)["unknown_passed"])


if __name__ == "__main__":
    unittest.main()
