from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "cross_session_benchmark.py"
spec = importlib.util.spec_from_file_location("cross_session_benchmark", BENCHMARK)
cross_session_benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["cross_session_benchmark"] = cross_session_benchmark
spec.loader.exec_module(cross_session_benchmark)


class CrossSessionBenchmarkTests(unittest.TestCase):
    def test_retrieval_only_repeats_and_multiplicative_scoring(self):
        tasks = cross_session_benchmark.load_tasks(
            families="auth_decision_evolution,unknown_owner_abstention",
            cases_per_family=1,
        )

        report = cross_session_benchmark.run_cross_session_harness(
            tasks,
            adapters=("no_memory", "bm25_rag", "capsule"),
            retrieval_only=True,
            repeats=2,
            randomize_adapter_order=True,
            score_formula="multiplicative",
        )

        self.assertEqual(len(report["results"]), 12)
        self.assertEqual(report["summary"]["score_formula"], "multiplicative")
        self.assertIsNone(report["summary"]["adapter_scores"]["capsule"]["success_rate"])
        self.assertIsNone(report["summary"]["capsule_beats_mem0_real"])
        self.assertGreater(report["summary"]["adapter_scores"]["capsule"]["avg_cross_session_score"], 0)
        self.assertIn("bootstrap_ci", report["summary"]["adapter_scores"]["capsule"])

    def test_added_task_families_load(self):
        tasks = cross_session_benchmark.load_tasks(
            families="stale_docs_current_code,dependency_version_pin,unknown_owner_abstention,long_horizon_project_memory",
            cases_per_family=1,
        )

        self.assertEqual(
            [task.family for task in tasks],
            [
                "stale_docs_current_code",
                "dependency_version_pin",
                "unknown_owner_abstention",
                "long_horizon_project_memory",
            ],
        )
        self.assertTrue(tasks[-2].unknown_expected)
        self.assertGreaterEqual(sum(len(session) for session in tasks[-1].sessions), 100)
        self.assertIn("src/ops/oncall.py", tasks[-1].expected_files)

    def test_cross_session_factory_exposes_extra_baselines(self):
        names = ("langgraph_state", "openai_embedding_rag", "vector_rag")

        for name in names:
            with self.subTest(adapter=name):
                adapter = cross_session_benchmark.make_cross_session_adapter(name)
                self.assertTrue(adapter.name)
                adapter.close()


if __name__ == "__main__":
    unittest.main()
