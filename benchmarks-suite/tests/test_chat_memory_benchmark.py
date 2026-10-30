from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "chat_memory_benchmark.py"
spec = importlib.util.spec_from_file_location("chat_memory_benchmark", BENCHMARK)
chat_memory_benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["chat_memory_benchmark"] = chat_memory_benchmark
spec.loader.exec_module(chat_memory_benchmark)


class ChatMemoryBenchmarkTests(unittest.TestCase):
    def test_all_eight_chat_memory_families_load(self):
        tasks = chat_memory_benchmark.load_tasks(families="all", cases_per_family=1)

        self.assertEqual(len(tasks), 8)
        self.assertEqual({task.family for task in tasks}, set(chat_memory_benchmark.FAMILIES))

    def test_local_chat_memory_harness_runs(self):
        tasks = chat_memory_benchmark.load_tasks(
            families="personal_preference,privacy_boundary,deletion_forgetting",
            cases_per_family=1,
        )

        report = chat_memory_benchmark.run_chat_memory_harness(
            tasks,
            adapters=("capsule", "bm25_rag", "no_memory"),
        )

        self.assertEqual(len(report["results"]), 9)
        self.assertEqual(report["summary"]["benchmark"], "chat_memory")
        self.assertIn("avg_privacy_penalty", report["summary"]["adapter_scores"]["capsule"])
        self.assertIn("avg_deletion_penalty", report["summary"]["adapter_scores"]["capsule"])

    def test_context_grading_accepts_harmless_separator_variants(self):
        task = chat_memory_benchmark.ChatMemoryTask(
            task_id="separator",
            family="unit",
            prompt="continue",
            sessions=(),
            required_in_context=("memory loss", "medium-detail", "RequireSessionToken"),
            stale_in_context=("fire magic",),
            privacy_forbidden=("medical anxiety",),
            deletion_forbidden=("555-1234",),
        )

        score = chat_memory_benchmark._eval_context(
            task,
            "FACTS: memory-loss; medium detail; require session token; fire-magic; medical_anxiety; 555 1234",
            raw_tokens=100,
        )

        self.assertEqual(score.required_found, 3)
        self.assertEqual(score.stale_found, 1)
        self.assertEqual(score.privacy_found, 1)
        self.assertEqual(score.deletion_found, 1)


if __name__ == "__main__":
    unittest.main()
