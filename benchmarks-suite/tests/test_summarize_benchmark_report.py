from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "summarize_benchmark_report.py"
spec = importlib.util.spec_from_file_location("summarize_benchmark_report", BENCHMARK)
summarize_benchmark_report = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["summarize_benchmark_report"] = summarize_benchmark_report
spec.loader.exec_module(summarize_benchmark_report)


class SummarizeBenchmarkReportTests(unittest.TestCase):
    def test_markdown_summary_contains_table_and_claim(self):
        report = {
            "summary": {
                "adapters": ["mem0_real", "capsule"],
                "best_adapter": "capsule",
                "best_cross_session_score": 0.8,
                "paired_deltas": {
                    "right": "mem0_real",
                    "pairs": 2,
                    "avg_cross_session_score_delta": 0.5,
                    "bootstrap_ci": {"low": 0.25, "high": 0.75},
                },
                "adapter_scores": {
                    "mem0_real": {
                        "success_rate": 1.0,
                        "context_success_rate": 0.5,
                        "avg_recall": 0.75,
                        "avg_stale_penalty": 0.5,
                        "avg_cross_session_score": 0.3,
                        "avg_raw_context_tokens": 1000,
                        "avg_prepared_context_tokens": 500,
                        "avg_token_savings": 0.5,
                    },
                    "capsule": {
                        "success_rate": 1.0,
                        "context_success_rate": 1.0,
                        "avg_recall": 1.0,
                        "avg_stale_penalty": 0.0,
                        "avg_cross_session_score": 0.8,
                        "avg_raw_context_tokens": 1000,
                        "avg_prepared_context_tokens": 200,
                        "avg_token_savings": 0.8,
                    },
                },
            },
            "results": [
                {"task_id": "a", "family": "long_horizon_project_memory", "mode": "retrieval_only"},
                {"task_id": "a", "family": "long_horizon_project_memory", "mode": "retrieval_only"},
            ],
        }

        text = summarize_benchmark_report.summarize_report(report, style="paper")

        self.assertIn("| Adapter |", text)
        self.assertIn("Capsule beat mem0_real", text)
        self.assertIn("percentage points higher score", text)
        self.assertIn("Plain-English Comparison", text)
        self.assertIn("Capsule compared with every baseline", text)
        self.assertIn("Current-fact recall", text)
        self.assertIn("long_horizon_project_memory", text)


if __name__ == "__main__":
    unittest.main()
