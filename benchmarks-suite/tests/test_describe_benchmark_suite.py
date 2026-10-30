from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "describe_benchmark_suite.py"
spec = importlib.util.spec_from_file_location("describe_benchmark_suite", BENCHMARK)
describe_benchmark_suite = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["describe_benchmark_suite"] = describe_benchmark_suite
spec.loader.exec_module(describe_benchmark_suite)


class DescribeBenchmarkSuiteTests(unittest.TestCase):
    def test_describes_task_requirements(self):
        tasks = describe_benchmark_suite.load_tasks(
            families="long_horizon_project_memory",
            cases_per_family=1,
        )

        text = describe_benchmark_suite.describe_tasks(tasks)

        self.assertIn("100+ event project memory", text)
        self.assertIn("Required current facts", text)
        self.assertIn("src/api/routes/export_jobs.py", text)
        self.assertIn("What The Scores Mean", text)


if __name__ == "__main__":
    unittest.main()
