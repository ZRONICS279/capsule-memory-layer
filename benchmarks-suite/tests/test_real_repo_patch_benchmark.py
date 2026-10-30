from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "real_repo_patch_benchmark.py"
spec = importlib.util.spec_from_file_location("real_repo_patch_benchmark", BENCHMARK)
real_repo_patch_benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["real_repo_patch_benchmark"] = real_repo_patch_benchmark
spec.loader.exec_module(real_repo_patch_benchmark)


class RealRepoPatchBenchmarkTests(unittest.TestCase):
    def test_example_case_file_validates(self):
        cases = real_repo_patch_benchmark.load_cases(ROOT / "evals" / "real_repo_cases.example.json")

        self.assertEqual(len(cases), 1)
        self.assertEqual(real_repo_patch_benchmark.validate_cases(cases), [])
        self.assertEqual(cases[0].expected_files, ("src/api/routes/export_jobs.py",))

    def test_validation_rejects_missing_memory(self):
        case = real_repo_patch_benchmark.RealRepoCase(
            case_id="bad",
            repo_url="https://github.com/example/project.git",
            base_ref="abc123",
            prompt="Fix it.",
            test_command="pytest",
            memory_sessions=(),
            expected_files=("src/file.py",),
        )

        errors = real_repo_patch_benchmark.validate_cases([case])

        self.assertTrue(any("memory_sessions" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
