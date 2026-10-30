from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "create_repo_patch_memory_eval.py"
spec = importlib.util.spec_from_file_location("create_repo_patch_memory_eval", BENCHMARK)
create_repo_patch_memory_eval = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["create_repo_patch_memory_eval"] = create_repo_patch_memory_eval
spec.loader.exec_module(create_repo_patch_memory_eval)


class CreateRepoPatchMemoryEvalTests(unittest.TestCase):
    def test_intensive_serializer_requires_direct_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            create_repo_patch_memory_eval.create_broken_repo(
                root,
                profile="intensive",
                noise_files=0,
                noise_lines=0,
            )

            serializer = (root / "src/orbitops/serializer.py").read_text(encoding="utf-8")
            expected = create_repo_patch_memory_eval.expected_outcomes("intensive")

        self.assertIn("src/orbitops/serializer.py", expected["expected_files"])
        self.assertIn("'schema_version': 'v2'", serializer)
        self.assertIn("'format': 'csv'", serializer)
        self.assertIn("'rows'", serializer)


if __name__ == "__main__":
    unittest.main()
