from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "build_researchpaper_artifacts.py"
spec = importlib.util.spec_from_file_location("build_researchpaper_artifacts", SCRIPT)
build_researchpaper_artifacts = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["build_researchpaper_artifacts"] = build_researchpaper_artifacts
spec.loader.exec_module(build_researchpaper_artifacts)


class BuildResearchpaperArtifactsTests(unittest.TestCase):
    def test_report_manifest_contains_expected_outputs(self):
        names = [item[1] for item in build_researchpaper_artifacts.REPORTS]

        self.assertIn("reports-long-horizon-retrieval.json", names)
        self.assertIn("reports-all-agentic.json", names)
        self.assertIn("reports-persistent-retrieval.json", names)


if __name__ == "__main__":
    unittest.main()
