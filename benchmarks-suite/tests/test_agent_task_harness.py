from pathlib import Path
import importlib.util
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "agent_task_harness.py"
ADAPTERS = ROOT / "benchmarks" / "agent_memory_adapters.py"
spec = importlib.util.spec_from_file_location("agent_task_harness", BENCHMARK)
agent_task_harness = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["agent_task_harness"] = agent_task_harness
spec.loader.exec_module(agent_task_harness)
adapter_spec = importlib.util.spec_from_file_location("agent_memory_adapters", ADAPTERS)
agent_memory_adapters = importlib.util.module_from_spec(adapter_spec)
assert adapter_spec.loader is not None
sys.modules["agent_memory_adapters"] = agent_memory_adapters
adapter_spec.loader.exec_module(agent_memory_adapters)


class AgentTaskHarnessTests(unittest.TestCase):
    def test_harness_runs_generated_repo_tasks_with_adapters(self):
        tasks = agent_task_harness.load_tasks(families="auth_migration,billing_retry", cases_per_family=1)

        report = agent_task_harness.run_harness(
            tasks,
            adapters=("no_memory", "bm25_rag", "capsule"),
            max_steps=2,
        )

        self.assertEqual(report["summary"]["tasks"], 2)
        self.assertEqual(report["summary"]["runs"], 6)
        self.assertIn("capsule", report["summary"]["adapter_scores"])
        self.assertGreaterEqual(report["summary"]["adapter_scores"]["capsule"]["success_rate"], 1.0)
        self.assertLess(report["summary"]["adapter_scores"]["no_memory"]["success_rate"], 1.0)

    def test_bm25_rag_retrieves_relevant_memory_event(self):
        adapter = agent_memory_adapters.make_adapter("bm25_rag")
        adapter.reset("bm25")
        adapter.ingest_event("Noise about unrelated generated files and old docs.")
        adapter.ingest_event(
            "Final config_key=PAYMENT_RETRY_LIMIT_0 edit_file=src/billing/retry_policy.py "
            "run_test=python3 -m unittest tests.test_retry_policy."
        )

        prepared = adapter.prepare_context("Apply the billing retry config key.", ROOT)

        self.assertIn("PAYMENT_RETRY_LIMIT_0", prepared.payload)
        self.assertNotIn("unrelated generated files", prepared.payload)

    def test_capsule_prompt_codec_keeps_facts_while_dropping_scaffolding(self):
        payload = "\n".join(
            [
                "MEMORY_HANDLE @case id=case v=7",
                "RULE active facts only; UNKNOWN if absent; CONFLICT if facts disagree.",
                "ACTIVE_FACTS:",
                "- project.edit_file=src/api/routes/export_jobs.py evidence=45c5edbb2bfbf625:project:edit_file",
                "- project.pagination_mode=cursor evidence=45c5edbb2bfbf625:project:pagination_mode",
                "- project.timeout_seconds=30 evidence=7f2457a650973cbf:project:timeout_seconds",
                "RESPONSE_RULE use exact active values; do not infer missing names/numbers/versions.",
            ]
        )

        compact = agent_memory_adapters.compact_capsule_prompt(payload)

        self.assertLess(agent_memory_adapters.estimate_tokens(compact), agent_memory_adapters.estimate_tokens(payload))
        self.assertIn("src/api/routes/export_jobs.py", compact)
        self.assertIn("cursor", compact)
        self.assertIn("30", compact)
        self.assertNotIn("MEMORY_HANDLE", compact)
        self.assertNotIn("45c5edbb2bfbf625:project:edit_file", compact)

    def test_vector_rag_alias_resolves_to_embedding_adapter(self):
        adapter = agent_memory_adapters.make_adapter("vector_rag")

        self.assertIsInstance(adapter, agent_memory_adapters.OpenAIEmbeddingRagAdapter)
        self.assertEqual(adapter.name, "vector_rag")

    def test_score_result_checks_forbidden_strings_in_repo_copy(self):
        task = agent_task_harness.AgentTask(
            task_id="forbidden",
            family="unit",
            prompt="fix stale value",
            memory_events=(),
            expected_files=("src/config.py",),
            test_command="python3 -m unittest tests.test_config",
            forbidden_strings=("STALE_VALUE",),
        )
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "src").mkdir()
            (repo / "src/config.py").write_text("VALUE = 'STALE_VALUE'\n", encoding="utf-8")

            result = agent_task_harness._score_result(
                task,
                "adapter",
                ("src/config.py",),
                True,
                "",
                1,
                0,
                0,
                0,
                10,
                5,
                0.5,
                0.0,
                0.0,
                "done",
                repo_path=repo,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.forbidden_found, ("STALE_VALUE",))

    def test_parse_tool_action_accepts_duplicated_json_object(self):
        action = agent_task_harness._parse_tool_action(
            '{"action":"read_file","path":"src/app.py"}\n'
            '{"action":"read_file","path":"src/app.py"}'
        )

        self.assertEqual(action, {"action": "read_file", "path": "src/app.py"})

    def test_repo_contains_flags_separator_variant_for_forbidden_fact(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "src").mkdir()
            (repo / "src/app.py").write_text("# normalize-legacy-invoice remains here\n", encoding="utf-8")

            found = agent_task_harness._repo_contains(repo, "normalize_legacy_invoice")

        self.assertTrue(found)


    def test_agent_memory_baselines_return_compact_relevant_context(self):
        adapter_names = ("mem0", "langgraph", "langgraph_checkpoint", "zep", "graphiti", "memgpt", "letta")
        for name in adapter_names:
            with self.subTest(adapter=name):
                adapter = agent_memory_adapters.make_adapter(name)
                adapter.reset(f"baseline-{name}")
                for idx in range(40):
                    adapter.ingest_event(
                        f"Noise event {idx}: generated_file=src/generated/module_{idx}.py "
                        f"owner=team-{idx} irrelevant_flag=NOISE_{idx} "
                        "This event describes unrelated implementation notes and stale docs."
                    )
                adapter.ingest_event("Legacy docs say route=/api/v1/admin-reports and owner=old-team.")
                adapter.ingest_event(
                    "Current migration says endpoint=/api/v3/reports/0/export "
                    "edit_file=src/api/routes/admin_reports.py owner=platform-team."
                )

                prepared = adapter.prepare_context("Update the admin reports endpoint route.", ROOT)

                self.assertGreater(prepared.token_estimate, 0)
                self.assertLess(prepared.token_estimate, prepared.raw_token_estimate)
                self.assertIn("/api/v3/reports/0/export", prepared.payload)


    def test_real_memory_adapters_have_strict_setup_behavior(self):
        langgraph = agent_memory_adapters.make_adapter("langgraph_real")
        langgraph.reset("real-langgraph")
        langgraph.ingest_event("Current endpoint=/api/v3/reports/0/export edit_file=src/api/routes/admin_reports.py")
        prepared = langgraph.prepare_context("Find the current reports endpoint", ROOT)
        self.assertIn("/api/v3/reports/0/export", prepared.payload)

        mem0 = agent_memory_adapters.make_adapter("mem0_real")
        if not getattr(mem0, "_memory", None):
            import os
            if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("AYYGENT_AI_API_KEY")):
                with self.assertRaisesRegex(RuntimeError, "mem0_real requires"):
                    mem0.reset("real-mem0")

        zep = agent_memory_adapters.make_adapter("zep_real")
        import os
        if not os.environ.get("ZEP_API_KEY"):
            with self.assertRaisesRegex(RuntimeError, "zep_real requires"):
                zep.reset("real-zep")

        letta = agent_memory_adapters.make_adapter("letta_real")
        with self.assertRaisesRegex(RuntimeError, "letta_real"):
            letta.reset("real-letta")

    def test_embedding_rag_adapter_requires_api_key(self):
        adapter = agent_memory_adapters.make_adapter("openai_embedding_rag")
        adapter.reset("embedding")
        adapter.ingest_event("Final endpoint=/api/v3/reports/0/export")

        if not adapter.api_key:
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                adapter.prepare_context("What endpoint should be used?", ROOT)

    def test_unknown_owner_requires_no_patch_and_unknown_message(self):
        tasks = agent_task_harness.load_tasks(families="unknown_owner", cases_per_family=1)

        report = agent_task_harness.run_harness(tasks, adapters=("capsule",), max_steps=2)

        result = report["results"][0]
        self.assertTrue(result["success"])
        self.assertTrue(result["unknown_passed"])
        self.assertEqual(result["changed_files"], [])


if __name__ == "__main__":
    unittest.main()
