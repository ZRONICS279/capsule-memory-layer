from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json

from semantic_capsule_memory import (
    AgentMemoryMiddleware,
    AgenticMemoryLayer,
    Capsule,
    CapsuleManager,
    CapsuleMemoryClient,
    CompactChatAudit,
    GroundedMemoryEngine,
    MemoryCompiler,
    MemoryWrite,
    SoftEmbeddingRuntime,
    TextSpineRuntime,
    audit_compact_chat,
    compact_memory_prompt,
    encode_prompt,
)
from semantic_capsule_memory.eval import load_cases, run_symbolic_eval
from semantic_capsule_memory.mcp_server import MemoryMCPServer
from semantic_capsule_memory.savings import estimate_input_savings


class CapsuleTests(unittest.TestCase):
    def test_capsule_extracts_exact_spine(self):
        capsule = Capsule("greenhouse", "@greenhouse")
        capsule = capsule.update(
            "For the greenhouse project, use ESP32-C3 with MicroPython. Pin 12 = BCM GPIO 18. Do not use Arduino C++.",
            scope="greenhouse",
        )

        self.assertEqual(capsule.query_fact("board", scope="greenhouse").answer, "ESP32-C3")
        self.assertEqual(capsule.query_fact("language", scope="greenhouse").answer, "MicroPython")
        self.assertEqual(capsule.query_fact("physical_pin_12", scope="greenhouse").answer, "BCM GPIO 18")
        self.assertEqual(capsule.query_fact("missing", scope="greenhouse").status, "unknown")

    def test_update_supersedes_active_choice(self):
        capsule = Capsule("project", "@project")
        capsule = capsule.update("Use ESP32-C3 with MicroPython.", scope="project")
        capsule = capsule.update("Actually switch from ESP32-C3 to Raspberry Pi Pico W.", scope="project")

        self.assertEqual(capsule.query_fact("board", scope="project").answer, "Raspberry Pi Pico W")
        superseded = [fact for fact in capsule.facts if fact.status == "superseded"]
        self.assertTrue(any(fact.value == "ESP32-C3" for fact in superseded))

    def test_rendered_context_is_compact_and_precise(self):
        raw = "Use ESP32-C3 with MicroPython. " * 200
        capsule = Capsule("greenhouse", "@greenhouse").update(raw, scope="greenhouse")

        context = capsule.render_context()

        self.assertLess(len(context.split()), len(raw.split()) / 5)
        self.assertIn("ESP32-C3", context)
        self.assertIn("RULE: use exact spine", context)

    def test_manager_persists_and_routes(self):
        with self.subTest("persist"):
            with TemporaryDirectory() as directory:
                store = Path(directory) / "store.json"
                manager = CapsuleManager(store)
                manager.create("greenhouse", "@greenhouse")
                manager.update("greenhouse", "Greenhouse uses ESP32-C3 and MicroPython.", scope="greenhouse")

                loaded = CapsuleManager(store)

                self.assertEqual(
                    loaded.get("greenhouse").query_fact("board", scope="greenhouse").answer,
                    "ESP32-C3",
                )
                self.assertEqual(loaded.similar("MicroPython greenhouse controller")[0][0], "greenhouse")

    def test_sqlite_store_persists(self):
        with TemporaryDirectory() as directory:
            store = Path(directory) / "store.sqlite"
            manager = CapsuleManager(store)
            manager.create("greenhouse", "@greenhouse")
            manager.update("greenhouse", "Use ESP32-C3 with MicroPython.", scope="greenhouse")

            loaded = CapsuleManager(store)

            self.assertEqual(
                loaded.get("greenhouse").query_fact("board", scope="greenhouse").answer,
                "ESP32-C3",
            )

    def test_negated_board_does_not_override_correction(self):
        capsule = Capsule("greenhouse", "@greenhouse")
        capsule = capsule.update(
            "First draft used ESP32. Correction: use ESP32-C3, not regular ESP32.",
            scope="greenhouse",
        )

        self.assertEqual(capsule.query_fact("board", scope="greenhouse").answer, "ESP32-C3")

    def test_sdk_prepares_text_memory(self):
        with TemporaryDirectory() as directory:
            client = CapsuleMemoryClient(Path(directory) / "store.json")
            client.remember(
                MemoryWrite(
                    capsule_id="greenhouse",
                    scope="greenhouse",
                    text="Use ESP32-C3 with MicroPython.",
                )
            )

            prepared = client.prepare("greenhouse", original_context="Use ESP32-C3 with MicroPython. " * 100)

            self.assertGreater(prepared.compression_ratio, 1.0)
            self.assertIn("ESP32-C3", prepared.text_context)

    def test_soft_runtime_reports_true_capsule_budget(self):
        capsule = Capsule("greenhouse", "@greenhouse").update("Use ESP32-C3.", scope="greenhouse")
        prepared = SoftEmbeddingRuntime(soft_token_count=32).prepare_memory(
            capsule,
            original_token_estimate=32000,
        )

        self.assertEqual(prepared.soft_token_count, 32)
        self.assertEqual(prepared.compression_ratio, 1000.0)

    def test_eval_harness_reports_accuracy_and_compression(self):
        cases = load_cases("evals/symbolic_cases.jsonl")
        results = run_symbolic_eval(cases)

        self.assertTrue(results)
        self.assertGreaterEqual(sum(result.correct for result in results), 6)

    def test_eval_harness_shows_compression_on_long_context(self):
        from semantic_capsule_memory.eval import EvalCase, EvalQuestion

        long_context = (
            "Daily build log: tests ran, docs changed, minor discussion happened. " * 400
            + "Final decision: use ESP32-C3 with MicroPython. Pump voltage: 12V."
        )
        cases = [
            EvalCase(
                capsule_id="long_greenhouse",
                context=long_context,
                scope="greenhouse",
                questions=(EvalQuestion(key="board", expected="ESP32-C3", scope="greenhouse"),),
            )
        ]
        result = run_symbolic_eval(cases)[0]

        self.assertGreater(result.compression_ratio, 10.0)

    def test_agent_middleware_injects_text_memory(self):
        with TemporaryDirectory() as directory:
            client = CapsuleMemoryClient(Path(directory) / "store.sqlite")
            client.remember(MemoryWrite("greenhouse", "Use ESP32-C3 with MicroPython.", "greenhouse"))
            middleware = AgentMemoryMiddleware(client)

            messages, metadata = middleware.prepare_chat(
                "greenhouse",
                [{"role": "user", "content": "What board?"}],
                original_context="Use ESP32-C3 with MicroPython. " * 100,
            )

            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("ESP32-C3", messages[0]["content"])
            self.assertEqual(metadata["memory_mode"], "text_spine")

    def test_agent_middleware_prepares_compact_chat_without_full_history(self):
        with TemporaryDirectory() as directory:
            client = CapsuleMemoryClient(Path(directory) / "store.sqlite")
            middleware = AgentMemoryMiddleware(client)
            full_history = (
                "Old verbose conversation and tool output. " * 200
                + "Final decision: use ESP32-C3 with MicroPython."
            )

            messages, metadata = middleware.prepare_compact_chat(
                "greenhouse",
                {"role": "user", "content": "What board should I use?"},
                system_messages=[{"role": "system", "content": "Answer from memory."}],
                transcript_to_remember=full_history,
                scope="greenhouse",
            )

            combined = "\n".join(message["content"] for message in messages)
            self.assertLess(len(combined), len(full_history))
            self.assertIn("ESP32-C3", messages[0]["content"])
            self.assertEqual(messages[-1]["content"], "What board should I use?")
            self.assertGreater(metadata["compression_ratio"], 1.0)
            self.assertFalse(metadata["compact_audit"]["full_transcript_leaked"])
            self.assertLess(metadata["compact_audit"]["outbound_tokens"], metadata["compact_audit"]["remembered_tokens"])

    def test_compact_chat_rejects_outbound_token_over_budget(self):
        with TemporaryDirectory() as directory:
            client = CapsuleMemoryClient(Path(directory) / "store.sqlite")
            middleware = AgentMemoryMiddleware(client)

            with self.assertRaises(ValueError):
                middleware.prepare_compact_chat(
                    "greenhouse",
                    {"role": "user", "content": "What board should I use?"},
                    transcript_to_remember="Final decision: use ESP32-C3 with MicroPython.",
                    scope="greenhouse",
                    max_outbound_tokens=1,
                )

    def test_compact_chat_audit_detects_full_transcript_leak(self):
        full_history = "Full transcript that should not be sent."
        audit = audit_compact_chat(
            [{"role": "user", "content": full_history}],
            remembered_context=full_history,
            estimated_memory_tokens=10,
        )

        self.assertIsInstance(audit, CompactChatAudit)
        self.assertTrue(audit.full_transcript_leaked)
        self.assertEqual(audit.tokens_saved, 0)

    def test_memory_compiler_selects_relevant_facts_under_budget(self):
        capsule = Capsule("greenhouse", "@greenhouse")
        capsule = capsule.update(
            "Use ESP32-C3 with MicroPython. Pin 12 = BCM GPIO 18. Pump voltage: 12V. Pump current: 2A. Do not use Arduino C++.",
            scope="greenhouse",
        )

        compiled = MemoryCompiler(token_budget=96).compile(
            capsule,
            query="Which pump voltage should I use?",
            original_context=("daily logs " * 1000),
        )

        self.assertIn("MEMORY_HANDLE @greenhouse", compiled.payload)
        self.assertIn("pump_voltage=12V", compiled.payload)
        self.assertGreater(compiled.compression_ratio, 10.0)
        self.assertTrue(compiled.evidence)

    def test_compiled_memory_surfaces_facts_the_query_does_not_name(self):
        """A stored fact has to come back without being asked for by name.

        Ranking by query overlap is right; dropping everything that does not
        overlap meant the caller had to know the answer to be told it.
        """

        capsule = Capsule("app", "@app")
        capsule = capsule.update("For this project, use biome.", scope="project")

        compiled = MemoryCompiler().compile(capsule, query="set up the linter")

        self.assertIn("preference=biome", compiled.payload)

    def test_project_query_keeps_the_preferences_it_needs(self):
        """A coding query used to drop every fact without a priority score,
        which is exactly when "prefer vitest" matters."""

        capsule = Capsule("app", "@app")
        capsule = capsule.update(
            "Prefer vitest for tests. timeout_seconds: 30",
            scope="project",
        )

        compiled = MemoryCompiler().compile(capsule, query="add a test for the panel")

        self.assertIn("vitest", compiled.payload)
        self.assertIn("timeout_seconds=30", compiled.payload)

    def test_the_budget_is_what_limits_the_spine(self):
        """Facts are left out because they do not fit, not because the query
        failed to mention them, and the relevant one is what survives."""

        capsule = Capsule("app", "@app")
        capsule = capsule.update(
            "pagination_mode: cursor. timeout_seconds: 30. schema_version: 4. on_call_owner: @carol",
            scope="project",
        )

        generous = MemoryCompiler(token_budget=512).compile(capsule, query="ship it")
        tight = MemoryCompiler(token_budget=48).compile(capsule, query="what is the timeout")

        self.assertGreater(len(generous.evidence), len(tight.evidence))
        self.assertIn("timeout_seconds=30", tight.payload)

    def test_compact_prompt_codec_preserves_fact_values(self):
        verbose = "\n".join(
            [
                "MEMORY_HANDLE @case id=case v=7",
                "RULE active facts only; UNKNOWN if absent; CONFLICT if facts disagree.",
                "ACTIVE_FACTS:",
                "- project.edit_file=src/api/routes/export_jobs.py evidence=45c5edbb2bfbf625:project:edit_file",
                "- project.pagination_mode=cursor evidence=45c5edbb2bfbf625:project:pagination_mode",
                "- project.timeout_seconds=30 evidence=7f2457a650973cbf:project:timeout_seconds",
                "- project.edit_file=pyproject.toml evidence=7f2457a650973cbf:project:edit_file",
                "RESPONSE_RULE use exact active values; do not infer missing names/numbers/versions.",
            ]
        )

        compact = compact_memory_prompt(verbose)
        encoded = encode_prompt(verbose, codec="compact")

        self.assertLess(len(compact), len(verbose))
        self.assertIn("src/api/routes/export_jobs.py", compact)
        self.assertIn("pyproject.toml", compact)
        self.assertIn("cursor", compact)
        self.assertIn("30", compact)
        self.assertIn("#e1", compact)
        self.assertNotIn("MEMORY_HANDLE", compact)
        self.assertNotIn("45c5edbb2bfbf625:project:edit_file", compact)
        self.assertEqual(encoded.evidence_refs["#e1"], "45c5edbb2bfbf625")

    def test_compact_prompt_codec_groups_facts_by_explicit_target_file(self):
        verbose = "\n".join(
            [
                "MEMORY_HANDLE @case id=case v=7",
                "RULE active facts only; UNKNOWN if absent; CONFLICT if facts disagree.",
                "ACTIVE_FACTS:",
                "- project.edit_file=src/api/routes/export_jobs.py evidence=45c5edbb2bfbf625:project:edit_file",
                "- project.pagination_mode=cursor evidence=45c5edbb2bfbf625:project:pagination_mode",
                "- project.timeout_seconds=30 evidence=45c5edbb2bfbf625:project:timeout_seconds",
                "RESPONSE_RULE use exact active values; do not infer missing names/numbers/versions.",
            ]
        )

        compact = compact_memory_prompt(verbose)

        self.assertIn("src/api/routes/export_jobs.py: PAGINATION_MODE=cursor; TIMEOUT_SECONDS=30 #e1", compact)
        self.assertNotIn("project.edit_file=src/api/routes/export_jobs.py", compact)
        self.assertNotIn("project.pagination_mode=cursor", compact)
        self.assertIn("cursor", compact)
        self.assertIn("30", compact)

    def test_compact_prompt_codec_does_not_group_ambiguous_file_sources(self):
        verbose = "\n".join(
            [
                "MEMORY_HANDLE @case id=case v=7",
                "RULE active facts only; UNKNOWN if absent; CONFLICT if facts disagree.",
                "ACTIVE_FACTS:",
                "- project.edit_file=src/api/routes/export_jobs.py evidence=45c5edbb2bfbf625:project:edit_file",
                "- project.edit_file=src/ops/oncall.py evidence=45c5edbb2bfbf625:project:edit_file",
                "- project.on_call_owner=@carol-wu evidence=45c5edbb2bfbf625:project:on_call_owner",
                "RESPONSE_RULE use exact active values; do not infer missing names/numbers/versions.",
            ]
        )

        compact = compact_memory_prompt(verbose)

        self.assertNotIn("src/api/routes/export_jobs.py:", compact)
        self.assertNotIn("src/ops/oncall.py:", compact)
        self.assertIn("project.on_call_owner=@carol-wu", compact)
        self.assertIn("project.edit_file=src/api/routes/export_jobs.py", compact)
        self.assertIn("project.edit_file=src/ops/oncall.py", compact)

    def test_compact_prompt_codec_groups_single_target_file_across_sources(self):
        verbose = "\n".join(
            [
                "MEMORY_HANDLE @case id=case v=7",
                "RULE active facts only; UNKNOWN if absent; CONFLICT if facts disagree.",
                "ACTIVE_FACTS:",
                "- project.pagination_mode=cursor evidence=1111111111111111:project:pagination_mode",
                "- project.config_key=TIMEOUT_SECONDS evidence=2222222222222222:project:config_key",
                "- project.timeout_seconds=30 evidence=3333333333333333:project:timeout_seconds",
                "- project.edit_file=src/api/routes/export_jobs.py evidence=4444444444444444:project:edit_file",
                "RESPONSE_RULE use exact active values; do not infer missing names/numbers/versions.",
            ]
        )

        encoded = encode_prompt(verbose, codec="compact")

        self.assertIn("src/api/routes/export_jobs.py:", encoded.payload)
        self.assertIn("PAGINATION_MODE=cursor", encoded.payload)
        self.assertIn("config_key=TIMEOUT_SECONDS", encoded.payload)
        self.assertIn("TIMEOUT_SECONDS=30", encoded.payload)
        self.assertNotIn("project.edit_file=src/api/routes/export_jobs.py", encoded.payload)
        self.assertEqual(encoded.evidence_refs["#e1"], "1111111111111111")
        self.assertEqual(encoded.evidence_refs["#e2"], "2222222222222222")
        self.assertEqual(encoded.evidence_refs["#e3"], "3333333333333333")

    def test_fact_relationships_capture_file_and_replacement_edges(self):
        capsule = Capsule("repo", "@repo")
        capsule = capsule.update(
            "REVISED AUTH DECISION: Replace LegacyApiKeyAuth with RequireSessionToken "
            "in src/orbitops/auth.py.",
            scope="project",
        )

        active = capsule.active_facts()
        relation_pairs = {
            (relation.kind, relation.target)
            for fact in active
            for relation in fact.relations
        }

        self.assertIn(("belongs_to_file", "src/orbitops/auth.py"), relation_pairs)
        self.assertIn(("replaces", "LegacyApiKeyAuth"), relation_pairs)
        self.assertIn(("supersedes", "LegacyApiKeyAuth"), relation_pairs)
        self.assertIn(("decision_for_task", "auth"), relation_pairs)

    def test_compiler_renders_relation_file_hint_for_safe_grouping(self):
        capsule = Capsule("repo", "@repo")
        capsule = capsule.update(
            "FINAL EXPORT CONFIG: PAGINATION_MODE='cursor' and TIMEOUT_SECONDS=30 "
            "in src/orbitops/export_jobs.py.",
            scope="project",
        )

        compiled = MemoryCompiler(token_budget=220).compile(
            capsule,
            query="Apply export job config",
            original_context="x " * 500,
        )
        encoded = encode_prompt(compiled.payload, codec="compact")

        self.assertIn("src/orbitops/export_jobs.py:", encoded.payload)
        self.assertIn("PAGINATION_MODE=cursor", encoded.payload)
        self.assertIn("TIMEOUT_SECONDS=30", encoded.payload)
        self.assertNotIn("file=src/orbitops/export_jobs.py", encoded.payload)

    def test_multi_file_events_do_not_create_ambiguous_file_relationships(self):
        capsule = Capsule("repo", "@repo")
        capsule = capsule.update(
            "FINAL OWNER DECISION: set PAYMENT_PLATFORM_OWNER='@carol-wu' while "
            "editing src/orbitops/oncall.py and src/orbitops/routes/admin_reports.py.",
            scope="project",
        )

        owner_facts = [fact for fact in capsule.active_facts() if fact.key == "payment_platform_owner"]
        self.assertTrue(owner_facts)
        owner_paths = {
            relation.target
            for fact in owner_facts
            for relation in fact.relations
            if relation.kind == "belongs_to_file"
        }
        self.assertEqual(owner_paths, set())

    def test_text_spine_runtime_can_emit_compact_prompt_codec(self):
        capsule = Capsule("deps", "@deps")
        capsule = capsule.update(
            "FINAL DEPENDENCY PIN: package=pydantic version=2.7.0 edit_file=pyproject.toml",
            scope="project",
        )

        verbose = TextSpineRuntime(token_budget=160, prompt_codec="verbose").prepare_memory(
            capsule,
            original_token_estimate=1200,
            query="Apply pydantic dependency pin",
        )
        compact = TextSpineRuntime(token_budget=160, prompt_codec="compact").prepare_memory(
            capsule,
            original_token_estimate=1200,
            query="Apply pydantic dependency pin",
        )

        self.assertLess(compact.estimated_text_tokens, verbose.estimated_text_tokens)
        self.assertIn("pydantic", compact.text_context)
        self.assertIn("2.7.0", compact.text_context)
        self.assertIn("pyproject.toml", compact.text_context)
        self.assertTrue(compact.evidence_refs)
        self.assertTrue(next(iter(compact.evidence_refs)).startswith("#e"))

    def test_text_spine_runtime_defaults_to_compact_prompt_codec(self):
        capsule = Capsule("deps", "@deps")
        capsule = capsule.update(
            "FINAL DEPENDENCY PIN: package=pydantic version=2.7.0 edit_file=pyproject.toml",
            scope="project",
        )

        prepared = TextSpineRuntime(token_budget=160).prepare_memory(
            capsule,
            original_token_estimate=1200,
            query="Apply pydantic dependency pin",
        )

        self.assertIn("MEM @deps", prepared.text_context)
        self.assertIn("pydantic", prepared.text_context)
        self.assertTrue(prepared.evidence_refs)
        self.assertNotIn("MEMORY_HANDLE", prepared.text_context)

    def test_memory_compiler_adds_generic_voltage_alias_for_voltage_queries(self):
        capsule = Capsule("lighting", "@lighting")
        capsule = capsule.update(
            "Lighting controller notes. Final decision: pump voltage: 24V. Controller board: ESP32-S3.",
            scope="lighting",
        )

        compiled = MemoryCompiler(token_budget=160).compile(
            capsule,
            query="What voltage is the final lighting controller using?",
            original_context=("lighting logs " * 1000),
        )

        self.assertIn("lighting.pump_voltage=24V", compiled.payload)
        self.assertIn("alias=lighting.voltage=24V", compiled.payload)

    def test_sdk_expands_exact_evidence_without_prompting_full_ledger(self):
        with TemporaryDirectory() as directory:
            client = CapsuleMemoryClient(Path(directory) / "store.sqlite")
            capsule = client.remember(
                MemoryWrite(
                    "greenhouse",
                    "Use ESP32-C3 with MicroPython. Pump voltage: 12V.",
                    "greenhouse",
                )
            )
            compiled = client.compile("greenhouse", query="pump voltage", original_context="x " * 1000)
            source = compiled.evidence[0].source

            expanded = client.expand_evidence(capsule.capsule_id, source)

            self.assertIn("Pump voltage: 12V", expanded)

    def test_grounded_engine_answers_from_compiled_memory(self):
        with TemporaryDirectory() as directory:
            engine = GroundedMemoryEngine(Path(directory) / "memory.sqlite", token_budget=160)
            raw = (
                "Daily logs and unrelated discussion. " * 500
                + "Final decision: use ESP32-C3 with MicroPython. Pump voltage: 12V. Do not use Arduino C++."
            )
            engine.ingest("greenhouse", raw, scope="greenhouse", label="@greenhouse")

            answer = engine.answer("greenhouse", "What pump voltage should I use?", original_context=raw)

            self.assertEqual(answer.status, "answered")
            self.assertEqual(answer.answer, "12V")
            self.assertGreater(answer.compiled.compression_ratio, 15.0)
            self.assertTrue(answer.evidence_sources)

    def test_grounded_engine_returns_unknown_instead_of_guessing(self):
        with TemporaryDirectory() as directory:
            engine = GroundedMemoryEngine(Path(directory) / "memory.sqlite")
            engine.ingest("greenhouse", "Use ESP32-C3 with MicroPython.", scope="greenhouse")

            answer = engine.answer("greenhouse", "What enclosure color did we pick?")

            self.assertEqual(answer.status, "unknown")
            self.assertEqual(answer.answer, "UNKNOWN")

    def test_engine_retrieves_raw_evidence_for_uncompiled_context(self):
        with TemporaryDirectory() as directory:
            engine = GroundedMemoryEngine(Path(directory) / "memory.sqlite")
            engine.ingest(
                "greenhouse",
                "Long meeting notes. The enclosure color discussion ended with blue anodized aluminum. Nothing else changed.",
                scope="greenhouse",
            )

            snippets = engine.retrieve("greenhouse", "enclosure color")

            self.assertTrue(snippets)
            self.assertIn("blue anodized aluminum", snippets[0].text)

    def test_engine_answers_from_raw_chunk_when_fact_was_not_extracted(self):
        with TemporaryDirectory() as directory:
            engine = GroundedMemoryEngine(Path(directory) / "memory.sqlite")
            engine.ingest(
                "greenhouse",
                "Meeting notes and side discussion. The enclosure finish discussion ended with blue anodized aluminum.",
                scope="greenhouse",
            )

            answer = engine.answer("greenhouse", "What enclosure color did we pick?")

            self.assertEqual(answer.status, "answered_from_evidence")
            self.assertEqual(answer.answer, "blue anodized aluminum")
            self.assertTrue(answer.retrieved)

    def test_ingest_chunks_large_context_and_evidence_is_small(self):
        with TemporaryDirectory() as directory:
            engine = GroundedMemoryEngine(Path(directory) / "memory.sqlite")
            raw = "\n".join([f"Log line {index}: nothing important." for index in range(400)])
            raw += "\nFinal decision: Pump voltage: 12V."
            capsule = engine.ingest("greenhouse", raw, scope="greenhouse")

            answer = engine.answer("greenhouse", "What pump voltage should I use?", original_context=raw)
            evidence = engine.expand("greenhouse", answer.evidence_sources[0])

            self.assertGreater(len(capsule.events), 1)
            self.assertLess(len(evidence), len(raw) / 2)
            self.assertEqual(answer.answer, "12V")

    def test_agentic_memory_layer_hits_sixty_percent_reduction_target(self):
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(Path(directory) / "memory.sqlite", default_token_budget=160, target_reduction=0.60)
            raw = (
                "Agent turn with non-durable tool chatter and repeated status. " * 900
                + "Final decision: use ESP32-C3. Pump voltage: 12V. Do not use Arduino C++."
            )
            memory.remember("greenhouse", raw, scope="greenhouse", label="@greenhouse")

            packet = memory.prepare_step("greenhouse", "What pump voltage should I use?", original_context=raw)
            answer = memory.answer("greenhouse", "What pump voltage should I use?", original_context=raw)

            self.assertTrue(packet.passed_target)
            self.assertGreaterEqual(packet.reduction, 0.60)
            self.assertEqual(answer.answer, "12V")

    def test_agentic_memory_layer_uses_ledger_tokens_without_original_context(self):
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(Path(directory) / "memory.sqlite", default_token_budget=160, target_reduction=0.75)
            raw = (
                "Agent turn with non-durable tool chatter and repeated status. " * 500
                + "Final decision: use ESP32-C3. Pump voltage: 12V."
            )
            memory.remember("greenhouse", raw, scope="greenhouse", label="@greenhouse")

            packet = memory.prepare_step("greenhouse", "What pump voltage should I use?")

            self.assertGreater(packet.raw_tokens, packet.memory_tokens)
            self.assertGreaterEqual(packet.reduction, 0.75)

    def test_agentic_memory_layer_can_emit_compact_prompt_codec(self):
        raw = (
            "Historical dependency chatter about old pydantic pins. " * 120
            + "FINAL DEPENDENCY PIN: package=pydantic version=2.7.0 edit_file=pyproject.toml"
        )
        with TemporaryDirectory() as verbose_dir, TemporaryDirectory() as compact_dir:
            verbose_memory = AgenticMemoryLayer(
                Path(verbose_dir) / "memory.sqlite",
                default_token_budget=180,
                target_reduction=0.60,
                lean_threshold=0,
                prompt_codec="verbose",
            )
            compact_memory = AgenticMemoryLayer(
                Path(compact_dir) / "memory.sqlite",
                default_token_budget=180,
                target_reduction=0.60,
                lean_threshold=0,
                prompt_codec="compact",
            )
            for memory in (verbose_memory, compact_memory):
                memory.remember("export", raw, scope="project", label="@export")

            verbose_packet = verbose_memory.prepare_step("export", "Apply pydantic dependency pin", original_context=raw)
            compact_packet = compact_memory.prepare_step("export", "Apply pydantic dependency pin", original_context=raw)

            self.assertLess(compact_packet.memory_tokens, verbose_packet.memory_tokens)
            self.assertIn("pydantic", compact_packet.payload)
            self.assertIn("2.7.0", compact_packet.payload)
            self.assertIn("pyproject.toml", compact_packet.payload)
            self.assertTrue(compact_packet.evidence_refs)
            self.assertIn("#e1", compact_packet.payload)
            self.assertNotIn("MEMORY_HANDLE", compact_packet.payload)

    def test_agentic_memory_layer_defaults_to_compact_coding_packet(self):
        raw = (
            "Historical dependency chatter about old pydantic pins. " * 120
            + "FINAL DEPENDENCY PIN: package=pydantic version=2.7.0 edit_file=pyproject.toml"
        )
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(Path(directory) / "memory.sqlite")
            memory.remember("export", raw, scope="project", label="@export")

            packet = memory.prepare_step("export", "Apply pydantic dependency pin", original_context=raw)

            self.assertIn("MEM @export", packet.payload)
            self.assertIn("pydantic", packet.payload)
            self.assertIn("pyproject.toml", packet.payload)
            self.assertTrue(packet.evidence_refs)
            self.assertNotIn("MEMORY_HANDLE", packet.payload)

    def test_agentic_memory_layer_autoscales_broad_repo_patch_tasks(self):
        events = (
            "FINAL AUTH DECISION: define RequireSessionToken in src/orbitops/auth.py.",
            "FINAL AUTH DECISION: src/orbitops/routes/admin_reports.py must import RequireSessionToken and set AUTH_MIDDLEWARE = RequireSessionToken.",
            "REJECTED AUTH DECISION: do not leave LegacyApiKeyAuth anywhere in active code.",
            "FINAL EXPORT CONFIG: PAGINATION_MODE='cursor' and TIMEOUT_SECONDS=30 in src/orbitops/export_jobs.py.",
            "REJECTED EXPORT CONFIG: PAGINATION_MODE='offset' and TIMEOUT_SECONDS=5 are stale.",
            "FINAL INVOICE DECISION: expose normalize_active_invoice in src/orbitops/invoices.py.",
            "FINAL CHECKOUT DECISION: Safari fix shipped. Set CHECKOUT_V2='enabled' in src/orbitops/flags.py.",
            "FINAL OWNER DECISION: payments-platform owner is @carol-wu. Update src/orbitops/oncall.py so PAYMENT_PLATFORM_OWNER='@carol-wu'.",
            "REJECTED OWNER DECISION: do not use @platform-team for payments-platform on call.",
            "FINAL DEPENDENCY DECISION: pin pydantic==2.8.2 in requirements.txt.",
            "REJECTED DEPENDENCY DECISION: pydantic==1.10.0 and pydantic==2.6.4 are stale.",
        )
        raw = "\n".join(events) + "\n" + ("Historical project chatter. " * 120)
        task = (
            "Apply final accepted project decisions. Edit src/orbitops/auth.py "
            "src/orbitops/routes/admin_reports.py src/orbitops/export_jobs.py "
            "src/orbitops/invoices.py src/orbitops/flags.py src/orbitops/oncall.py "
            "requirements.txt and make tests pass."
        )
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(
                Path(directory) / "memory.sqlite",
                default_token_budget=160,
                target_reduction=0.75,
                lean_threshold=0,
                prompt_codec="compact",
            )
            for event in events:
                memory.remember("repo", event, scope="project", label="@repo")

            packet = memory.prepare_step("repo", task, original_context=raw, retrieve_limit=2)

            self.assertLess(packet.memory_tokens, packet.raw_tokens)
            self.assertEqual(packet.retrieved, ())
            self.assertIn("src/orbitops/routes/admin_reports.py", packet.payload)
            self.assertIn("@carol-wu", packet.payload)
            self.assertIn("PATCH:", packet.payload)
            self.assertIn("src/orbitops/export_jobs.py: PAGINATION_MODE=cursor; TIMEOUT_SECONDS=30", packet.payload)
            self.assertIn("src/orbitops/oncall.py: PAYMENT_PLATFORM_OWNER=@carol-wu", packet.payload)
            self.assertIn("requirements.txt: pydantic==2.8.2", packet.payload)
            self.assertIn("DO_NOT_USE:", packet.payload)
            self.assertIn("LegacyApiKeyAuth", packet.payload)
            self.assertNotIn("AUTH_MIDDLEWARE=LegacyApiKeyAuth", packet.payload)
            self.assertNotIn("2.8.2:", packet.payload)
            self.assertIn("PAGINATION_MODE=offset", packet.payload)
            self.assertIn("pydantic==1.10.0", packet.payload)
            self.assertIn("@platform-team", packet.payload)
            do_not_use = packet.payload.split("DO_NOT_USE:", 1)[1]
            self.assertNotIn("- cursor", do_not_use)
            self.assertIn("EXECUTION_RULE", packet.payload)

    def test_agentic_memory_packet_preserves_followup_file_focus(self):
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(
                Path(directory) / "memory.sqlite",
                default_token_budget=220,
                target_reduction=0.60,
            )
            memory.remember(
                "ayygent",
                "USER: create a new files called myfirst.py and write a quick hello world function",
                scope="conversation",
            )
            memory.remember(
                "ayygent",
                'ASSISTANT_FINAL: Created `myfirst.py` with a simple hello_world() function.',
                scope="result",
            )
            memory.remember("ayygent", "USER: where is the file u created called myfirst.py", scope="conversation")
            memory.remember(
                "ayygent",
                "ASSISTANT_FINAL: I do not have evidence that I actually created myfirst.py.",
                scope="result",
            )

            packet = memory.prepare_step("ayygent", "yeah tell me if it exists")

            self.assertIn("STATE:", packet.payload)
            self.assertIn("myfirst.py", packet.payload)
            self.assertIn("GROUND inspect repo before file/edit claims.", packet.payload)

    def test_agentic_memory_recalls_first_and_ordered_conversation(self):
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(
                Path(directory) / "memory.sqlite",
                default_token_budget=192,
                target_reduction=0.60,
            )
            exchanges = [
                ("hello", "Hello! How can I help you today?"),
                ("hows it going", "It is going well."),
                ("can you read whats going on in this directory?", "Yes, a Python package named ayygent."),
            ]
            for user, assistant in exchanges:
                memory.remember("new", f"USER: {user}", scope="conversation")
                memory.remember("new", f"ASSISTANT_FINAL: {assistant}", scope="result")

            packet = memory.prepare_step("new", "whats first thing i said to you?")

            # The first user message must be recoverable in order.
            self.assertIn("CONVERSATION_SO_FAR", packet.payload)
            self.assertIn("1. user: hello", packet.payload)
            # The dialog is rendered in chronological order.
            self.assertLess(
                packet.payload.index("1. user: hello"),
                packet.payload.index("can you read whats going on in this directory?"),
            )
            # The conversation block replaces the duplicate task-state dialog lines.
            self.assertNotIn("latest_user evidence=", packet.payload)

    def test_agentic_long_conversation_truncates_with_first_message_anchor(self):
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(
                Path(directory) / "memory.sqlite",
                default_token_budget=192,
                target_reduction=0.60,
                conversation_budget=120,
            )
            memory.remember("new", "USER: my very first distinctive request about apples", scope="conversation")
            memory.remember("new", "ASSISTANT_FINAL: noted apples", scope="result")
            for index in range(40):
                memory.remember("new", f"USER: filler message number {index} about pears and grapes", scope="conversation")
                memory.remember("new", f"ASSISTANT_FINAL: acknowledged filler {index}", scope="result")

            packet = memory.prepare_step("new", "what was the first thing i asked?")

            # Older turns are elided under budget, but the first message is anchored.
            self.assertIn("FIRST_USER_MESSAGE: my very first distinctive request about apples", packet.payload)
            self.assertIn("earlier turn(s) elided", packet.payload)

    def test_agentic_conversation_recall_is_capsule_isolated(self):
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(
                Path(directory) / "memory.sqlite",
                default_token_budget=192,
                target_reduction=0.60,
            )
            memory.remember("default", "USER: explain README.md and pyproject.toml", scope="conversation")
            memory.remember("default", "ASSISTANT_FINAL: It is a packaging project.", scope="result")
            memory.remember("new", "USER: brand new topic about coffee", scope="conversation")

            packet = memory.prepare_step("new", "what were we talking about?")

            self.assertIn("coffee", packet.payload)
            self.assertNotIn("README.md", packet.payload)
            self.assertNotIn("pyproject.toml", packet.payload)

    def test_mcp_server_tools_call_memory_core(self):
        with TemporaryDirectory() as directory:
            server = MemoryMCPServer(Path(directory) / "memory.sqlite")
            remember = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_remember",
                        "arguments": {
                            "capsule_id": "greenhouse",
                            "scope": "greenhouse",
                            "text": "Use ESP32-C3 with MicroPython. Pump voltage: 12V.",
                        },
                    },
                }
            )
            answer = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_answer",
                        "arguments": {
                            "capsule_id": "greenhouse",
                            "question": "What pump voltage should I use?",
                        },
                    },
                }
            )

            self.assertIn("result", remember)
            text = answer["result"]["content"][0]["text"]
            self.assertIn('"answer": "12V"', text)

    def test_mcp_retrieve_and_expand_preserve_old_stale_facts(self):
        with TemporaryDirectory() as directory:
            server = MemoryMCPServer(Path(directory) / "memory.sqlite")
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_remember",
                        "arguments": {
                            "capsule_id": "repo",
                            "scope": "project",
                            "text": "DECISION (superseded): Use LegacyApiKeyAuth for admin reports.",
                        },
                    },
                }
            )
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_remember",
                        "arguments": {
                            "capsule_id": "repo",
                            "scope": "project",
                            "text": "REVISED DECISION: Replace LegacyApiKeyAuth with RequireSessionToken across admin routes.",
                        },
                    },
                }
            )

            retrieved = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_retrieve",
                        "arguments": {
                            "capsule_id": "repo",
                            "query": "What stale rejected auth decision mentioned LegacyApiKeyAuth?",
                            "limit": 2,
                        },
                    },
                }
            )
            snippets = json.loads(retrieved["result"]["content"][0]["text"])["snippets"]
            self.assertTrue(any("LegacyApiKeyAuth" in snippet["text"] for snippet in snippets))

            expanded = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_expand",
                        "arguments": {
                            "capsule_id": "repo",
                            "source": snippets[0]["source"],
                        },
                    },
                }
            )
            self.assertIn("LegacyApiKeyAuth", expanded["result"]["content"][0]["text"])

    def test_current_prompt_hides_stale_but_history_prompt_can_show_it(self):
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(
                Path(directory) / "memory.sqlite",
                default_token_budget=180,
                lean_threshold=0,
            )
            events = (
                "DECISION (superseded): Use LegacyApiKeyAuth for admin reports.",
                "Implementation starts: AUTH_MIDDLEWARE = LegacyApiKeyAuth in src/auth/middleware.py.",
                "REVISED DECISION: Replace LegacyApiKeyAuth with RequireSessionToken across admin routes.",
                "active_auth_middleware=RequireSessionToken edit_file=src/auth/middleware.py.",
            )
            for event in events:
                memory.remember("repo", event, scope="project", label="@repo")

            current = memory.prepare_step(
                "repo",
                "Apply the current admin auth migration in src/auth/middleware.py.",
                original_context="\n".join(events),
            )
            self.assertIn("RequireSessionToken", current.payload)
            self.assertNotIn("LegacyApiKeyAuth", current.payload)

            history = memory.prepare_step(
                "repo",
                "Why was LegacyApiKeyAuth replaced by RequireSessionToken?",
                original_context="\n".join(events),
            )
            self.assertIn("RequireSessionToken", history.payload)
            self.assertIn("LegacyApiKeyAuth", history.payload)

    def test_mcp_compact_turn_returns_strict_outbound_messages(self):
        with TemporaryDirectory() as directory:
            server = MemoryMCPServer(Path(directory) / "memory.sqlite")
            transcript = (
                "Long prior transcript with tool chatter. " * 300
                + "Final decision: use ESP32-C3 with MicroPython. Pump voltage: 12V."
            )
            response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_compact_turn",
                        "arguments": {
                            "capsule_id": "greenhouse",
                            "scope": "greenhouse",
                            "transcript_to_remember": transcript,
                            "current_message": {"role": "user", "content": "What pump voltage should I use?"},
                            "system_messages": [{"role": "system", "content": "Answer from MCP memory."}],
                            "max_recent_messages": 0,
                            "max_outbound_tokens": 900,
                            "min_reduction": 0.60,
                        },
                    },
                }
            )

            data = json.loads(response["result"]["content"][0]["text"])
            outbound = json.dumps(data["messages"])

            self.assertTrue(data["passed"])
            self.assertGreaterEqual(data["reduction"], 0.60)
            self.assertNotIn(transcript, outbound)
            self.assertEqual(data["messages"][-1]["content"], "What pump voltage should I use?")
            self.assertIn("pump_voltage=12V", data["messages"][0]["content"])

    def test_mcp_compact_turn_rejects_recent_history_replay(self):
        with TemporaryDirectory() as directory:
            server = MemoryMCPServer(Path(directory) / "memory.sqlite")
            response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_compact_turn",
                        "arguments": {
                            "capsule_id": "greenhouse",
                            "transcript_to_remember": "Final decision: use ESP32-C3.",
                            "current_message": {"role": "user", "content": "What board?"},
                            "recent_messages": [{"role": "user", "content": "old message"}],
                            "max_recent_messages": 0,
                        },
                    },
                }
            )

            self.assertIn("error", response)
            self.assertIn("recent_messages exceeds", response["error"]["message"])

    def test_estimates_input_cost_savings_from_prepared_packet(self):
        with TemporaryDirectory() as directory:
            memory = AgenticMemoryLayer(Path(directory) / "memory.sqlite")
            raw = (
                "Irrelevant transcript noise. " * 100
                + "Final greenhouse decision: use ESP32-C3 with MicroPython. Pump voltage: 12V."
            )
            memory.remember("greenhouse", raw, scope="greenhouse")
            packet = memory.prepare_step("greenhouse", "What pump voltage should I use?", original_context=raw)

            estimate = estimate_input_savings(packet, input_price_per_1m=2.0, turns=10)

            self.assertEqual(estimate.raw_tokens_total, estimate.raw_tokens_per_turn * 10)
            self.assertEqual(estimate.memory_tokens_total, estimate.memory_tokens_per_turn * 10)
            self.assertGreater(estimate.tokens_saved_per_turn, 0)
            self.assertGreater(estimate.input_cost_saved, 0)


if __name__ == "__main__":
    unittest.main()
