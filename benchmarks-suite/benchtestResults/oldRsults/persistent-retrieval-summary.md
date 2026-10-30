# Benchmark Summary

Capsule beat mem0_real_async by +0.825 average cross-session score over 20 paired runs; +82.5 percentage points higher score (5.7x, +471% relative) (95% bootstrap CI +0.667 to +0.958).

## What Was Tested

20 task(s), 80 adapter run(s), mode(s): agentic. Families: auth_decision_evolution, config_iteration, dependency_version_pin, export_jobs_config, feature_flag_rollout, long_horizon_project_memory, route_migration, service_ownership, stale_docs_current_code, unknown_owner_abstention.

## Results

| Adapter | Success | CtxOK | Recall | Stale | Score | Raw Tok | Prep Tok | Savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mem0_real_async | 0.900 | 0.100 | 0.825 | 0.700 | 0.175 | 831 | 284 | 27.0% |
| capsule | 0.850 | 1.000 | 1.000 | 0.000 | 1.000 | 831 | 193 | 47.4% |
| bm25_rag | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 831 | 0 | 100.0% |
| no_memory | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 831 | 0 | 100.0% |

## Plain-English Comparison

Compared with `mem0_real_async`, capsule:
- Agent task success: 85.0% vs 90.0% (-5.0 points, 0.9x, -6% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 10.0% (+90.0 points, 10.0x, +900% relative).
- Current-fact recall: 100.0% vs 82.5% (+17.5 points, 1.2x, +21% relative).
- Stale fact leakage: 0.0% vs 70.0% (70.0 points lower, 100% reduction).
- Cross-session score: 100.0% vs 17.5% (+82.5 points, 5.7x, +471% relative).
- Token savings: 47.4% vs 27.0% (+20.4 points, 1.8x, +76% relative).

## Interpretation

The result supports a narrow claim about cross-session agentic coding memory: current-fact recall, stale-fact suppression, and token efficiency under evolving project history. It does not by itself prove global superiority across every memory workload.

## Caveat

bm25_rag, no_memory have no persistent store in skip-ingest/two-phase runs, so their retrieval rows are expected to be empty. Use these rows as lower-bound/product-realism baselines, not as a fresh-ingest RAG comparison.
