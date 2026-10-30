# Benchmark Summary

Capsule beat mem0_real_async by +0.875 average cross-session score over 20 paired runs; +87.5 percentage points higher score (8.0x, +700% relative) (95% bootstrap CI +0.767 to +0.967).

## What Was Tested

10 task(s), 80 adapter run(s), mode(s): agentic. Families: auth_decision_evolution, config_iteration, dependency_version_pin, export_jobs_config, feature_flag_rollout, long_horizon_project_memory, route_migration, service_ownership, stale_docs_current_code, unknown_owner_abstention.

## Results

| Adapter | Success | CtxOK | Recall | Stale | Score | Raw Tok | Prep Tok | Savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mem0_real_async | 0.900 | 0.000 | 0.775 | 0.700 | 0.125 | 794 | 283 | 26.1% |
| capsule | 0.750 | 1.000 | 1.000 | 0.000 | 1.000 | 794 | 194 | 44.8% |
| bm25_rag | 0.850 | 0.000 | 0.900 | 0.883 | 0.083 | 794 | 107 | 64.7% |
| no_memory | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 794 | 0 | 100.0% |

## Plain-English Comparison

Compared with `mem0_real_async`, capsule:
- Agent task success: 75.0% vs 90.0% (-15.0 points, 0.8x, -17% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 77.5% (+22.5 points, 1.3x, +29% relative).
- Stale fact leakage: 0.0% vs 70.0% (70.0 points lower, 100% reduction).
- Cross-session score: 100.0% vs 12.5% (+87.5 points, 8.0x, +700% relative).
- Token savings: 44.8% vs 26.1% (+18.7 points, 1.7x, +72% relative).

## Interpretation

The result supports a narrow claim about cross-session agentic coding memory: current-fact recall, stale-fact suppression, and token efficiency under evolving project history. It does not by itself prove global superiority across every memory workload.

## Caveat

no_memory have no persistent store in skip-ingest/two-phase runs, so their retrieval rows are expected to be empty. Use these rows as lower-bound/product-realism baselines, not as a fresh-ingest RAG comparison.
