# Benchmark Summary

Capsule beat mem0_real_async by +0.883 average cross-session score over 20 paired runs; +88.3 percentage points higher score (8.6x, +757% relative) (95% bootstrap CI +0.767 to +0.975).

## What Was Tested

20 task(s), 80 adapter run(s), mode(s): retrieval_only. Families: auth_decision_evolution, config_iteration, dependency_version_pin, export_jobs_config, feature_flag_rollout, long_horizon_project_memory, route_migration, service_ownership, stale_docs_current_code, unknown_owner_abstention.

## Results

| Adapter | Success | CtxOK | Recall | Stale | Score | Raw Tok | Prep Tok | Savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mem0_real_async | n/a | 0.000 | 0.767 | 0.700 | 0.117 | 831 | 286 | 26.9% |
| capsule | n/a | 1.000 | 1.000 | 0.000 | 1.000 | 831 | 193 | 47.4% |
| bm25_rag | n/a | 0.000 | 0.900 | 0.908 | 0.058 | 831 | 107 | 66.0% |
| no_memory | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 831 | 0 | 100.0% |

## Plain-English Comparison

Capsule compared with every baseline:

Compared with `mem0_real_async`, capsule:
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 76.7% (+23.3 points, 1.3x, +30% relative).
- Stale fact leakage: 0.0% vs 70.0% (70.0 points lower, 100% reduction).
- Cross-session score: 100.0% vs 11.7% (+88.3 points, 8.6x, +757% relative).
- Token savings: 47.4% vs 26.9% (+20.5 points, 1.8x, +76% relative).

Compared with `bm25_rag`, capsule:
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 90.0% (+10.0 points, 1.1x, +11% relative).
- Stale fact leakage: 0.0% vs 90.8% (90.8 points lower, 100% reduction).
- Cross-session score: 100.0% vs 5.8% (+94.2 points, 17.2x, +1615% relative).
- Token savings: 47.4% vs 66.0% (-18.6 points, 0.7x, -28% relative).

Compared with `no_memory`, capsule:
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 0.0% (+100.0 percentage points).
- Stale fact leakage: 0.0% vs 0.0% (+0.0 percentage points lower).
- Cross-session score: 100.0% vs 0.0% (+100.0 percentage points).
- Token savings: 47.4% vs 100.0% (-52.6 points, 0.5x, -53% relative).

## Interpretation

The result supports a narrow claim about cross-session agentic coding memory: current-fact recall, stale-fact suppression, and token efficiency under evolving project history. It does not by itself prove global superiority across every memory workload.
