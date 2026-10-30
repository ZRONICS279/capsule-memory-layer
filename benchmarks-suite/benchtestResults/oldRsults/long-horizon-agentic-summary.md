# Benchmark Summary

Capsule beat mem0_real_async by +0.847 average cross-session score over 2 paired runs; +84.7 percentage points higher score (6.5x, +554% relative) (95% bootstrap CI +0.833 to +0.861).

## What Was Tested

2 task(s), 8 adapter run(s), mode(s): agentic. Families: long_horizon_project_memory.

## Results

| Adapter | Success | CtxOK | Recall | Stale | Score | Raw Tok | Prep Tok | Savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mem0_real_async | 0.500 | 0.000 | 0.167 | 0.083 | 0.153 | 5292 | 414 | 92.1% |
| capsule | 0.500 | 1.000 | 1.000 | 0.000 | 1.000 | 5292 | 466 | 91.2% |
| bm25_rag | 0.000 | 0.000 | 0.500 | 0.333 | 0.333 | 5292 | 152 | 97.2% |
| no_memory | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 5292 | 0 | 100.0% |

## Plain-English Comparison

Compared with `mem0_real_async`, capsule:
- Agent task success: 50.0% vs 50.0% (+0.0 points, 1.0x, +0% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 16.7% (+83.3 points, 6.0x, +500% relative).
- Stale fact leakage: 0.0% vs 8.3% (8.3 points lower, 100% reduction).
- Cross-session score: 100.0% vs 15.3% (+84.7 points, 6.5x, +554% relative).
- Token savings: 91.2% vs 92.1% (-0.9 points, 1.0x, -1% relative).

## Interpretation

The result supports a narrow claim about cross-session agentic coding memory: current-fact recall, stale-fact suppression, and token efficiency under evolving project history. It does not by itself prove global superiority across every memory workload.

## Caveat

no_memory have no persistent store in skip-ingest/two-phase runs, so their retrieval rows are expected to be empty. Use these rows as lower-bound/product-realism baselines, not as a fresh-ingest RAG comparison.
