# Benchmark Summary

Capsule beat mem0_real_async by +1.000 average cross-session score over 3 paired runs; +100.0 percentage points higher score (95% bootstrap CI +1.000 to +1.000).

## What Was Tested

3 task(s), 12 adapter run(s), mode(s): retrieval_only. Families: long_horizon_project_memory.

## Results

| Adapter | Success | CtxOK | Recall | Stale | Score | Raw Tok | Prep Tok | Savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mem0_real_async | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 5535 | 445 | 91.9% |
| capsule | n/a | 1.000 | 1.000 | 0.000 | 1.000 | 5535 | 466 | 91.6% |
| bm25_rag | n/a | 0.000 | 0.500 | 0.333 | 0.333 | 5535 | 152 | 97.3% |
| no_memory | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 5535 | 0 | 100.0% |

## Plain-English Comparison

Capsule compared with every baseline:

Compared with `mem0_real_async`, capsule:
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 0.0% (+100.0 percentage points).
- Stale fact leakage: 0.0% vs 0.0% (+0.0 percentage points lower).
- Cross-session score: 100.0% vs 0.0% (+100.0 percentage points).
- Token savings: 91.6% vs 91.9% (-0.3 points, 1.0x, -0% relative).

Compared with `bm25_rag`, capsule:
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 50.0% (+50.0 points, 2.0x, +100% relative).
- Stale fact leakage: 0.0% vs 33.3% (33.3 points lower, 100% reduction).
- Cross-session score: 100.0% vs 33.3% (+66.7 points, 3.0x, +200% relative).
- Token savings: 91.6% vs 97.3% (-5.7 points, 0.9x, -6% relative).

Compared with `no_memory`, capsule:
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 0.0% (+100.0 percentage points).
- Stale fact leakage: 0.0% vs 0.0% (+0.0 percentage points lower).
- Cross-session score: 100.0% vs 0.0% (+100.0 percentage points).
- Token savings: 91.6% vs 100.0% (-8.4 points, 0.9x, -8% relative).

## Interpretation

The result supports a narrow claim about cross-session agentic coding memory: current-fact recall, stale-fact suppression, and token efficiency under evolving project history. It does not by itself prove global superiority across every memory workload.
