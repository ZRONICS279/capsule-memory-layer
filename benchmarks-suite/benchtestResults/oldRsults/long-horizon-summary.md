# Benchmark Summary

Capsule beat mem0_real by +0.389 average cross-session score over 1 paired runs (95% bootstrap CI +0.389 to +0.389).

## What Was Tested

1 task(s), 4 adapter run(s), mode(s): retrieval_only. Families: long_horizon_project_memory.

## Results

| Adapter | Success | CtxOK | Recall | Stale | Score | Raw Tok | Prep Tok | Savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mem0_real | n/a | 0.000 | 0.333 | 0.167 | 0.278 | 5049 | 440 | 91.3% |
| capsule | n/a | 0.000 | 0.667 | 0.000 | 0.667 | 5049 | 559 | 88.9% |
| bm25_rag | n/a | 0.000 | 0.500 | 0.333 | 0.333 | 5049 | 152 | 97.0% |
| no_memory | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 5049 | 0 | 100.0% |

## Interpretation

The result supports a narrow claim about cross-session agentic coding memory: current-fact recall, stale-fact suppression, and token efficiency under evolving project history. It does not by itself prove global superiority across every memory workload.
