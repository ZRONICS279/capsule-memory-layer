# Benchmark Summary

Best adapter: bm25_rag with cross-session score 0.500.

## What Was Tested

8 task(s), 32 adapter run(s), mode(s): chat_retrieval. Families: conversational_continuity, deletion_forgetting, episodic_recall, general_chat_mixed, personal_preference, preference_update, privacy_boundary, temporal_preference.

## Results

| Adapter | Success | CtxOK | Recall | Stale | Privacy | Delete | Score | Raw Tok | Prep Tok | Savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| capsule | n/a | 0.375 | 0.771 | 0.250 | 0.125 | 0.125 | 0.458 | 130 | 147 | 10.3% |
| bm25_rag | n/a | 0.500 | 0.938 | 0.375 | 0.000 | 0.125 | 0.500 | 130 | 56 | 52.3% |
| langgraph_real | n/a | 0.375 | 1.000 | 0.375 | 0.125 | 0.125 | 0.375 | 130 | 142 | 0.2% |
| no_memory | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 130 | 0 | 100.0% |

## Plain-English Comparison

Capsule compared with every baseline:

Compared with `bm25_rag`, capsule:
- Strict clean-context pass rate (`CtxOK`): 37.5% vs 50.0% (-12.5 points, 0.8x, -25% relative).
- Current-fact recall: 77.1% vs 93.8% (-16.7 points, 0.8x, -18% relative).
- Stale fact leakage: 25.0% vs 37.5% (12.5 points lower, 33% reduction).
- Cross-session score: 45.8% vs 50.0% (-4.2 points, 0.9x, -8% relative).
- Token savings: 10.3% vs 52.3% (-42.0 points, 0.2x, -80% relative).
- Privacy leakage: 12.5% vs 0.0% (-12.5 percentage points lower).
- Deletion/forgetting leakage: 12.5% vs 12.5% (0.0 points lower, 0% reduction).

Compared with `langgraph_real`, capsule:
- Strict clean-context pass rate (`CtxOK`): 37.5% vs 37.5% (+0.0 points, 1.0x, +0% relative).
- Current-fact recall: 77.1% vs 100.0% (-22.9 points, 0.8x, -23% relative).
- Stale fact leakage: 25.0% vs 37.5% (12.5 points lower, 33% reduction).
- Cross-session score: 45.8% vs 37.5% (+8.3 points, 1.2x, +22% relative).
- Token savings: 10.3% vs 0.2% (+10.1 points, 51.5x, +5050% relative).
- Privacy leakage: 12.5% vs 12.5% (0.0 points lower, 0% reduction).
- Deletion/forgetting leakage: 12.5% vs 12.5% (0.0 points lower, 0% reduction).

Compared with `no_memory`, capsule:
- Strict clean-context pass rate (`CtxOK`): 37.5% vs 0.0% (+37.5 percentage points).
- Current-fact recall: 77.1% vs 0.0% (+77.1 percentage points).
- Stale fact leakage: 25.0% vs 0.0% (-25.0 percentage points lower).
- Cross-session score: 45.8% vs 0.0% (+45.8 percentage points).
- Token savings: 10.3% vs 100.0% (-89.7 points, 0.1x, -90% relative).
- Privacy leakage: 12.5% vs 0.0% (-12.5 percentage points lower).
- Deletion/forgetting leakage: 12.5% vs 0.0% (-12.5 percentage points lower).

## Interpretation

Use this table to compare current-fact recall, stale leakage, context success, and token savings. Include mem0_real or mem0_real_async in the run for a direct capsule-vs-mem0 claim.
