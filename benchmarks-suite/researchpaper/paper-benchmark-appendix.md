# Paper Benchmark Appendix

# Benchmark Suite Description

This suite evaluates agentic coding memory under evolving project history. Each task pre-ingests prior work sessions into a memory adapter, then asks the agent or retrieval layer to use only retrieved memory, not the raw transcript.

## Task Families

### `auth_decision_evolution`

Tests whether memory preserves the final auth decision after an earlier auth approach was superseded.

- Task id example: `cs_auth_evolution_000`
- Prior sessions: 4
- Memory events: 24
- Agent prompt: Continue the admin reports auth migration and make the targeted test pass.
- Expected files: `src/auth/middleware.py`, `src/api/routes/admin_reports.py`
- Test command: `python3 -m unittest tests.test_admin_reports`
- Required current facts: `RequireSessionToken`, `src/auth/middleware.py`
- Stale facts penalized: `LegacyApiKeyAuth`
- Forbidden output/code patterns: `LegacyApiKeyAuth`
- Abstention expected: `False`

### `config_iteration`

Tests whether memory keeps the final config key after multiple rejected naming iterations.

- Task id example: `cs_config_iteration_000`
- Prior sessions: 3
- Memory events: 13
- Agent prompt: Apply the current billing retry config key and run the retry policy test.
- Expected files: `src/billing/retry_policy.py`
- Test command: `python3 -m unittest tests.test_retry_policy`
- Required current facts: `PAYMENT_RETRY_LIMIT_0`, `src/billing/retry_policy.py`
- Stale facts penalized: `BILLING_RETRY_COUNT`, `MAX_RETRIES_3`
- Forbidden output/code patterns: `BILLING_RETRY_COUNT`
- Abstention expected: `False`

### `route_migration`

Tests whether memory recalls the final API endpoint and file path while suppressing deprecated routes.

- Task id example: `cs_route_migration_000`
- Prior sessions: 3
- Memory events: 13
- Agent prompt: Finish the report export route rename and make the route test pass.
- Expected files: `src/api/routes/report_export.py`
- Test command: `python3 -m unittest tests.test_report_export`
- Required current facts: `/api/v3/reports/0/export`, `src/api/routes/report_export.py`
- Stale facts penalized: `/api/v1/export`
- Forbidden output/code patterns: `/api/v1/export`
- Abstention expected: `False`

### `feature_flag_rollout`

Tests whether memory handles changing rollout states such as disabled, canary, reverted, and enabled.

- Task id example: `cs_feature_flag_000`
- Prior sessions: 5
- Memory events: 39
- Agent prompt: Apply the final checkout feature flag state and make the flag test pass.
- Expected files: `src/checkout/flags.py`
- Test command: `python3 -m unittest tests.test_checkout_flags`
- Required current facts: `enabled`, `src/checkout/flags.py`
- Stale facts penalized: `disabled`, `canary`
- Forbidden output/code patterns: `'disabled'`, `'canary'`
- Abstention expected: `False`

### `service_ownership`

Tests whether memory resolves ownership changes and returns only the current owner.

- Task id example: `cs_service_ownership_000`
- Prior sessions: 4
- Memory events: 30
- Agent prompt: Update the payments-platform on-call owner to the current owner and make the test pass.
- Expected files: `src/ops/oncall.py`
- Test command: `python3 -m unittest tests.test_oncall`
- Required current facts: `carol-wu`, `src/ops/oncall.py`
- Stale facts penalized: `alice-chen`, `bob-martinez`, `platform-team`
- Forbidden output/code patterns: `@alice-chen`, `@bob-martinez`, `@platform-team`
- Abstention expected: `False`

### `export_jobs_config`

Tests multi-fact recall: two final config values must be retrieved together without stale defaults.

- Task id example: `cs_export_jobs_000`
- Prior sessions: 4
- Memory events: 29
- Agent prompt: Apply the finalized export jobs config (pagination mode and timeout) and make the test pass.
- Expected files: `src/api/routes/export_jobs.py`
- Test command: `python3 -m unittest tests.test_export_jobs`
- Required current facts: `cursor`, `TIMEOUT_SECONDS`, `src/api/routes/export_jobs.py`
- Stale facts penalized: `offset`
- Forbidden output/code patterns: `'offset'`, `TIMEOUT_SECONDS = 5`
- Abstention expected: `False`

### `stale_docs_current_code`

Tests stale-document conflict handling: the agent must follow current code/test facts, not old docs.

- Task id example: `cs_stale_docs_code_000`
- Prior sessions: 3
- Memory events: 12
- Agent prompt: Use the active invoice normalizer from current code, not stale docs.
- Expected files: `src/invoices/normalizer.py`
- Test command: `python3 -m unittest tests.test_normalizer`
- Required current facts: `normalize_active_invoice_0`, `src/invoices/normalizer.py`
- Stale facts penalized: `normalize_legacy_invoice`
- Forbidden output/code patterns: `normalize_legacy_invoice`
- Abstention expected: `False`

### `dependency_version_pin`

Tests dependency decision memory after older package versions were superseded.

- Task id example: `cs_dependency_pin_000`
- Prior sessions: 3
- Memory events: 12
- Agent prompt: Apply the accepted dependency pin and make the dependency pin test pass.
- Expected files: `pyproject.toml`
- Test command: `python3 -m unittest tests.test_dependency_pins`
- Required current facts: `pydantic`, `2.7.0`, `pyproject.toml`
- Stale facts penalized: `1.10.0`, `2.6.4`
- Forbidden output/code patterns: `pydantic==1.10.0`, `pydantic==2.6.4`
- Abstention expected: `False`

### `unknown_owner_abstention`

Tests abstention: memory should not invent a fact when no final owner decision exists.

- Task id example: `cs_unknown_owner_000`
- Prior sessions: 3
- Memory events: 9
- Agent prompt: Who owns observability alert routing? Only act if a selected owner exists.
- Expected files: `none`
- Test command: `python3 -m unittest tests.test_observability`
- Required current facts: `no selected owner`, `UNKNOWN`
- Stale facts penalized: `platform-team`
- Forbidden output/code patterns: `none`
- Abstention expected: `True`

### `long_horizon_project_memory`

Tests 100+ event project memory under heavy noise, repeated stale facts, and multiple current facts.

- Task id example: `cs_long_horizon_project_000`
- Prior sessions: 11
- Memory events: 130
- Agent prompt: Apply the current export jobs config and current payments-platform on-call owner. Make the export jobs and on-call tests pass.
- Expected files: `src/api/routes/export_jobs.py`, `src/ops/oncall.py`
- Test command: `python3 -m unittest tests.test_export_jobs tests.test_oncall`
- Required current facts: `cursor`, `TIMEOUT_SECONDS`, `30`, `@carol-wu`, `src/api/routes/export_jobs.py`, `src/ops/oncall.py`
- Stale facts penalized: `offset`, `page_number`, `batch`, `@alice-chen`, `@bob-martinez`, `@platform-team`
- Forbidden output/code patterns: `'offset'`, `TIMEOUT_SECONDS = 5`, `@alice-chen`, `@bob-martinez`, `@platform-team`
- Abstention expected: `False`

## What The Scores Mean

- Recall: fraction of required current facts found in retrieved memory.
- Stale: fraction of superseded facts that leaked into active retrieved memory.
- Score: with `--score-formula multiplicative`, `recall * (1 - stale)`.
- CtxOK: strict pass where all required facts are present and no stale facts leaked.
- Success: agentic run only; tests pass, expected files are changed, forbidden patterns are absent.
- Token savings: prepared memory context compared with injecting the full raw session history.

## What This Proves

A win on retrieval-only runs proves cleaner memory retrieval: more current facts, fewer stale facts, fewer tokens. A win on agentic runs proves downstream task value: the coding agent solved more tasks or solved them with lower memory/context cost. These results support a claim about cross-session software-engineering memory, not universal memory superiority.


# 1. Long-Horizon Retrieval-Only Results

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


# 2. Long-Horizon Agentic Solving Results

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

Capsule compared with every baseline:

Compared with `mem0_real_async`, capsule:
- Agent task success: 50.0% vs 50.0% (+0.0 points, 1.0x, +0% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 16.7% (+83.3 points, 6.0x, +500% relative).
- Stale fact leakage: 0.0% vs 8.3% (8.3 points lower, 100% reduction).
- Cross-session score: 100.0% vs 15.3% (+84.7 points, 6.5x, +554% relative).
- Token savings: 91.2% vs 92.1% (-0.9 points, 1.0x, -1% relative).

Compared with `bm25_rag`, capsule:
- Agent task success: 50.0% vs 0.0% (+50.0 percentage points).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 50.0% (+50.0 points, 2.0x, +100% relative).
- Stale fact leakage: 0.0% vs 33.3% (33.3 points lower, 100% reduction).
- Cross-session score: 100.0% vs 33.3% (+66.7 points, 3.0x, +200% relative).
- Token savings: 91.2% vs 97.2% (-6.0 points, 0.9x, -6% relative).

Compared with `no_memory`, capsule:
- Agent task success: 50.0% vs 50.0% (+0.0 points, 1.0x, +0% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 0.0% (+100.0 percentage points).
- Stale fact leakage: 0.0% vs 0.0% (+0.0 percentage points lower).
- Cross-session score: 100.0% vs 0.0% (+100.0 percentage points).
- Token savings: 91.2% vs 100.0% (-8.8 points, 0.9x, -9% relative).

## Interpretation

The result supports a narrow claim about cross-session agentic coding memory: current-fact recall, stale-fact suppression, and token efficiency under evolving project history. It does not by itself prove global superiority across every memory workload.

## Caveat

no_memory have no persistent store in skip-ingest/two-phase runs, so their retrieval rows are expected to be empty. Use these rows as lower-bound/product-realism baselines, not as a fresh-ingest RAG comparison.


# 3. Full Cross-Session Retrieval-Only Results

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


# 4. Full Cross-Session Agentic Benchmark Results

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

Capsule compared with every baseline:

Compared with `mem0_real_async`, capsule:
- Agent task success: 75.0% vs 90.0% (-15.0 points, 0.8x, -17% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 77.5% (+22.5 points, 1.3x, +29% relative).
- Stale fact leakage: 0.0% vs 70.0% (70.0 points lower, 100% reduction).
- Cross-session score: 100.0% vs 12.5% (+87.5 points, 8.0x, +700% relative).
- Token savings: 44.8% vs 26.1% (+18.7 points, 1.7x, +72% relative).

Compared with `bm25_rag`, capsule:
- Agent task success: 75.0% vs 85.0% (-10.0 points, 0.9x, -12% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 90.0% (+10.0 points, 1.1x, +11% relative).
- Stale fact leakage: 0.0% vs 88.3% (88.3 points lower, 100% reduction).
- Cross-session score: 100.0% vs 8.3% (+91.7 points, 12.0x, +1100% relative).
- Token savings: 44.8% vs 64.7% (-19.9 points, 0.7x, -31% relative).

Compared with `no_memory`, capsule:
- Agent task success: 75.0% vs 75.0% (+0.0 points, 1.0x, +0% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 0.0% (+100.0 percentage points).
- Stale fact leakage: 0.0% vs 0.0% (+0.0 percentage points lower).
- Cross-session score: 100.0% vs 0.0% (+100.0 percentage points).
- Token savings: 44.8% vs 100.0% (-55.2 points, 0.4x, -55% relative).

## Interpretation

The result supports a narrow claim about cross-session agentic coding memory: current-fact recall, stale-fact suppression, and token efficiency under evolving project history. It does not by itself prove global superiority across every memory workload.

## Caveat

no_memory have no persistent store in skip-ingest/two-phase runs, so their retrieval rows are expected to be empty. Use these rows as lower-bound/product-realism baselines, not as a fresh-ingest RAG comparison.


# 5. Persistent Two-Phase Memory Results

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

Capsule compared with every baseline:

Compared with `mem0_real_async`, capsule:
- Agent task success: 85.0% vs 90.0% (-5.0 points, 0.9x, -6% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 10.0% (+90.0 points, 10.0x, +900% relative).
- Current-fact recall: 100.0% vs 82.5% (+17.5 points, 1.2x, +21% relative).
- Stale fact leakage: 0.0% vs 70.0% (70.0 points lower, 100% reduction).
- Cross-session score: 100.0% vs 17.5% (+82.5 points, 5.7x, +471% relative).
- Token savings: 47.4% vs 27.0% (+20.4 points, 1.8x, +76% relative).

Compared with `bm25_rag`, capsule:
- Agent task success: 85.0% vs 75.0% (+10.0 points, 1.1x, +13% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 0.0% (+100.0 percentage points).
- Stale fact leakage: 0.0% vs 0.0% (+0.0 percentage points lower).
- Cross-session score: 100.0% vs 0.0% (+100.0 percentage points).
- Token savings: 47.4% vs 100.0% (-52.6 points, 0.5x, -53% relative).

Compared with `no_memory`, capsule:
- Agent task success: 85.0% vs 75.0% (+10.0 points, 1.1x, +13% relative).
- Strict clean-context pass rate (`CtxOK`): 100.0% vs 0.0% (+100.0 percentage points).
- Current-fact recall: 100.0% vs 0.0% (+100.0 percentage points).
- Stale fact leakage: 0.0% vs 0.0% (+0.0 percentage points lower).
- Cross-session score: 100.0% vs 0.0% (+100.0 percentage points).
- Token savings: 47.4% vs 100.0% (-52.6 points, 0.5x, -53% relative).

## Interpretation

The result supports a narrow claim about cross-session agentic coding memory: current-fact recall, stale-fact suppression, and token efficiency under evolving project history. It does not by itself prove global superiority across every memory workload.

## Caveat

bm25_rag, no_memory have no persistent store in skip-ingest/two-phase runs, so their retrieval rows are expected to be empty. Use these rows as lower-bound/product-realism baselines, not as a fresh-ingest RAG comparison.


# 6. Long-Term / General Chat Memory Results

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
