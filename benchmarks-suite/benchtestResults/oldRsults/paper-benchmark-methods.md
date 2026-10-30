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
