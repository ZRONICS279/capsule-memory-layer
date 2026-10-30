# Benchmark Suite Description

This suite evaluates agentic coding memory under evolving project history. Each task pre-ingests prior work sessions into a memory adapter, then asks the agent or retrieval layer to use only retrieved memory, not the raw transcript.

## Task Families

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
