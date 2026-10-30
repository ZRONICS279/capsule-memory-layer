# Real-World Memory Benchmark Protocol

This protocol supports a narrow, defensible claim:

> Semantic capsule memory improves cross-session software-engineering memory under evolving requirements by reducing stale fact leakage while preserving current fact recall, compared with mem0 under identical agent, model, tool, and context-budget conditions.

Do not claim that capsule is globally "better memory" unless the benchmark suite covers every memory workload you intend to include in that phrase. A memory system can be better at stale fact suppression and worse at autobiographical chat recall, open-domain personal preference recall, or semantic similarity search. The paper should name the workload where the evidence is strong.

## Executable Benchmarks

The main executable benchmark is:

```bash
python3 benchmarks/cross_session_benchmark.py \
  --adapters mem0_real,capsule,bm25_rag,no_memory \
  --families all \
  --cases-per-family 5 \
  --repeats 3 \
  --randomize-adapter-order \
  --score-formula multiplicative \
  --json-out cross-session-report.json \
  --csv-out cross-session-table.csv \
  --latex-out cross-session-table.tex \
  --progress
```

For a retrieval-only paper table that avoids agent/API variance:

```bash
python3 benchmarks/cross_session_benchmark.py \
  --adapters mem0_real,capsule,bm25_rag,no_memory \
  --families all \
  --cases-per-family 10 \
  --retrieval-only \
  --score-formula multiplicative \
  --json-out retrieval-only-report.json \
  --csv-out retrieval-only-table.csv \
  --latex-out retrieval-only-table.tex \
  --progress
```

For a true two-day persistence simulation:

```bash
python3 benchmarks/cross_session_benchmark.py \
  --adapters mem0_real,capsule \
  --families all \
  --cases-per-family 10 \
  --persistent-dir ./cs-store-paper \
  --load-only \
  --progress

python3 benchmarks/cross_session_benchmark.py \
  --adapters mem0_real,capsule \
  --families all \
  --cases-per-family 10 \
  --persistent-dir ./cs-store-paper \
  --skip-ingest \
  --score-formula multiplicative \
  --json-out cross-session-paper-report.json \
  --csv-out cross-session-paper-table.csv \
  --latex-out cross-session-paper-table.tex \
  --progress
```

Use the project virtualenv if you want the installed spaCy model to be visible:

```bash
.venv/bin/python benchmarks/cross_session_benchmark.py ...
```

## Current Task Families

- `auth_decision_evolution`: stale auth middleware is superseded by a current session-token decision.
- `config_iteration`: rejected config keys are superseded by the final billing retry key.
- `route_migration`: old API endpoint is replaced by the final route.
- `feature_flag_rollout`: disabled/canary/reverted states are superseded by final rollout.
- `service_ownership`: multiple stale owners are superseded by the final owner.
- `export_jobs_config`: two independent final config values must be recalled together.
- `stale_docs_current_code`: stale documentation conflicts with current code/test facts.
- `dependency_version_pin`: older dependency pins are superseded by the final accepted pin.
- `unknown_owner_abstention`: no current owner exists, so the agent should abstain instead of inventing one.
- `long_horizon_project_memory`: 100+ event project history with repeated stale owners/configs and multiple final facts required in one task.

## Metrics

Primary retrieval metrics:

- Current fact recall: fraction of required current facts present in retrieved context.
- Stale fact leakage: fraction of superseded facts present as active context.
- Cross-session score: `recall * (1 - stale_leakage)` when using `--score-formula multiplicative`.
- Context success rate: all required current facts present and no stale facts leaked.
- Token savings: prepared context tokens versus raw session transcript tokens.

Primary agentic metrics:

- Task success rate: hidden tests pass, expected files touched, no forbidden strings, no hallucinated files.
- Total tokens: model input plus output tokens.
- Tool steps and retries.
- Agent failure mode: wrong file, stale fact, missing current fact, or transport/API error.

The JSON report includes bootstrap confidence intervals for recall, stale leakage, context success, and cross-session score. It also includes paired capsule-vs-mem0 deltas when both adapters are present.

## Additional Things To Test

These dimensions test whether "better memory" generalizes beyond this benchmark:

- Preference memory: user preferences that should persist unless explicitly changed.
- Episodic recall: exact prior events, not just extracted project facts.
- Abstention: whether the system says UNKNOWN when memory is insufficient.
- Privacy deletion: whether deleted facts stop appearing after erasure requests.
- Contradiction handling: whether mutually inconsistent facts are exposed, merged, or resolved.
- Multi-tenant isolation: whether facts from one user/project leak into another.
- Temporal reasoning: whether the system understands dates, recency, and version order.
- Paraphrase retrieval: whether facts are found when the query uses different wording.
- Long-noise robustness: whether recall survives hundreds or thousands of irrelevant events.
- Update latency: how quickly a newly written fact becomes retrievable.
- Cost and latency: ingest time, retrieval time, model tokens, and storage footprint.
- Failure recovery: behavior after partial writes, interrupted ingestion, and corrupted stores.

If capsule wins only the cross-session stale-fact workloads, the honest claim is still valuable. It means capsule is better for evolving software-engineering memory, not that it is universally better at every memory task.

## Real Repository Patch Tasks

Real-repo tasks should be a separate benchmark because they need network access, repo checkout, dependency setup, and per-project test commands. The case format should be data-driven:

```json
{
  "case_id": "flask-issue-1234",
  "repo_url": "https://github.com/example/project.git",
  "base_ref": "commit-before-fix",
  "issue_url": "https://github.com/example/project/issues/1234",
  "gold_pr_url": "https://github.com/example/project/pull/1250",
  "memory_sessions": [
    ["Issue triage notes...", "Rejected approach..."],
    ["Maintainer review...", "Accepted current approach..."]
  ],
  "prompt": "Fix the reported bug using current project decisions.",
  "test_command": "python -m pytest tests/test_target.py",
  "expected_files": ["src/package/target.py"],
  "forbidden_patterns": ["rejected_old_api"]
}
```

The benchmark runner should:

- clone or reuse a local cache of the repository
- checkout `base_ref`
- install dependencies once per repo cache
- copy a clean worktree per adapter/run
- ingest `memory_sessions`
- run retrieval-only scoring
- run the agentic patch task
- run the case-specific tests
- score changed files, forbidden patterns, tests, token usage, cost, and latency

Good first real-repo candidates are small Python or TypeScript repositories with fast tests and historical bugfix PRs. Avoid large monorepos until the runner can cache dependencies and isolate failures.

## Suggested Paper Table

Report both retrieval-only and downstream agentic results:

```text
Adapter      Success  CtxOK  Recall  Stale  Score  Tokens
----------------------------------------------------------
no_memory       n/a  0.000   0.000  0.000  0.000       0
bm25_rag        n/a  0.000   1.000  1.000  0.000       0
mem0_real     0.xxx  x.xxx   x.xxx  x.xxx  x.xxx    xxxx
capsule       0.xxx  x.xxx   x.xxx  x.xxx  x.xxx    xxxx
```

Then include paired deltas:

```text
Capsule - mem0_real cross-session score delta: +X.XXX, 95% CI [L, H]
Capsule stale leakage reduction: Y percentage points
Capsule downstream success gain: Z percentage points
```
