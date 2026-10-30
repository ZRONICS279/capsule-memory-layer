# Real Memory Baseline Setup

This benchmark has two kinds of memory adapters:

- Contract/simulation baselines: `mem0`, `langgraph`, `zep`, `graphiti`, `memgpt`, `letta`
- Real SDK/back-end baselines: `mem0_real`, `langgraph_real`, `zep_real`, `graphiti_real`, `letta_real`, `memgpt_real`

Use the `_real` names when you want to compare against actual systems instead of local approximations.

## Install

```bash
cd benchmarks-suite
python3 -m pip install -r requirements-real-memory.txt
```

## Environment

`mem0_real` needs an OpenAI-compatible key because Mem0 performs real LLM extraction and embedding search.

```bash
export OPENAI_API_KEY="your-key"
# or
export AYYGENT_AI_API_KEY="your-key"
```

Optional Mem0 overrides:

```bash
export SCM_MEM0_LLM_MODEL="gpt-4.1-mini"
export SCM_MEM0_EMBEDDING_MODEL="text-embedding-3-small"
export SCM_MEM0_EMBEDDING_DIMS=1536
```

`zep_real` / `graphiti_real` also need Zep account/server configuration:

```bash
export ZEP_API_KEY="your-zep-key"
```

Current note: `zep-python` fails to import under Python 3.14 in this environment because of Pydantic v1 compatibility. Use a Python 3.12 virtualenv for real Zep runs.

`letta_real` / `memgpt_real` need a modern Letta client/server install and configuration. The package named `letta` resolved to an old empty package here under Python 3.14, so use Python 3.12 and Letta's current setup docs before running those adapters.

## Run Real Mem0 vs Capsule

```bash
cd benchmarks-suite

python3 benchmarks/agent_task_harness.py \
  --agent openai \
  --model gpt-4.1-mini \
  --adapters mem0_real,capsule \
  --families all \
  --cases-per-family 3 \
  --noise-events 80 \
  --noise-lines 10 \
  --stale-events 15 \
  --repo-files 60 \
  --repo-noise-lines 60 \
  --max-steps 15 \
  --json-out agent-task-report-real-mem0-vs-capsule.json \
  --progress
```

## Run Real LangGraph Store vs Capsule

```bash
cd benchmarks-suite

python3 benchmarks/agent_task_harness.py \
  --agent openai \
  --model gpt-4.1-mini \
  --adapters langgraph_real,capsule \
  --families all \
  --cases-per-family 3 \
  --noise-events 80 \
  --noise-lines 10 \
  --stale-events 15 \
  --repo-files 60 \
  --repo-noise-lines 60 \
  --max-steps 15 \
  --json-out agent-task-report-real-langgraph-vs-capsule.json \
  --progress
```

Important: `langgraph_real` is real LangGraph storage/checkpoint memory. By itself, it is not a full semantic memory product like Mem0/Zep/Capsule. Treat it as a real framework-state baseline, not as LangGraph plus a custom RAG system.
