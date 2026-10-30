# Capsule Memory

A research prototype for a real lower-token agent memory layer.

The target is not ordinary RAG and not a text summary. The target is an
agent-usable memory handle:

```text
@project_memory
  -> lossless event ledger
  -> exact fact graph
  -> budgeted memory compiler
  -> evidence expansion
  -> optional soft capsule vectors / prefix cache
```

The handle is tiny. The stored memory is not. Agents use the handle to receive a
compiled micro-context for the current task, with evidence ids that can expand
back to exact raw events when precision is needed.

This is the practical route to 75-90% lower repeated prompt tokens without
throwing memory away.

## What Is Included

- `semantic_capsule_memory/soft_tokens.py`: trainable capsule encoder and LLM embedding injector
- `training/train_soft_capsule.py`: training loop for frozen causal LMs using `inputs_embeds`
- `semantic_capsule_memory/compiler.py`: budgeted memory compiler with evidence handles
- `semantic_capsule_memory/sdk.py`: public client API for agents/apps
- `semantic_capsule_memory/runtime.py`: text, soft-embedding, and prefix-cache runtime modes
- `semantic_capsule_memory/adapters.py`: chat/agent message injection middleware
- `semantic_capsule_memory/eval.py`: compression and precision eval harness
- `semantic_capsule_memory/storage.py`: JSON and SQLite storage backends
- `semantic_capsule_memory/capsule.py`: exact symbolic spine and versioned inspection layer
- `examples/synthetic_capsule_records.jsonl`: tiny starter dataset format

The symbolic spine stays because dense vectors alone blur exact things like
`ESP32-C3` vs `ESP32`, versions, pins, file paths, and API contracts.

## Quick Start

```bash
python -m semantic_capsule_memory.cli --store .capsules/store.json create greenhouse --label @greenhouse
python -m semantic_capsule_memory.cli --store .capsules/store.json update greenhouse "For the greenhouse project, use ESP32-C3 with MicroPython. Pin 12 = BCM GPIO 18. Do not use Arduino C++." --scope greenhouse
python -m semantic_capsule_memory.cli --store .capsules/store.json context greenhouse
python -m semantic_capsule_memory.cli --store .capsules/store.json query greenhouse board --scope greenhouse
```

## MCP Quick Start

Install the package in your agent environment, then run the stdio MCP server:

```bash
scm-mcp
```

For a source checkout, the equivalent module form is:

```bash
python3 -m semantic_capsule_memory.mcp_server
```

The server exposes these tools: `memory_remember`, `memory_prepare_step`,
`memory_answer`, `memory_expand`, `memory_retrieve`, and
`memory_compact_turn`.

The MCP server is tuned for coding agents by default: prepared memory uses the
compact prompt codec, repo-shaped tasks can autoscale within a small cap, and
code-edit memories may include a file-scoped patch checklist plus stale values
to avoid. Set `autoscale=False` or pass explicit caps in SDK usage when a
non-coding workload needs stricter prompt size control.

Point Codex at a durable local capsule store by setting
`SEMANTIC_CAPSULE_STORE` in the MCP server config:

```json
{
  "mcpServers": {
    "semantic_capsule_memory": {
      "command": "scm-mcp",
      "env": {
        "SEMANTIC_CAPSULE_STORE": "/absolute/path/.capsules/codex_memory.sqlite"
      }
    }
  }
}
```

If `SEMANTIC_CAPSULE_STORE` is omitted, the server stores capsules at
`.capsules/mcp_memory.sqlite` under its working directory. See
[docs/mcp-setup.md](docs/mcp-setup.md)
for capsule-id naming guidance, the full setup example, smoke test, and strict
compact-turn proof.

## SDK Usage

```python
from semantic_capsule_memory import CapsuleMemoryClient, MemoryWrite

memory = CapsuleMemoryClient(".capsules/store.json")
memory.remember(MemoryWrite(
    capsule_id="greenhouse",
    scope="greenhouse",
    text="Use ESP32-C3 with MicroPython. Pin 12 = BCM GPIO 18.",
))

prepared = memory.prepare("greenhouse")
agent_prompt_memory = prepared.text_context
```

Use SQLite for durable local deployments:

```python
memory = CapsuleMemoryClient(".capsules/store.sqlite")
```

Text-only mode works with any LLM today by injecting only the compact symbolic spine.
The memory compiler keeps that payload task-specific and budgeted. Soft-token
mode is the deeper runtime path for local models that support embedding or prefix
cache injection.

## Agent Adapter

```python
from semantic_capsule_memory import AgentMemoryMiddleware

middleware = AgentMemoryMiddleware(memory)
messages, metadata = middleware.prepare_chat(
    "greenhouse",
    [{"role": "user", "content": "What board should I target?"}],
)
```

`messages` can be sent to any OpenAI-compatible/chat-style model API in text
mode. For soft-token mode, `metadata["requires_embedding_runtime"]` tells the
caller to use an embedding-capable local runtime.

## Memory Compiler

```python
compiled = memory.compile(
    "greenhouse",
    query="Which pump voltage should I use?",
    original_context=long_session_text,
    token_budget=192,
)

print(compiled.payload)
print(compiled.compression_ratio)
```

The compiler does not summarize the full session. It selects the smallest set of
active facts relevant to the task, attaches evidence ids, and leaves the raw
ledger available through `expand_evidence()`.

## Train Soft Capsules

Requires `torch` and `transformers`:

```bash
python training/train_soft_capsule.py \
  --model /path/to/local/causal-lm \
  --data examples/synthetic_capsule_records.jsonl \
  --output runs/greenhouse_capsule \
  --capsule-tokens 32
```

## Run Evals

```bash
python training/run_eval.py --cases evals/symbolic_cases.jsonl
```

## Design

See [docs/architecture.md](docs/architecture.md).
