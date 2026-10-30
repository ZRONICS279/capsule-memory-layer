# MCP Setup

Run the server:

```bash
cd .
python3 -m semantic_capsule_memory.mcp_server
```

Useful environment variable:

```bash
SEMANTIC_CAPSULE_STORE=/absolute/path/memory.sqlite
```

Example MCP client config:

```json
{
  "mcpServers": {
    "semantic-capsule-memory": {
      "command": "python3",
      "args": ["-m", "semantic_capsule_memory.mcp_server"],
      "cwd": ".",
      "env": {
        "SEMANTIC_CAPSULE_STORE": ".capsules/codex_memory.sqlite"
      }
    }
  }
}
```

Tools exposed:

- `memory_remember`
- `memory_prepare_step`
- `memory_answer`
- `memory_expand`
- `memory_retrieve`
- `memory_compact_turn`

The default server configuration is optimized for coding agents. Memory packets
use compact prompt formatting, repo/code-edit tasks can autoscale within a
bounded budget, and patch-shaped memory is rendered as a per-file checklist with
stale values separated under `DO_NOT_USE`.

## Capsule IDs for agents

Every MCP memory tool takes a `capsule_id`. Agents should choose it
automatically before the first memory call, then reuse the same id for the same
durable context.

Use a short, stable, searchable name:

- Project workspace: repo or directory slug, for example `semantic-capsule-memory`
- Codex thread or task: repo slug plus purpose, for example
  `semantic-capsule-memory-readme-mcp`
- User-level memory: stable user or account slug, for example `user-zcs`

Normalize ids to lowercase ASCII with words separated by hyphens. Avoid random
ids unless the capsule is intentionally disposable. Use the same value for
`scope` when the memory belongs to one project or thread, and use labels like
`@semantic-capsule-memory-readme-mcp` for human-readable handles.

Recommended agent flow:

1. Derive `capsule_id` from the current repo, workspace, thread title, or task.
2. Call `memory_prepare_step` with that id at the start of work.
3. If the server returns `Unknown capsule`, create it implicitly by calling
   `memory_remember` with the same `capsule_id`, the initial task context,
   `scope`, and `label`.
4. Reuse that exact id for later `memory_prepare_step`, `memory_answer`,
   `memory_retrieve`, and `memory_expand` calls.

The MCP server stores capsules in the database configured by
`SEMANTIC_CAPSULE_STORE`, so the easiest way to locate capsules later is to keep
the store path stable and use deterministic ids.

Smoke test:

```bash
python3 examples/mcp_smoke_test.py
```

Strict compact-turn proof:

```bash
python3 examples/strict_compact_runner.py
```

This runner stores a full transcript in MCP, receives a compact outbound message
list, asserts the full transcript is absent from that outbound request, and
prints raw token count vs outbound token count.
