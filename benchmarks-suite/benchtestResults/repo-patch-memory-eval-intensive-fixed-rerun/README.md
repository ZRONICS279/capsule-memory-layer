# Repo Patch Memory Eval

        Profile: `intensive`

        This folder contains identical broken repo workspaces for each memory
        layer. Give every agent the same `task_prompt.md`, the same
        `memory_sessions.md`, and only its own workspace path.

        Workspaces:

        - `workspaces/capsule`
- `workspaces/mem0_real_async`
- `workspaces/langgraph_real`
- `workspaces/vector_rag`
- `workspaces/bm25_rag`

        Score each run with test success, stale cleanup, expected file touches,
        total input/output tokens, number of turns/tool calls, wall-clock time,
        and diff size.
