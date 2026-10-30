# Repo Patch Memory Eval

        This folder contains identical broken repo workspaces for each memory
        layer. Give every agent the same `task_prompt.md`, the same
        `memory_sessions.md`, and only its own workspace path.

        Workspaces:

        - `workspaces/capsule`
- `workspaces/mem0_real_async`
- `workspaces/langgraph_real`
- `workspaces/vector_rag`
- `workspaces/bm25_rag`

        Score each run with:

        - test success: `python3 -m unittest discover -s tests`
        - stale cleanup: `python3 tools/check_contract.py`
        - touched expected files from `expected_outcomes.json`
        - total input/output tokens
        - number of turns/tool calls
        - wall-clock time

        The only variable should be the memory layer.
