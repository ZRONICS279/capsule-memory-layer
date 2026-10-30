# Agentic Memory Benchmark Scorecard

Agent model: gpt-4.1-mini for agentic coding rows; retrieval-only rows use memory retrieval only

Each cell shows score, recall, CtxOK, stale leakage, token savings, prepared context tokens, ingest speed, and task success when available.

Score is computed from required-fact recall after leakage penalties. In coding runs: `Score = Recall x (1 - Stale leakage)`. A 100% score means every required current fact was retrieved with 0% stale leakage.

CtxOK means all required current facts appeared in context and no stale facts leaked. MemOK means task success plus CtxOK.

| Benchmark | Capsule | mem0 async | BM25 RAG | LangGraph | Vector RAG | No memory |
|---|---|---|---|---|---|---|
| Long-horizon retrieval | Score 100.0%<br>Recall 100.0%<br>CtxOK 100.0%<br>Stale 0.0%<br>Save 96.1%<br>CtxTok 218<br>RawTok 5.5k<br>Ingest 0.1s | - | - | - | - | - |
| Long-horizon agentic | - | - | - | - | - | - |
| Full retrieval | Score 100.0%<br>Recall 100.0%<br>CtxOK 100.0%<br>Stale 0.0%<br>Save 62.9%<br>CtxTok 138<br>RawTok 868<br>Ingest <0.1s | - | - | - | - | - |
| Full retrieval + baselines | - | - | - | - | - | - |
| Full agentic coding | - | - | - | - | - | - |
| Persistent two-phase | - | - | - | - | - | - |
| General chatbot memory | Score 50.0%<br>Recall 84.4%<br>CtxOK 50.0%<br>Stale 25.0%<br>Save 27.8%<br>CtxTok 122<br>RawTok 164<br>Ingest <0.1s | - | - | - | - | - |
