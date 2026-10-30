# Agentic Memory Benchmark Scorecard

Agent model: gpt-5.4-mini for agentic coding rows; retrieval-only rows use memory retrieval only

Each cell shows score, recall, CtxOK, stale leakage, token savings, prepared context tokens, ingest speed, and task success when available.

Score is computed from required-fact recall after leakage penalties. In coding runs: `Score = Recall x (1 - Stale leakage)`. A 100% score means every required current fact was retrieved with 0% stale leakage.

CtxOK means all required current facts appeared in context and no stale facts leaked. MemOK means task success plus CtxOK.

| Benchmark | Capsule | mem0 async | BM25 RAG | LangGraph | Vector RAG | No memory |
|---|---|---|---|---|---|---|
| Long-horizon retrieval | Score 0.0%<br>Recall 100.0%<br>CtxOK 0.0%<br>Stale 100.0%<br>Save 90.3%<br>CtxTok 536<br>RawTok 5.5k<br>Ingest 0.1s | - | - | - | - | - |
| Long-horizon agentic | Score 0.0%<br>Recall 100.0%<br>CtxOK 0.0%<br>Stale 100.0%<br>Save 90.1%<br>CtxTok 536<br>RawTok 5.4k<br>Ingest 0.1s<br>Success 100.0%<br>MemOK 0.0% | - | - | - | - | - |
| Full retrieval | Score 13.3%<br>Recall 100.0%<br>CtxOK 10.0%<br>Stale 86.7%<br>Save 31.3%<br>CtxTok 288<br>RawTok 868<br>Ingest <0.1s | - | - | - | - | - |
| Full retrieval + baselines | - | - | - | - | - | - |
| Full agentic coding | Score 13.3%<br>Recall 100.0%<br>CtxOK 10.0%<br>Stale 86.7%<br>Save 29.7%<br>CtxTok 287<br>RawTok 868<br>Ingest <0.1s<br>Success 78.3%<br>MemOK 10.0% | - | - | - | - | - |
| Persistent two-phase | - | - | - | - | - | - |
| General chatbot memory | Score 50.0%<br>Recall 84.4%<br>CtxOK 50.0%<br>Stale 25.0%<br>Save 25.4%<br>CtxTok 135<br>RawTok 164<br>Ingest <0.1s | - | - | - | - | - |
