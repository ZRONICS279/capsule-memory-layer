# Agentic Memory Benchmark Scorecard

Agent model: gpt-5.4-mini for agentic coding rows; retrieval-only rows use memory retrieval only

Each cell shows score, recall, CtxOK, stale leakage, token savings, prepared context tokens, ingest speed, and task success when available.

Score is computed from required-fact recall after leakage penalties. In coding runs: `Score = Recall x (1 - Stale leakage)`. A 100% score means every required current fact was retrieved with 0% stale leakage.

CtxOK means all required current facts appeared in context and no stale facts leaked. MemOK means task success plus CtxOK.

| Benchmark | Capsule | mem0 async | BM25 RAG | LangGraph | Vector RAG | No memory |
|---|---|---|---|---|---|---|
| Long-horizon retrieval | Score 100.0%<br>Recall 100.0%<br>CtxOK 100.0%<br>Stale 0.0%<br>Save 95.2%<br>CtxTok 266<br>RawTok 5.5k<br>Ingest 0.1s | - | - | - | - | - |
| Long-horizon agentic | Score 100.0%<br>Recall 100.0%<br>CtxOK 100.0%<br>Stale 0.0%<br>Save 95.0%<br>CtxTok 266<br>RawTok 5.4k<br>Ingest 0.2s<br>Success 75.0%<br>MemOK 75.0% | - | - | - | - | - |
| Full retrieval | Score 100.0%<br>Recall 100.0%<br>CtxOK 100.0%<br>Stale 0.0%<br>Save 74.8%<br>CtxTok 100<br>RawTok 868<br>Ingest <0.1s | - | - | - | - | - |
| Full retrieval + baselines | - | - | - | - | - | - |
| Full agentic coding | Score 100.0%<br>Recall 100.0%<br>CtxOK 100.0%<br>Stale 0.0%<br>Save 74.1%<br>CtxTok 99<br>RawTok 868<br>Ingest <0.1s<br>Success 75.0%<br>MemOK 75.0% | - | - | - | - | - |
| Persistent two-phase | - | - | - | - | - | - |
| General chatbot memory | Score 45.8%<br>Recall 80.2%<br>CtxOK 37.5%<br>Stale 25.0%<br>Save 28.3%<br>CtxTok 121<br>RawTok 164<br>Ingest <0.1s | - | - | - | - | - |
