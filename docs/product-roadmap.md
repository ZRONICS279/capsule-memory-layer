# Product Roadmap

## Product Claim

Capsule Memory reduces repeated agent context by turning prior state into a durable capsule:

- text-only agents get a compact exact symbolic spine today
- embedding-capable agents get soft capsule tokens
- future runtimes can inject prefix KV cache or capsule-aware cross-attention

## Open-Source MVP

The first downloadable version should be useful without custom model training:

- Python SDK
- JSONL/local file store
- exact symbolic spine
- capsule inspection
- token compression evals
- adapters for common agent frameworks

The MVP must not claim lossless one-token memory for normal text-only APIs. It should say:

```text
Text-only mode shrinks context by extracting exact state.
Soft-token mode is experimental and requires model/runtime embedding access.
```

## Company Version

Companies need:

- pluggable storage: local JSON, SQLite, Postgres
- audit logs and version history
- deterministic eval reports
- PII controls and redaction hooks
- deployment docs
- framework adapters
- model/runtime adapters for vLLM, llama.cpp, Transformers, and custom inference

## Technical Milestones

1. Stabilize SDK and storage API.
2. Add SQLite storage.
3. Add exact-fact extraction plugins.
4. Add LangChain, LlamaIndex, AutoGen, and OpenAI Agents style adapters.
5. Build benchmark dataset: corrections, versions, code facts, project state, contradictions.
6. Run text-spine baseline.
7. Run soft-token training against a small local open model.
8. Compare full context vs summary vs RAG vs capsule.
9. Add prefix-cache runtime for an inference engine that supports it.
10. Publish package and docs.

## Success Metrics

- 10x to 100x fewer repeated input tokens in text-only mode for state-heavy agents.
- 32 to 128 soft tokens for long memory in embedding mode.
- 95%+ exact-fact accuracy on correction/version/pin/path evals.
- Unknown instead of hallucinated answer when a fact is absent.
- Inspectable memory state for every answer.
