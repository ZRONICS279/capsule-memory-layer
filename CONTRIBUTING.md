# Contributing

Capsule Memory is trying to solve a hard problem honestly: lower-token
agent memory without hiding lossy summaries behind marketing language.

Good contributions include:

- better exact-fact extraction with tests
- storage backends
- agent framework adapters
- soft-token training improvements
- evaluation datasets with corrections, contradictions, versions, file paths, pins, and API contracts
- benchmark reports comparing full context, summaries, RAG, text spine, and soft capsules

Rules for changes:

- Do not claim normal text tokens can carry arbitrary hidden memory.
- Add eval cases for every new extraction rule.
- Unknown is better than hallucinated.
- Keep memory inspectable.

Run:

```bash
python3 -m unittest discover -s tests
python3 training/run_eval.py --cases evals/symbolic_cases.jsonl
```
