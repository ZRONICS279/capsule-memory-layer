# Capsule Memory

This repository targets a real agent memory handle:

```text
@capsule
  -> lossless event ledger
  -> exact fact graph
  -> semantic routing sketch
  -> budgeted memory compiler
  -> evidence expansion
  -> optional soft vectors / prefix KV
```

The invention is not a vague summary. It is a compact operational interface to a
larger memory substrate. The agent sees only the memory needed for the current
step, while the full raw context remains stored and expandable.

## Design Principle

Do not pay to resend the whole past. Do not delete the past either.

Store losslessly, compile selectively, expand exactly.

## Runtime Shape

```text
@capsule_label
  -> lossless raw events
  -> active/superseded/conflicting fact graph
  -> query-specific compiled memory
  -> evidence handles
  -> exact expansion on demand
  -> soft capsule vectors when runtime supports them
  -> prefix cache when runtime supports it
```

The goal is to reduce repeated context tokens without turning memory into vague summaries.

## Memory Compiler

The compiler builds a micro-context under a hard token budget:

```text
MEMORY_HANDLE @greenhouse id=greenhouse v=7
CONTRACT lossless-ledger-backed; use evidence ids; say UNKNOWN if absent.
ACTIVE_FACTS:
- greenhouse.board=ESP32-C3 evidence=abc123:greenhouse:board
- greenhouse.language=MicroPython evidence=abc123:greenhouse:language
- greenhouse.pump_voltage=12V evidence=def456:greenhouse:pump_voltage
OMITTED_FACTS 41; call expand(handle or query) before guessing.
```

This is how agents get lower token use without losing memory:

- active facts are tiny
- every fact has provenance
- omitted facts are visible as omitted, not silently forgotten
- raw event text can be expanded by evidence id
- conflicts are represented instead of blurred

## Optional Latent Path

For local/inference runtimes that support it, the same capsule can also produce:

```text
long context -> encoder -> 16-128 capsule vectors -> inputs_embeds or prefix KV
```

That path is for deeper compression and speed. The memory handle/ledger/compiler
architecture still stays in place so the system remains inspectable.

## What This Prototype Does Today

- Defines a trainable soft-token encoder in `soft_tokens.py`.
- Provides a training loop that freezes an LLM and optimizes the capsule encoder.
- Compiles task-specific memory payloads under a token budget.
- Keeps evidence handles that expand back to raw events.
- Extracts exact symbolic facts for precision-sensitive data.
- Preserves version history by superseding old active choices instead of deleting them.
- Keeps a deterministic sketch/JSON path only as a phase-0 inspection harness.

## Why Hybrid

Dense vectors are good at broad similarity and task shape, but weak at exact facts like `ESP32-C3`, `pin 12`, file paths, and API versions.

The symbolic spine is the guardrail:

```text
[greenhouse] board: ESP32-C3
[greenhouse] language: MicroPython
[greenhouse] physical_pin_12: BCM GPIO 18
```

This is intentionally tiny compared with the raw transcript, but it is not fuzzy.

## Advanced Compression Track

The compatible memory compiler already reduces repeated text context for normal
agent APIs. The deeper compression track trains capsule vectors for runtimes
that accept embeddings or prefix state. That path needs:

- an encoder that maps long context to learned memory vectors
- a model trained or tuned to attend to those vectors
- a decoder/inspector that can expose capsule contents
- update training so `capsule_t + event -> capsule_t+1`

The training script implements the first version of that path:
soft-prompt capsules through `inputs_embeds`.

## Anti-Hallucination Contract

The memory layer should not invent missing facts. Querying an absent exact key returns `unknown`.

For LLM usage, inject `Capsule.render_context()` and enforce this rule:

```text
Use exact spine for names, versions, numbers, paths, pins, APIs, and decisions; answer unknown if absent.
```

## Next Research Step

Replace `VectorSketch` with a trainable encoder that emits 16-128 soft tokens. Keep the exact symbolic spine. Compare against:

- full context
- text summary
- embedding retrieval
- capsule vector + symbolic spine

Useful evaluation questions:

- What board is active?
- What was rejected?
- What changed?
- Which exact pin or version is required?
- Which facts conflict?
