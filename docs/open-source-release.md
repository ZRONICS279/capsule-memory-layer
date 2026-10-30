# Open-Source Release Plan

## Package

Name candidate: `semantic-capsule-memory`

Install:

```bash
pip install semantic-capsule-memory
```

Core usage:

```python
from semantic_capsule_memory.sdk import CapsuleMemoryClient, MemoryWrite

memory = CapsuleMemoryClient(".capsules/store.json")
memory.remember(MemoryWrite(
    capsule_id="greenhouse",
    scope="greenhouse",
    text="Use ESP32-C3 with MicroPython. Pin 12 = BCM GPIO 18.",
))

prepared = memory.prepare("greenhouse")
print(prepared.text_context)
```

## README Promise

Be precise:

- compatible mode works today with any LLM because it injects compact text
- soft-token mode is the path to true non-text capsules
- normal API text tokens cannot carry hidden arbitrary memory

## Repository Checklist

- license
- contributing guide
- security policy
- examples
- eval report command
- storage backends
- framework adapters
- CI tests
- published docs

