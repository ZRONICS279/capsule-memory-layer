from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_capsule_memory import CapsuleMemoryClient, MemoryWrite
from semantic_capsule_memory.adapters import AgentMemoryMiddleware


store = ROOT / ".capsules" / "example.sqlite"
if store.exists():
    store.unlink()

memory = CapsuleMemoryClient(store)
memory.remember(
    MemoryWrite(
        capsule_id="greenhouse",
        scope="greenhouse",
        text="Use ESP32-C3 with MicroPython. Pin 12 = BCM GPIO 18. Do not use Arduino C++.",
    )
)

middleware = AgentMemoryMiddleware(memory)
messages, metadata = middleware.prepare_chat(
    "greenhouse",
    [{"role": "user", "content": "What board should my project target?"}],
    original_context="Use ESP32-C3 with MicroPython. " * 1000,
)

print(metadata)
print(messages[0]["content"])
