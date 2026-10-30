from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_capsule_memory import AgentMemoryMiddleware, CapsuleMemoryClient


store = ROOT / ".capsules" / "compact_agent_loop.sqlite"
if store.exists():
    store.unlink()

memory = CapsuleMemoryClient(store)
middleware = AgentMemoryMiddleware(memory)

full_transcript = (
    "Verbose prior user/assistant/tool transcript that should not be replayed to the model. " * 300
    + "Final decision: use ESP32-C3 with MicroPython. Pump voltage: 12V."
)

messages, metadata = middleware.prepare_compact_chat(
    "greenhouse",
    {"role": "user", "content": "What pump voltage should I use?"},
    system_messages=[{"role": "system", "content": "Use durable memory. Say UNKNOWN if absent."}],
    transcript_to_remember=full_transcript,
    scope="greenhouse",
)

print(metadata)
print("compact_audit:", metadata["compact_audit"])
print("outbound_messages:")
for message in messages:
    print(f"- {message['role']}: {message['content'][:220]}")
