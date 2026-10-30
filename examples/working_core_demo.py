from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_capsule_memory.engine import GroundedMemoryEngine


def main() -> int:
    with TemporaryDirectory() as directory:
        engine = GroundedMemoryEngine(Path(directory) / "memory.sqlite", token_budget=160)
        long_noise = "Daily discussion, logs, tool output, and irrelevant implementation chatter. " * 500
        important = (
            "Final greenhouse decision: use ESP32-C3 with MicroPython. "
            "Pin 12 = BCM GPIO 18. Pump voltage: 12V. Pump current: 2A. "
            "Do not use Arduino C++."
        )
        full_context = long_noise + important

        engine.ingest("greenhouse", full_context, scope="greenhouse", label="@greenhouse")

        for question in [
            "What board should the greenhouse target?",
            "What pump voltage should I use?",
            "What should I avoid using?",
            "What enclosure color did we pick?",
        ]:
            answer = engine.answer("greenhouse", question, original_context=full_context)
            print({"question": question, "status": answer.status, "answer": answer.answer})
            print(engine.stats("greenhouse", compiled=answer.compiled))
            if answer.evidence_sources:
                print("evidence:", engine.expand("greenhouse", answer.evidence_sources[0])[-180:])
            else:
                print("retrieved:", [snippet.text for snippet in engine.retrieve("greenhouse", question)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
