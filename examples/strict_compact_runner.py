"""Tiny agent-loop proof: send compact memory + current turn, never history."""

from pathlib import Path
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def frame(message: dict) -> bytes:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def read_frame(stream) -> dict:
    headers = {}
    while True:
        line = stream.readline()
        if not line:
            raise RuntimeError("server closed stdout")
        line = line.decode("ascii").strip()
        if not line:
            break
        key, _, value = line.partition(":")
        headers[key.lower()] = value.strip()
    return json.loads(stream.read(int(headers["content-length"])).decode("utf-8"))


def call(proc, message: dict) -> dict:
    proc.stdin.write(frame(message))
    proc.stdin.flush()
    return read_frame(proc.stdout)


def tool_payload(response: dict) -> dict:
    return json.loads(response["result"]["content"][0]["text"])


def main() -> int:
    transcript = (
        "Old transcript turn with verbose tool logs and analysis that should not be replayed. " * 600
        + "Final decision: use ESP32-C3 with MicroPython. Pump voltage: 12V. Do not use Arduino C++."
    )
    current = {"role": "user", "content": "What pump voltage should I use?"}

    with TemporaryDirectory() as directory:
        env = dict(os.environ)
        env["SEMANTIC_CAPSULE_STORE"] = str(Path(directory) / "memory.sqlite")
        proc = subprocess.Popen(
            [sys.executable, "-m", "semantic_capsule_memory.mcp_server"],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            call(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            response = call(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_compact_turn",
                        "arguments": {
                            "capsule_id": "demo_thread",
                            "scope": "conversation",
                            "transcript_to_remember": transcript,
                            "current_message": current,
                            "system_messages": [
                                {"role": "system", "content": "Use MCP memory. Say UNKNOWN if absent."}
                            ],
                            "max_recent_messages": 0,
                            "max_outbound_tokens": 900,
                            "min_reduction": 0.60,
                            "retrieve_limit": 1,
                        },
                    },
                },
            )
            data = tool_payload(response)
            outbound = json.dumps(data["messages"], sort_keys=True)
            assert transcript not in outbound
            assert len(data["messages"]) == 3
            assert data["passed"]

            print(json.dumps({
                "raw_tokens": data["raw_tokens"],
                "outbound_tokens": data["outbound_tokens"],
                "reduction": round(data["reduction"], 4),
                "messages_sent_to_model": data["messages"],
            }, indent=2))
        finally:
            proc.terminate()
            proc.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
