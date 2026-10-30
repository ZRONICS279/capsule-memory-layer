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
    body = stream.read(int(headers["content-length"]))
    return json.loads(body.decode("utf-8"))


def call(proc, message: dict) -> dict:
    proc.stdin.write(frame(message))
    proc.stdin.flush()
    return read_frame(proc.stdout)


def call_jsonl(proc, message: dict) -> dict:
    proc.stdin.write(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("server closed stdout")
    return json.loads(line.decode("utf-8"))


def main() -> int:
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
            print(call(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
            print(call(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}))
            print(call(proc, {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "memory_remember",
                    "arguments": {
                        "capsule_id": "greenhouse",
                        "scope": "greenhouse",
                        "text": "Use ESP32-C3 with MicroPython. Pump voltage: 12V. Do not use Arduino C++.",
                    },
                },
            }))
            print(call(proc, {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "memory_answer",
                    "arguments": {
                        "capsule_id": "greenhouse",
                        "question": "What pump voltage should I use?",
                    },
                },
            }))
        finally:
            proc.terminate()
            proc.wait(timeout=5)
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
            print(call_jsonl(proc, {"jsonrpc": "2.0", "id": 5, "method": "initialize", "params": {}}))
            print(call_jsonl(proc, {"jsonrpc": "2.0", "id": 6, "method": "tools/list", "params": {}}))
        finally:
            proc.terminate()
            proc.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
