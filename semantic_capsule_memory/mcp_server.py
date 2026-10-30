"""Dependency-light MCP stdio server for Capsule Memory."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os
import sys
import time
from typing import Any

from .agentic import AgenticMemoryLayer, MemoryPacket
from .budget import estimate_tokens
from .engine import EvidenceSnippet, MemoryAnswer


PROTOCOL_VERSION = "2025-06-18"


def debug_log(message: str) -> None:
    path = os.environ.get("SEMANTIC_CAPSULE_MCP_DEBUG_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{time.time():.6f} {message}\n")
    except OSError:
        pass


def default_store_path() -> Path:
    configured = os.environ.get("SEMANTIC_CAPSULE_STORE")
    if configured:
        return Path(configured)
    return Path.cwd() / ".capsules" / "mcp_memory.sqlite"


class MemoryMCPServer:
    def __init__(self, store: str | Path | None = None):
        self.memory = AgenticMemoryLayer(store or default_store_path())

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        debug_log(f"request method={method!r} id={request_id!r}")

        if request_id is None and method and method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "semantic-capsule-memory", "version": "0.1.0"},
                }
            elif method == "tools/list":
                result = {"tools": tool_definitions()}
            elif method == "tools/call":
                params = request.get("params") or {}
                result = self.call_tool(params.get("name", ""), params.get("arguments") or {})
            elif method == "ping":
                result = {}
            else:
                return error_response(request_id, -32601, f"Unknown method: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:  # MCP tool errors should be returned, not crash the server.
            return error_response(request_id, -32000, str(exc))

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "memory_remember":
            capsule = self.memory.remember(
                require_str(args, "capsule_id"),
                require_str(args, "text"),
                scope=str(args.get("scope", "global")),
                label=args.get("label"),
            )
            return tool_json(
                {
                    "capsule_id": capsule.capsule_id,
                    "version": capsule.version,
                    "events": len(capsule.events),
                    "active_facts": len(capsule.active_facts()),
                }
            )

        if name == "memory_prepare_step":
            packet = self.memory.prepare_step(
                require_str(args, "capsule_id"),
                require_str(args, "task"),
                original_context=str(args.get("original_context", "")),
                retrieve_limit=int(args.get("retrieve_limit", 2)),
            )
            return tool_json(memory_packet_to_dict(packet))

        if name == "memory_answer":
            answer = self.memory.answer(
                require_str(args, "capsule_id"),
                require_str(args, "question"),
                original_context=str(args.get("original_context", "")),
            )
            return tool_json(memory_answer_to_dict(answer))

        if name == "memory_expand":
            text = self.memory.expand(require_str(args, "capsule_id"), require_str(args, "source"))
            return tool_text(text)

        if name == "memory_retrieve":
            snippets = self.memory.engine.retrieve(
                require_str(args, "capsule_id"),
                require_str(args, "query"),
                limit=int(args.get("limit", 3)),
            )
            return tool_json({"snippets": [evidence_to_dict(snippet) for snippet in snippets]})

        if name == "memory_compact_turn":
            return tool_json(self.compact_turn(args))

        raise ValueError(f"Unknown tool: {name}")

    def compact_turn(self, args: dict[str, Any]) -> dict[str, Any]:
        capsule_id = require_str(args, "capsule_id")
        current_message = normalize_message(args.get("current_message"), "current_message")
        transcript = str(args.get("transcript_to_remember", ""))
        scope = str(args.get("scope", capsule_id))
        system_messages = normalize_messages(args.get("system_messages", []), "system_messages")
        recent_messages = normalize_messages(args.get("recent_messages", []), "recent_messages")
        max_recent_messages = int(args.get("max_recent_messages", 0))
        max_outbound_tokens = int(args.get("max_outbound_tokens", 1200))
        min_reduction = float(args.get("min_reduction", 0.60))
        retrieve_limit = int(args.get("retrieve_limit", 2))

        if len(recent_messages) > max_recent_messages:
            raise ValueError(f"recent_messages exceeds max_recent_messages: {len(recent_messages)} > {max_recent_messages}")

        if transcript:
            self.memory.remember(capsule_id, transcript, scope=scope, label=f"@{capsule_id}")

        task = current_message["content"]
        packet = self.memory.prepare_step(
            capsule_id,
            task,
            original_context=transcript,
            retrieve_limit=retrieve_limit,
        )
        memory_message = {
            "role": "system",
            "content": (
                "Use this MCP memory packet as durable context. Do not infer facts absent from it; "
                "ask for memory_expand/memory_retrieve if exact evidence is needed.\n\n"
                f"{packet.payload}"
            ),
        }
        messages = [memory_message, *system_messages, *recent_messages, current_message]
        outbound_text = json.dumps(messages, sort_keys=True)
        raw_tokens = estimate_tokens(transcript)
        outbound_tokens = estimate_tokens(outbound_text)
        reduction = 0.0 if raw_tokens == 0 else max(0.0, 1.0 - (outbound_tokens / raw_tokens))
        passed = outbound_tokens <= max_outbound_tokens and reduction >= min_reduction

        if transcript and transcript in outbound_text:
            raise ValueError("strict compaction failed: outbound messages contain the full transcript")
        if outbound_tokens > max_outbound_tokens:
            raise ValueError(f"strict compaction failed: outbound_tokens {outbound_tokens} > max_outbound_tokens {max_outbound_tokens}")
        if raw_tokens and reduction < min_reduction:
            raise ValueError(f"strict compaction failed: reduction {reduction:.4f} < min_reduction {min_reduction:.4f}")

        return {
            "messages": messages,
            "raw_tokens": raw_tokens,
            "outbound_tokens": outbound_tokens,
            "reduction": reduction,
            "min_reduction": min_reduction,
            "max_outbound_tokens": max_outbound_tokens,
            "passed": passed,
            "capsule_id": capsule_id,
            "memory_packet": memory_packet_to_dict(packet),
        }


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "memory_remember",
            "description": "Ingest lossless agent memory into a capsule.",
            "inputSchema": {
                "type": "object",
                "required": ["capsule_id", "text"],
                "properties": {
                    "capsule_id": {"type": "string"},
                    "text": {"type": "string"},
                    "scope": {"type": "string", "default": "global"},
                    "label": {"type": "string"},
                },
            },
        },
        {
            "name": "memory_prepare_step",
            "description": "Compile a small task-specific memory packet with token reduction metrics.",
            "inputSchema": {
                "type": "object",
                "required": ["capsule_id", "task"],
                "properties": {
                    "capsule_id": {"type": "string"},
                    "task": {"type": "string"},
                    "original_context": {"type": "string"},
                    "retrieve_limit": {"type": "integer", "default": 2},
                },
            },
        },
        {
            "name": "memory_answer",
            "description": "Answer a question from grounded memory or return UNKNOWN/CONFLICT.",
            "inputSchema": {
                "type": "object",
                "required": ["capsule_id", "question"],
                "properties": {
                    "capsule_id": {"type": "string"},
                    "question": {"type": "string"},
                    "original_context": {"type": "string"},
                },
            },
        },
        {
            "name": "memory_expand",
            "description": "Expand an exact raw evidence chunk by source id.",
            "inputSchema": {
                "type": "object",
                "required": ["capsule_id", "source"],
                "properties": {"capsule_id": {"type": "string"}, "source": {"type": "string"}},
            },
        },
        {
            "name": "memory_retrieve",
            "description": "Retrieve exact raw snippets for a query without loading the full ledger.",
            "inputSchema": {
                "type": "object",
                "required": ["capsule_id", "query"],
                "properties": {
                    "capsule_id": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 3},
                },
            },
        },
        {
            "name": "memory_compact_turn",
            "description": "Store a full turn transcript and return a strict compact outbound message list.",
            "inputSchema": {
                "type": "object",
                "required": ["capsule_id", "current_message"],
                "properties": {
                    "capsule_id": {"type": "string"},
                    "current_message": {
                        "type": "object",
                        "required": ["role", "content"],
                        "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                    },
                    "transcript_to_remember": {"type": "string"},
                    "scope": {"type": "string"},
                    "system_messages": {"type": "array"},
                    "recent_messages": {"type": "array"},
                    "max_recent_messages": {"type": "integer", "default": 0},
                    "max_outbound_tokens": {"type": "integer", "default": 1200},
                    "min_reduction": {"type": "number", "default": 0.60},
                    "retrieve_limit": {"type": "integer", "default": 2},
                },
            },
        },
    ]


def memory_packet_to_dict(packet: MemoryPacket) -> dict[str, Any]:
    return {
        "capsule_id": packet.capsule_id,
        "task": packet.task,
        "payload": packet.payload,
        "raw_tokens": packet.raw_tokens,
        "memory_tokens": packet.memory_tokens,
        "reduction": packet.reduction,
        "target_reduction": packet.target_reduction,
        "passed_target": packet.passed_target,
        "evidence_refs": packet.evidence_refs,
        "retrieved": [evidence_to_dict(snippet) for snippet in packet.retrieved],
    }


def memory_answer_to_dict(answer: MemoryAnswer) -> dict[str, Any]:
    return {
        "status": answer.status,
        "answer": answer.answer,
        "confidence": answer.confidence,
        "capsule_id": answer.capsule_id,
        "evidence_sources": list(answer.evidence_sources),
        "compiled_payload": answer.compiled.payload,
        "compiled_tokens": answer.compiled.token_estimate,
        "compression_ratio": answer.compiled.compression_ratio,
        "retrieved": [evidence_to_dict(snippet) for snippet in answer.retrieved],
    }


def evidence_to_dict(snippet: EvidenceSnippet) -> dict[str, Any]:
    return asdict(snippet)


def tool_json(data: dict[str, Any]) -> dict[str, Any]:
    return tool_text(json.dumps(data, indent=2, sort_keys=True))


def tool_text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required string argument: {key}")
    return value


def normalize_message(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a message object")
    role = value.get("role")
    content = value.get("content")
    if not isinstance(role, str) or not isinstance(content, str) or not content:
        raise ValueError(f"{field} must include string role and content")
    return {"role": role, "content": content}


def normalize_messages(value: Any, field: str) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [normalize_message(item, f"{field}[{index}]") for index, item in enumerate(value)]


def error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def read_message(stream) -> tuple[dict[str, Any], str] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if line == b"":
            return None
        raw_line = line.decode("utf-8").strip()
        if raw_line.startswith("{"):
            return json.loads(raw_line), "jsonl"
        line = raw_line
        if not line:
            break
        key, _, value = line.partition(":")
        headers[key.lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = stream.read(length)
    return json.loads(body.decode("utf-8")), "content-length"


def write_message(stream, message: dict[str, Any], framing: str = "content-length") -> None:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if framing == "jsonl":
        stream.write(body + b"\n")
        stream.flush()
        return
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stream.write(body)
    stream.flush()


def serve_stdio(server: MemoryMCPServer | None = None) -> None:
    debug_log("server starting")
    server = server or MemoryMCPServer()
    debug_log("server ready")
    while True:
        message = read_message(sys.stdin.buffer)
        if message is None:
            debug_log("stdin closed")
            return
        request, framing = message
        debug_log(f"received framing={framing}")
        response = server.handle(request)
        if response is not None:
            debug_log(f"response id={response.get('id')!r} keys={list(response.keys())!r}")
            write_message(sys.stdout.buffer, response, framing)


def main() -> int:
    serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
