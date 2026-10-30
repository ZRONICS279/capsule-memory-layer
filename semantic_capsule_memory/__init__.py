"""Capsule Memory prototype."""

from .budget import count_tokens, estimate_tokens, set_default_encoding, tokenizer_backend
from .capsule import Capsule, CapsuleManager, Fact, QueryResult
from .compiler import CompiledMemory, EvidenceHandle, MemoryCompiler
from .engine import EvidenceSnippet, GroundedMemoryEngine, MemoryAnswer
from .prompt_codec import EncodedPrompt, compact_memory_prompt, encode_prompt
from .runtime import PreparedMemory, RuntimeMode, SoftEmbeddingRuntime, TextSpineRuntime
from .sdk import CapsuleMemoryClient, MemoryWrite
from .adapters import AgentMemoryMiddleware, CompactChatAudit, audit_compact_chat, inject_text_memory
from .agentic import AgenticMemoryLayer, MemoryPacket
from .symbolic import Relation

__all__ = [
    "Capsule",
    "CapsuleManager",
    "CapsuleMemoryClient",
    "CompiledMemory",
    "EvidenceHandle",
    "EvidenceSnippet",
    "EncodedPrompt",
    "Fact",
    "GroundedMemoryEngine",
    "AgentMemoryMiddleware",
    "CompactChatAudit",
    "AgenticMemoryLayer",
    "MemoryAnswer",
    "MemoryPacket",
    "MemoryWrite",
    "MemoryCompiler",
    "PreparedMemory",
    "QueryResult",
    "Relation",
    "RuntimeMode",
    "SoftEmbeddingRuntime",
    "TextSpineRuntime",
    "audit_compact_chat",
    "compact_memory_prompt",
    "encode_prompt",
    "inject_text_memory",
    "count_tokens",
    "estimate_tokens",
    "set_default_encoding",
    "tokenizer_backend",
]
