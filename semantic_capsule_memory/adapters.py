"""Agent framework adapters.

The adapters avoid framework lock-in by using simple message dictionaries:
`{"role": "system"|"user"|"assistant", "content": "..."}`.
Most chat/agent frameworks can convert to and from this shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .budget import estimate_tokens
from .runtime import PreparedMemory, RuntimeMode
from .sdk import CapsuleMemoryClient


Message = dict[str, str]


@dataclass(frozen=True)
class CompactChatAudit:
    outbound_messages: int
    outbound_tokens: int
    remembered_tokens: int
    estimated_memory_tokens: int
    tokens_saved: int
    reduction: float
    full_transcript_leaked: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "outbound_messages": self.outbound_messages,
            "outbound_tokens": self.outbound_tokens,
            "remembered_tokens": self.remembered_tokens,
            "estimated_memory_tokens": self.estimated_memory_tokens,
            "tokens_saved": self.tokens_saved,
            "reduction": self.reduction,
            "full_transcript_leaked": self.full_transcript_leaked,
        }


def inject_text_memory(messages: list[Message], prepared: PreparedMemory) -> list[Message]:
    """Prepend compact memory to chat messages for text-only LLM APIs."""

    if prepared.mode != RuntimeMode.TEXT_SPINE:
        raise ValueError(f"inject_text_memory only supports text mode, got {prepared.mode}")
    if not prepared.text_context:
        return list(messages)

    memory_message = {
        "role": "system",
        "content": (
            "Agent memory capsule follows. Treat it as durable state, but do not "
            "invent facts that are absent from the exact spine.\n\n"
            f"{prepared.text_context}"
        ),
    }
    return [memory_message, *messages]


@dataclass
class AgentMemoryMiddleware:
    """Small integration object for agent loops."""

    memory: CapsuleMemoryClient

    def prepare_chat(
        self,
        capsule_id: str,
        messages: list[Message],
        *,
        original_context: str = "",
        query: str = "",
    ) -> tuple[list[Message], dict[str, Any]]:
        if not query and messages:
            query = messages[-1].get("content", "")
        prepared = self.memory.prepare(capsule_id, original_context=original_context, query=query)
        metadata = {
            "capsule_id": prepared.capsule_id,
            "capsule_version": prepared.version,
            "memory_mode": prepared.mode.value,
            "estimated_memory_tokens": prepared.estimated_text_tokens,
            "estimated_original_tokens": prepared.estimated_original_tokens,
            "compression_ratio": prepared.compression_ratio,
        }

        if prepared.mode == RuntimeMode.TEXT_SPINE:
            return inject_text_memory(messages, prepared), metadata

        metadata["soft_token_count"] = prepared.soft_token_count
        metadata["requires_embedding_runtime"] = True
        return list(messages), metadata

    def prepare_compact_chat(
        self,
        capsule_id: str,
        current_message: Message,
        *,
        system_messages: list[Message] | None = None,
        recent_messages: list[Message] | None = None,
        transcript_to_remember: str = "",
        scope: str = "global",
        original_context: str = "",
        query: str = "",
        max_recent_messages: int = 4,
        max_outbound_tokens: int | None = None,
        require_no_transcript_leak: bool = True,
    ) -> tuple[list[Message], dict[str, Any]]:
        """Prepare a chat request without replaying the full transcript.

        Store the full turn transcript in the capsule, then send only compact
        memory plus optional system/recent messages and the current user turn.
        """

        if max_recent_messages < 0:
            raise ValueError("max_recent_messages must be non-negative")

        if transcript_to_remember:
            from .sdk import MemoryWrite

            self.memory.remember(
                MemoryWrite(
                    capsule_id=capsule_id,
                    scope=scope,
                    text=transcript_to_remember,
                    label=f"@{capsule_id}",
                )
            )

        recent = list(recent_messages or [])[-max_recent_messages:]
        outbound = [
            *(system_messages or []),
            *recent,
            current_message,
        ]
        if not query:
            query = current_message.get("content", "")

        messages, metadata = self.prepare_chat(
            capsule_id,
            outbound,
            original_context=original_context or transcript_to_remember,
            query=query,
        )
        audit = audit_compact_chat(
            messages,
            remembered_context=original_context or transcript_to_remember,
            estimated_memory_tokens=int(metadata.get("estimated_memory_tokens") or 0),
        )

        if require_no_transcript_leak and audit.full_transcript_leaked:
            raise ValueError("full transcript leaked into compact outbound messages")
        if max_outbound_tokens is not None and audit.outbound_tokens > max_outbound_tokens:
            raise ValueError(
                f"compact outbound token estimate {audit.outbound_tokens} exceeds limit {max_outbound_tokens}"
            )

        metadata["compact_audit"] = audit.to_json()
        return messages, metadata


def audit_compact_chat(
    messages: list[Message],
    *,
    remembered_context: str = "",
    estimated_memory_tokens: int = 0,
) -> CompactChatAudit:
    outbound_text = "\n".join(message.get("content", "") for message in messages)
    outbound_tokens = estimate_tokens(outbound_text)
    remembered_tokens = estimate_tokens(remembered_context)
    leaked = bool(remembered_context and remembered_context in outbound_text)
    saved = max(0, remembered_tokens - outbound_tokens)

    return CompactChatAudit(
        outbound_messages=len(messages),
        outbound_tokens=outbound_tokens,
        remembered_tokens=remembered_tokens,
        estimated_memory_tokens=estimated_memory_tokens,
        tokens_saved=saved,
        reduction=0.0 if remembered_tokens == 0 else saved / remembered_tokens,
        full_transcript_leaked=leaked,
    )
