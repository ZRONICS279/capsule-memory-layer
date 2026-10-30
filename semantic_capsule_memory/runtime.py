"""Runtime adapters for using capsule memory with agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from .capsule import Capsule
from .compiler import MemoryCompiler
from .prompt_codec import encode_prompt


class RuntimeMode(str, Enum):
    TEXT_SPINE = "text_spine"
    SOFT_EMBEDDINGS = "soft_embeddings"
    PREFIX_CACHE = "prefix_cache"


@dataclass(frozen=True)
class PreparedMemory:
    """Memory payload ready for an agent/model call."""

    mode: RuntimeMode
    capsule_id: str
    version: int
    text_context: str = ""
    soft_token_count: int = 0
    estimated_text_tokens: int = 0
    estimated_original_tokens: int = 0
    evidence_refs: dict[str, str] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        if self.estimated_text_tokens == 0:
            return float("inf")
        return self.estimated_original_tokens / self.estimated_text_tokens


class AgentRuntime(Protocol):
    """Contract an agent integration implements."""

    mode: RuntimeMode

    def prepare_memory(self, capsule: Capsule, *, original_token_estimate: int = 0, query: str = "") -> PreparedMemory:
        ...


class TextSpineRuntime:
    """Works with any text-only LLM API today.

    This does not achieve one-token latent memory. It is the compatible fallback
    that injects only the exact symbolic spine, not the full transcript.
    """

    mode = RuntimeMode.TEXT_SPINE

    def __init__(self, max_facts: int = 24, token_budget: int = 256, prompt_codec: str = "compact"):
        self.max_facts = max_facts
        self.compiler = MemoryCompiler(token_budget=token_budget)
        if prompt_codec not in {"verbose", "compact"}:
            raise ValueError("prompt_codec must be 'verbose' or 'compact'")
        self.prompt_codec = prompt_codec

    def prepare_memory(self, capsule: Capsule, *, original_token_estimate: int = 0, query: str = "") -> PreparedMemory:
        compiled = self.compiler.compile(capsule, query=query)
        encoded = encode_prompt(compiled.payload, codec=self.prompt_codec)
        context = encoded.payload
        return PreparedMemory(
            mode=self.mode,
            capsule_id=capsule.capsule_id,
            version=capsule.version,
            text_context=context,
            estimated_text_tokens=encoded.token_estimate,
            estimated_original_tokens=original_token_estimate,
            evidence_refs=encoded.evidence_refs,
        )


class SoftEmbeddingRuntime:
    """Runtime descriptor for models that accept dynamic embeddings.

    The actual tensor path lives in `soft_tokens.py`; this class gives agent
    frameworks a stable capability signal and budget object.
    """

    mode = RuntimeMode.SOFT_EMBEDDINGS

    def __init__(self, soft_token_count: int):
        if soft_token_count <= 0:
            raise ValueError("soft_token_count must be positive")
        self.soft_token_count = soft_token_count

    def prepare_memory(self, capsule: Capsule, *, original_token_estimate: int = 0, query: str = "") -> PreparedMemory:
        return PreparedMemory(
            mode=self.mode,
            capsule_id=capsule.capsule_id,
            version=capsule.version,
            soft_token_count=self.soft_token_count,
            estimated_text_tokens=self.soft_token_count,
            estimated_original_tokens=original_token_estimate,
        )


class PrefixCacheRuntime:
    """Runtime descriptor for engines that can inject a capsule as KV cache."""

    mode = RuntimeMode.PREFIX_CACHE

    def __init__(self, prefix_token_equivalent: int):
        if prefix_token_equivalent <= 0:
            raise ValueError("prefix_token_equivalent must be positive")
        self.prefix_token_equivalent = prefix_token_equivalent

    def prepare_memory(self, capsule: Capsule, *, original_token_estimate: int = 0, query: str = "") -> PreparedMemory:
        return PreparedMemory(
            mode=self.mode,
            capsule_id=capsule.capsule_id,
            version=capsule.version,
            soft_token_count=self.prefix_token_equivalent,
            estimated_text_tokens=self.prefix_token_equivalent,
            estimated_original_tokens=original_token_estimate,
        )
