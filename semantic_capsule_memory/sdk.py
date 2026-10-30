"""Public SDK surface for applications and agent frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .capsule import Capsule, CapsuleManager
from .compiler import CompiledMemory, MemoryCompiler
from .budget import estimate_tokens
from .runtime import AgentRuntime, PreparedMemory, TextSpineRuntime


@dataclass(frozen=True)
class MemoryWrite:
    capsule_id: str
    text: str
    scope: str = "global"
    label: str | None = None


class CapsuleMemoryClient:
    """Small stable API companies can embed in agents."""

    def __init__(self, store: str | Path, runtime: AgentRuntime | None = None):
        self.manager = CapsuleManager(store)
        self.runtime = runtime or TextSpineRuntime()
        self.compiler = MemoryCompiler()

    def remember(self, write: MemoryWrite) -> Capsule:
        if write.capsule_id not in self.manager.capsules:
            self.manager.create(write.capsule_id, write.label)
        return self.manager.update(write.capsule_id, write.text, scope=write.scope)

    def prepare(self, capsule_id: str, *, original_context: str = "", query: str = "") -> PreparedMemory:
        capsule = self.manager.get(capsule_id)
        return self.runtime.prepare_memory(
            capsule,
            original_token_estimate=estimate_tokens(original_context),
            query=query,
        )

    def compile(self, capsule_id: str, *, query: str = "", original_context: str = "", token_budget: int = 256) -> CompiledMemory:
        capsule = self.manager.get(capsule_id)
        return MemoryCompiler(token_budget=token_budget).compile(
            capsule,
            query=query,
            original_context=original_context,
        )

    def expand_evidence(self, capsule_id: str, source: str) -> str:
        capsule = self.manager.get(capsule_id)
        for event in capsule.events:
            if event.source == source:
                return event.text
        raise KeyError(f"Unknown evidence source for {capsule_id}: {source}")

    def inspect(self, capsule_id: str) -> dict:
        return self.manager.get(capsule_id).inspect()

    def query(self, capsule_id: str, key: str, *, scope: str | None = None):
        return self.manager.get(capsule_id).query_fact(key, scope=scope)
