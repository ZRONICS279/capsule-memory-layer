"""Memory adapter baselines for the agent task harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections import Counter, defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
import json
import math
import os

# Suppress mem0's PostHog telemetry before any mem0 import.  The telemetry
# module reads this env var at import time; setting it here ensures it is
# already false when RealMem0Adapter lazily imports mem0.
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import re
import time
import urllib.request
from urllib.error import HTTPError, URLError

from semantic_capsule_memory.agentic import AgenticMemoryLayer
from semantic_capsule_memory.budget import estimate_tokens
from semantic_capsule_memory.prompt_codec import encode_prompt
from semantic_capsule_memory.symbolic import extract_facts


KEY_VALUE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]{1,40})\s*(?:=|:)\s*([^\n;.]+)")


@dataclass(frozen=True)
class PreparedContext:
    adapter: str
    payload: str
    token_estimate: int
    raw_token_estimate: int
    memory_payload_tokens: int = 0

    @property
    def token_savings(self) -> float:
        if self.raw_token_estimate <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.token_estimate / self.raw_token_estimate))


def compact_capsule_prompt(payload: str) -> str:
    """Use the package prompt codec so benchmarks measure production behavior."""

    return encode_prompt(payload, codec="compact").payload


class MemoryAdapter(ABC):
    name = "base"

    def __init__(self) -> None:
        self.events: list[str] = []
        self.tool_results: list[str] = []

    def reset(self, task_id: str) -> None:
        self.events = []
        self.tool_results = []

    def ingest_event(self, text: str, *, scope: str = "project") -> None:
        self.events.append(text)

    def remember_tool_result(self, text: str) -> None:
        self.tool_results.append(text)

    def close(self) -> None:
        pass

    @abstractmethod
    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        ...

    def raw_context(self) -> str:
        chunks = [*self.events, *self.tool_results]
        return "\n\n".join(chunks)


class NoMemoryAdapter(MemoryAdapter):
    name = "no_memory"

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        raw_tokens = estimate_tokens(self.raw_context())
        return PreparedContext(self.name, "", 0, raw_tokens)


class FullContextAdapter(MemoryAdapter):
    name = "full_context"

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        payload = self.raw_context()
        tokens = estimate_tokens(payload)
        return PreparedContext(self.name, payload, tokens, tokens)


class SlidingWindowAdapter(MemoryAdapter):
    name = "sliding_window"

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        self.window = window

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        payload = "\n\n".join([*self.events[-self.window :], *self.tool_results[-self.window :]])
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))


class SummaryMemoryAdapter(MemoryAdapter):
    name = "summary"

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        facts: list[str] = []
        seen: set[str] = set()
        for event in self.events:
            for match in KEY_VALUE_RE.finditer(event):
                item = f"{match.group(1).strip()}={match.group(2).strip()}"
                if item not in seen:
                    seen.add(item)
                    facts.append(item)
        payload = "SUMMARY_FACTS:\n" + "\n".join(f"- {fact}" for fact in facts[:20]) if facts else ""
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))


class LexicalOverlapAdapter(MemoryAdapter):
    """Weak lexical retrieval baseline.

    This is not industry RAG. It is intentionally kept as a cheap baseline that
    scores chunks by raw query-term overlap.
    """

    name = "lexical_overlap"

    def __init__(self, top_k: int = 4) -> None:
        super().__init__()
        self.top_k = top_k

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        query_terms = _terms(task)
        scored: list[tuple[int, str]] = []
        for event in [*self.events, *self.tool_results]:
            score = len(query_terms & _terms(event))
            if score:
                scored.append((score, event))
        payload = "\n\n".join(event for _, event in sorted(scored, reverse=True)[: self.top_k])
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))


class KeywordRagAdapter(LexicalOverlapAdapter):
    """Backward-compatible alias for old reports.

    Prefer ``lexical_overlap`` or ``bm25_rag`` in new benchmark runs.
    """

    name = "keyword_rag"


class BM25RagAdapter(MemoryAdapter):
    """Sparse RAG baseline using Okapi BM25 over memory/tool chunks."""

    name = "bm25_rag"

    def __init__(self, top_k: int = 4, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.top_k = top_k
        self.k1 = k1
        self.b = b

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        chunks = [*self.events, *self.tool_results]
        query_terms = list(_terms(task))
        scored = _bm25_rank(chunks, query_terms, k1=self.k1, b=self.b)
        payload = "\n\n".join(chunk for _, chunk in scored[: self.top_k])
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))


class OpenAIEmbeddingRagAdapter(MemoryAdapter):
    """Embedding RAG baseline using OpenAI embeddings and cosine retrieval.

    This mirrors the common industry pattern: embed memory chunks, embed the
    current task, retrieve top-k semantically nearest chunks. It uses an
    in-process vector index so the harness stays dependency-light; external
    vector DBs like Chroma/Qdrant/Pinecone mostly change storage/scale, not this
    retrieval contract.
    """

    name = "openai_embedding_rag"

    def __init__(self, top_k: int = 4, model: str | None = None) -> None:
        super().__init__()
        self.top_k = top_k
        self.model = model or os.environ.get("SCM_EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = os.environ.get("AYYGENT_AI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.max_chunk_chars = int(os.environ.get("SCM_EMBEDDING_MAX_CHARS", "12000"))
        self.batch_size = int(os.environ.get("SCM_EMBEDDING_BATCH_SIZE", "32"))
        self._indexed_count = 0
        self._vectors: list[list[float]] = []

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        self._indexed_count = 0
        self._vectors = []

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        if not self.api_key:
            raise RuntimeError("set AYYGENT_AI_API_KEY or OPENAI_API_KEY to use openai_embedding_rag")

        chunks = [*self.events, *self.tool_results]
        if len(chunks) > self._indexed_count:
            new_vectors: list[list[float]] = []
            pending = [_embedding_input(chunk, self.max_chunk_chars) for chunk in chunks[self._indexed_count :]]
            for start in range(0, len(pending), self.batch_size):
                new_vectors.extend(self._embed(pending[start : start + self.batch_size]))
            self._vectors.extend(new_vectors)
            self._indexed_count = len(chunks)

        if not chunks:
            return PreparedContext(self.name, "", 0, 0)

        query_vector = self._embed([_embedding_input(task, self.max_chunk_chars)])[0]
        scored = [(_cosine(query_vector, vector), chunk) for vector, chunk in zip(self._vectors, chunks)]
        payload = "\n\n".join(chunk for _, chunk in sorted(scored, reverse=True)[: self.top_k])
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        body = json.dumps({"model": self.model, "input": inputs}).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_exc: BaseException | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                data = sorted(payload["data"], key=lambda item: item["index"])
                return [item["embedding"] for item in data]
            except HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"embedding request failed HTTP {exc.code}: {body_text[:1000]}") from exc
                last_exc = RuntimeError(f"HTTP {exc.code}: {body_text[:1000]}")
            except (TimeoutError, URLError) as exc:
                last_exc = exc
            time.sleep(min(4.0, 0.75 * (2**attempt)))
        raise RuntimeError(f"embedding request failed: {last_exc}")


class ChromaRagAdapter(MemoryAdapter):
    """Chroma-backed vector RAG baseline.

    Requires ``chromadb``. By default it uses OpenAI embeddings through Chroma
    when OPENAI_API_KEY/AYYGENT_AI_API_KEY is available; otherwise it lets Chroma
    use its default embedding function.
    """

    name = "chroma_rag"

    def __init__(self, top_k: int = 4) -> None:
        super().__init__()
        self.top_k = top_k
        self._client = None
        self._collection = None
        self._indexed_count = 0
        self._tmp: TemporaryDirectory[str] | None = None

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        self.close()
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as exc:
            raise RuntimeError("Install Chroma with: python3 -m pip install chromadb") from exc

        self._tmp = TemporaryDirectory()
        self._client = chromadb.PersistentClient(path=str(Path(self._tmp.name) / "chroma"))
        api_key = os.environ.get("AYYGENT_AI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        embedding_function = None
        if api_key:
            embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=os.environ.get("SCM_EMBEDDING_MODEL", "text-embedding-3-small"),
            )
        self._collection = self._client.get_or_create_collection(
            name=re.sub(r"[^A-Za-z0-9_-]+", "_", task_id)[:60] or "task",
            embedding_function=embedding_function,
        )
        self._indexed_count = 0

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        if self._collection is None:
            raise RuntimeError("Chroma adapter was not reset")
        chunks = [*self.events, *self.tool_results]
        if len(chunks) > self._indexed_count:
            new_chunks = chunks[self._indexed_count :]
            ids = [f"chunk-{idx}" for idx in range(self._indexed_count, len(chunks))]
            docs = [_embedding_input(chunk, int(os.environ.get("SCM_CHROMA_MAX_CHARS", "12000"))) for chunk in new_chunks]
            self._collection.add(ids=ids, documents=docs, metadatas=[{"index": i} for i in range(self._indexed_count, len(chunks))])
            self._indexed_count = len(chunks)
        if not chunks:
            return PreparedContext(self.name, "", 0, 0)
        result = self._collection.query(query_texts=[task], n_results=min(self.top_k, len(chunks)))
        docs = (result.get("documents") or [[]])[0]
        payload = "\n\n".join(docs)
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = None
        self._client = None
        self._collection = None
        self._indexed_count = 0


class RepoSearchAdapter(MemoryAdapter):
    name = "repo_search"

    def __init__(self, max_files: int = 8, chars_per_file: int = 900) -> None:
        super().__init__()
        self.max_files = max_files
        self.chars_per_file = chars_per_file

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        query_terms = _terms(task)
        scored: list[tuple[int, str]] = []
        for path in sorted(repo_path.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".md", ".toml", ".txt"}:
                continue
            rel = path.relative_to(repo_path)
            text = path.read_text(encoding="utf-8")
            score = len(query_terms & _terms(f"{rel} {text}"))
            if score:
                scored.append((score, f"FILE {rel}\n{text[: self.chars_per_file]}"))
        payload = "\n\n".join(chunk for _, chunk in sorted(scored, reverse=True)[: self.max_files])
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))


class LangGraphStateAdapter(MemoryAdapter):
    name = "langgraph_state"

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        state: dict[str, list[str]] = defaultdict(list)
        for event in self.events:
            for fact in extract_facts(event, scope="project"):
                item = f"{fact.key}={fact.value}"
                if item not in state[fact.key]:
                    state[fact.key].append(item)
        lines = ["STATE:"]
        for key in sorted(state):
            lines.extend(f"- {item}" for item in state[key][-3:])
        payload = "\n".join(lines) if len(lines) > 1 else ""
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))


class Mem0StyleAdapter(MemoryAdapter):
    """Mem0-style extracted memory baseline.

    Mem0's public contract is closer to "extract durable memories, update them,
    retrieve relevant memories" than plain vector RAG. This local adapter models
    that behavior without requiring a hosted Mem0 account: it extracts compact
    facts, deduplicates/updates them by key, and retrieves a small task-relevant
    memory set with BM25. Use this as the always-runnable baseline; wire a hosted
    Mem0 SDK adapter separately if you want vendor API behavior.
    """

    name = "mem0_style"

    def __init__(self, top_k: int = 8) -> None:
        super().__init__()
        self.top_k = top_k
        self._memories: dict[str, str] = {}
        self._chronological: list[str] = []

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        self._memories = {}
        self._chronological = []

    def ingest_event(self, text: str, *, scope: str = "project") -> None:
        super().ingest_event(text, scope=scope)
        self._extract_memories(text, scope=scope)

    def remember_tool_result(self, text: str) -> None:
        super().remember_tool_result(text)
        self._extract_memories(text, scope="tool")

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        memories = list(self._memories.values()) or self._chronological
        scored = _bm25_rank(memories, list(_terms(task)), k1=1.3, b=0.65)
        selected = [chunk for _, chunk in scored[: self.top_k]]
        if len(selected) < min(self.top_k, len(memories)):
            seen = set(selected)
            for item in reversed(memories):
                if item not in seen:
                    selected.append(item)
                    seen.add(item)
                if len(selected) >= self.top_k:
                    break
        payload = "MEM0_RETRIEVED_MEMORIES:\n" + "\n".join(f"- {item}" for item in selected) if selected else ""
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))

    def _extract_memories(self, text: str, *, scope: str) -> None:
        found = False
        for fact in extract_facts(text, scope=scope):
            key = fact.key.lower()
            memory = f"{fact.key}={fact.value}"
            self._memories[key] = memory
            found = True
        for match in KEY_VALUE_RE.finditer(text):
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            memory = f"{match.group(1).strip()}={value}"
            self._memories[key] = memory
            found = True
        if not found:
            compact = _salient_sentence(text)
            if compact:
                self._chronological.append(compact)
                if len(self._chronological) > 80:
                    self._chronological = self._chronological[-80:]


class LangGraphCheckpointAdapter(MemoryAdapter):
    """LangGraph-style checkpoint plus store baseline.

    LangGraph commonly separates short-term graph state/checkpoints from a
    long-term store. This adapter keeps the latest state fields, recent tool
    observations, and a small retrieved long-term store.
    """

    name = "langgraph_checkpoint"

    def __init__(self, top_k: int = 5, recent_tools: int = 3) -> None:
        super().__init__()
        self.top_k = top_k
        self.recent_tools = recent_tools
        self._state: dict[str, str] = {}
        self._store: list[str] = []

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        self._state = {}
        self._store = []

    def ingest_event(self, text: str, *, scope: str = "project") -> None:
        super().ingest_event(text, scope=scope)
        for fact in extract_facts(text, scope=scope):
            self._state[fact.key] = f"{fact.key}={fact.value}"
        salient = _salient_sentence(text)
        if salient:
            self._store.append(salient)

    def remember_tool_result(self, text: str) -> None:
        super().remember_tool_result(text)
        salient = _salient_sentence(text)
        if salient:
            self._store.append(f"tool: {salient}")

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        state_lines = [self._state[key] for key in sorted(self._state)]
        retrieved = [chunk for _, chunk in _bm25_rank(self._store, list(_terms(task)), k1=1.4, b=0.7)[: self.top_k]]
        recent = self.tool_results[-self.recent_tools :]
        sections: list[str] = []
        if state_lines:
            sections.append("CHECKPOINT_STATE:\n" + "\n".join(f"- {line}" for line in state_lines[-40:]))
        if retrieved:
            sections.append("LONG_TERM_STORE:\n" + "\n".join(f"- {line}" for line in retrieved))
        if recent:
            sections.append("RECENT_TOOL_OBSERVATIONS:\n" + "\n\n".join(recent))
        payload = "\n\n".join(sections)
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))


class ZepGraphitiStyleAdapter(MemoryAdapter):
    """Zep/Graphiti-style temporal graph memory baseline.

    This models the useful part for these repo tasks: current entity facts,
    temporal supersession, and relationship-like edges. It deliberately favors
    latest facts when stale and current memories conflict.
    """

    name = "zep_graphiti_style"

    def __init__(self, top_k: int = 12) -> None:
        super().__init__()
        self.top_k = top_k
        self._tick = 0
        self._facts: dict[str, tuple[int, str]] = {}
        self._edges: list[tuple[int, str]] = []

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        self._tick = 0
        self._facts = {}
        self._edges = []

    def ingest_event(self, text: str, *, scope: str = "project") -> None:
        super().ingest_event(text, scope=scope)
        self._ingest_graph_text(text, scope=scope)

    def remember_tool_result(self, text: str) -> None:
        super().remember_tool_result(text)
        self._ingest_graph_text(text, scope="tool")

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        query = _terms(task)
        fact_items = [value for _, value in sorted(self._facts.values(), reverse=True)]
        edge_items = [edge for _, edge in sorted(self._edges, reverse=True)]
        candidates = fact_items + edge_items
        scored: list[tuple[float, str]] = []
        for idx, item in enumerate(candidates):
            lexical = len(query & _terms(item))
            recency = 1.0 / (idx + 1)
            if lexical or idx < 8:
                scored.append((lexical * 10.0 + recency, item))
        selected = [item for _, item in sorted(scored, reverse=True)[: self.top_k]]
        payload = "TEMPORAL_KNOWLEDGE_GRAPH:\n" + "\n".join(f"- {item}" for item in selected) if selected else ""
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))

    def _ingest_graph_text(self, text: str, *, scope: str) -> None:
        self._tick += 1
        before = len(self._facts)
        for fact in extract_facts(text, scope=scope):
            key = fact.key.lower()
            self._facts[key] = (self._tick, f"{fact.key}={fact.value} @t{self._tick}")
        for match in KEY_VALUE_RE.finditer(text):
            key = match.group(1).strip().lower()
            self._facts[key] = (self._tick, f"{match.group(1).strip()}={match.group(2).strip()} @t{self._tick}")
        if len(self._facts) == before:
            salient = _salient_sentence(text)
            if salient:
                self._edges.append((self._tick, f"event @t{self._tick}: {salient}"))
                if len(self._edges) > 100:
                    self._edges = self._edges[-100:]


class MemGPTStyleAdapter(MemoryAdapter):
    """MemGPT/Letta-style core plus archival memory baseline."""

    name = "memgpt_style"

    def __init__(self, archival_top_k: int = 4, core_limit: int = 24) -> None:
        super().__init__()
        self.archival_top_k = archival_top_k
        self.core_limit = core_limit
        self._core: dict[str, str] = {}
        self._archival: list[str] = []

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        self._core = {}
        self._archival = []

    def ingest_event(self, text: str, *, scope: str = "project") -> None:
        super().ingest_event(text, scope=scope)
        self._ingest(text, scope=scope)

    def remember_tool_result(self, text: str) -> None:
        super().remember_tool_result(text)
        self._ingest(text, scope="tool")

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        core_lines = list(self._core.values())[-self.core_limit :]
        archival = [chunk for _, chunk in _bm25_rank(self._archival, list(_terms(task)), k1=1.5, b=0.75)[: self.archival_top_k]]
        sections: list[str] = []
        if core_lines:
            sections.append("CORE_MEMORY:\n" + "\n".join(f"- {line}" for line in core_lines))
        if archival:
            sections.append("ARCHIVAL_SEARCH_RESULTS:\n" + "\n".join(f"- {line}" for line in archival))
        payload = "\n\n".join(sections)
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))

    def _ingest(self, text: str, *, scope: str) -> None:
        extracted = False
        for fact in extract_facts(text, scope=scope):
            self._core[fact.key.lower()] = f"{fact.key}={fact.value}"
            extracted = True
        for match in KEY_VALUE_RE.finditer(text):
            self._core[match.group(1).strip().lower()] = f"{match.group(1).strip()}={match.group(2).strip()}"
            extracted = True
        salient = _salient_sentence(text)
        if salient:
            self._archival.append(salient)
        elif extracted:
            self._archival.extend(self._core.values())
        if len(self._archival) > 160:
            self._archival = self._archival[-160:]


# Vendor-name aliases. These are local, dependency-free contract baselines rather
# than hosted SDK integrations, which keeps benchmark runs reproducible.
class Mem0Adapter(Mem0StyleAdapter):
    name = "mem0"


class ZepAdapter(ZepGraphitiStyleAdapter):
    name = "zep"


class GraphitiAdapter(ZepGraphitiStyleAdapter):
    name = "graphiti"


class LangGraphAdapter(LangGraphCheckpointAdapter):
    name = "langgraph"


class MemGPTAdapter(MemGPTStyleAdapter):
    name = "memgpt"


class LettaAdapter(MemGPTStyleAdapter):
    name = "letta"


class RealMem0Adapter(MemoryAdapter):
    """Real Mem0 SDK baseline using mem0.Memory.

    This is intentionally separate from ``mem0``/``mem0_style``. It uses the
    installed Mem0 package and its memory extraction/search pipeline. It needs an
    OpenAI-compatible key because Mem0 performs LLM extraction plus embeddings.

    Parameters
    ----------
    infer:
        When True (default), each ``ingest_event`` call sends the text to an LLM
        to extract and deduplicate semantic facts — mem0's key differentiator for
        cross-session use. When False, events are stored as raw vector chunks
        (fast, no LLM cost at ingest time; use ``mem0_real_fast`` alias).
    store_path:
        Optional persistent directory for the Qdrant vector store and history DB.
        If None a temporary directory is created and cleaned up on ``close()``.
        Pass a fixed path to enable true cross-session persistence across multiple
        benchmark runs.
    """

    name = "mem0_real"

    def __init__(self, top_k: int = 8, *, infer: bool = True, store_path: Path | None = None) -> None:
        super().__init__()
        self.top_k = top_k
        self.infer = infer
        self._store_path = store_path
        self._tmp: TemporaryDirectory[str] | None = None
        self._memory = None
        self._user_id = "agent-task"

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        self.close()
        api_key = os.environ.get("AYYGENT_AI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("mem0_real requires AYYGENT_AI_API_KEY or OPENAI_API_KEY")
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise RuntimeError("Install real Mem0 SDK with: python3 -m pip install mem0ai") from exc
        # Use a caller-supplied persistent path or create a fresh temp dir.
        if self._store_path is not None:
            self._store_path.mkdir(parents=True, exist_ok=True)
            base = self._store_path
        else:
            self._tmp = TemporaryDirectory()
            base = Path(self._tmp.name)
        self._user_id = task_id
        config = _real_mem0_config(task_id, api_key, base)
        try:
            self._memory = Memory.from_config(config)
        except Exception as exc:
            raise RuntimeError(f"mem0_real failed to initialize Mem0 SDK: {exc}") from exc

    def ingest_event(self, text: str, *, scope: str = "project") -> None:
        super().ingest_event(text, scope=scope)
        if self._memory is None:
            raise RuntimeError("mem0_real adapter was not reset")
        self._memory.add(
            [{"role": "user", "content": text}],
            user_id=self._user_id,
            metadata={"scope": scope},
            infer=self.infer,
        )

    def remember_tool_result(self, text: str) -> None:
        super().remember_tool_result(text)
        if self._memory is None:
            raise RuntimeError("mem0_real adapter was not reset")
        self._memory.add(
            [{"role": "assistant", "content": f"TOOL_RESULT: {text}"}],
            user_id=self._user_id,
            metadata={"scope": "tool"},
            infer=self.infer,
        )

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        if self._memory is None:
            raise RuntimeError("mem0_real adapter was not reset")
        result = self._memory.search(task, top_k=self.top_k, filters={"user_id": self._user_id})
        memories = _mem0_search_texts(result)
        payload = "MEM0_REAL_SEARCH_RESULTS:\n" + "\n".join(f"- {item}" for item in memories) if memories else ""
        mem_tokens = estimate_tokens(payload)
        return PreparedContext(self.name, payload, mem_tokens, estimate_tokens(self.raw_context()), mem_tokens)

    def close(self) -> None:
        self._memory = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


class RealMem0AsyncAdapter(RealMem0Adapter):
    """Real Mem0 SDK baseline using AsyncMemory with concurrent writes.

    The benchmark still waits for all scheduled writes before retrieval. This
    tests whether mem0's async write API reduces wall-clock ingest time without
    changing the memory-quality contract.
    """

    name = "mem0_real_async"

    def __init__(
        self,
        top_k: int = 8,
        *,
        infer: bool = True,
        store_path: Path | None = None,
        max_pending: int = 64,
    ) -> None:
        super().__init__(top_k=top_k, infer=infer, store_path=store_path)
        self.max_pending = max_pending
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._pending: list[Future] = []

    def reset(self, task_id: str) -> None:
        MemoryAdapter.reset(self, task_id)
        self.close()
        api_key = os.environ.get("AYYGENT_AI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("mem0_real_async requires AYYGENT_AI_API_KEY or OPENAI_API_KEY")
        try:
            from mem0 import AsyncMemory
        except ImportError as exc:
            raise RuntimeError("Install real Mem0 SDK with: python3 -m pip install mem0ai") from exc

        if self._store_path is not None:
            self._store_path.mkdir(parents=True, exist_ok=True)
            base = self._store_path
        else:
            self._tmp = TemporaryDirectory()
            base = Path(self._tmp.name)
        self._user_id = task_id
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        config = _real_mem0_config(task_id, api_key, base)
        maybe_memory = AsyncMemory.from_config(config)
        if asyncio.iscoroutine(maybe_memory):
            self._memory = asyncio.run_coroutine_threadsafe(maybe_memory, self._loop).result()
        else:
            self._memory = maybe_memory

    def ingest_event(self, text: str, *, scope: str = "project") -> None:
        super(RealMem0Adapter, self).ingest_event(text, scope=scope)
        self._schedule_add(
            [{"role": "user", "content": text}],
            metadata={"scope": scope},
        )

    def remember_tool_result(self, text: str) -> None:
        super(RealMem0Adapter, self).remember_tool_result(text)
        self._schedule_add(
            [{"role": "assistant", "content": f"TOOL_RESULT: {text}"}],
            metadata={"scope": "tool"},
        )

    def flush_ingest(self) -> None:
        while self._pending:
            pending, self._pending = self._pending, []
            for future in pending:
                future.result()

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        if self._memory is None or self._loop is None:
            raise RuntimeError("mem0_real_async adapter was not reset")
        self.flush_ingest()
        future = asyncio.run_coroutine_threadsafe(
            self._memory.search(task, top_k=self.top_k, filters={"user_id": self._user_id}),
            self._loop,
        )
        result = future.result()
        memories = _mem0_search_texts(result)
        payload = "MEM0_REAL_ASYNC_SEARCH_RESULTS:\n" + "\n".join(f"- {item}" for item in memories) if memories else ""
        mem_tokens = estimate_tokens(payload)
        return PreparedContext(self.name, payload, mem_tokens, estimate_tokens(self.raw_context()), mem_tokens)

    def close(self) -> None:
        if self._memory is not None:
            self.flush_ingest()
        self._memory = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None
        self._pending = []
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def _schedule_add(self, messages, *, metadata: dict) -> None:
        if self._memory is None or self._loop is None:
            raise RuntimeError("mem0_real_async adapter was not reset")
        future = asyncio.run_coroutine_threadsafe(
            self._memory.add(
                messages,
                user_id=self._user_id,
                metadata=metadata,
                infer=self.infer,
            ),
            self._loop,
        )
        self._pending.append(future)
        if len(self._pending) >= self.max_pending:
            self.flush_ingest()

    def _run_loop(self) -> None:
        if self._loop is None:
            return
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()


class RealMem0FastAdapter(RealMem0Adapter):
    """mem0_real with infer=False: raw vector store, no LLM extraction at ingest.

    Use this to isolate whether mem0's LLM fact-extraction step (infer=True)
    actually pays off over plain embedding RAG, without the per-event LLM cost.
    """

    name = "mem0_real_fast"

    def __init__(self, top_k: int = 8, *, store_path: Path | None = None) -> None:
        super().__init__(top_k=top_k, infer=False, store_path=store_path)


def _real_mem0_config(task_id: str, api_key: str, base: Path) -> dict:
    embed_model = os.environ.get("SCM_MEM0_EMBEDDING_MODEL", os.environ.get("SCM_EMBEDDING_MODEL", "text-embedding-3-small"))
    llm_model = os.environ.get("SCM_MEM0_LLM_MODEL", os.environ.get("SCM_AGENT_MODEL", "gpt-4.1-mini"))
    dims = int(os.environ.get("SCM_MEM0_EMBEDDING_DIMS", "1536"))
    return {
        "llm": {"provider": "openai", "config": {"api_key": api_key, "model": llm_model, "temperature": 0.0}},
        "embedder": {"provider": "openai", "config": {"api_key": api_key, "model": embed_model, "embedding_dims": dims}},
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": re.sub(r"[^A-Za-z0-9_-]+", "_", task_id)[:60] or "task",
                "path": str(base / "qdrant"),
                "embedding_model_dims": dims,
                "on_disk": False,
            },
        },
        "history_db_path": str(base / "history.db"),
    }


class RealLangGraphAdapter(MemoryAdapter):
    """Real LangGraph Store baseline using langgraph.store.memory.InMemoryStore."""

    name = "langgraph_real"

    def __init__(self, top_k: int = 8) -> None:
        super().__init__()
        self.top_k = top_k
        self._store = None
        self._namespace: tuple[str, ...] = ("agent-task",)
        self._counter = 0

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        try:
            from langgraph.store.memory import InMemoryStore
        except ImportError as exc:
            raise RuntimeError("Install real LangGraph with: python3 -m pip install langgraph") from exc
        self._store = InMemoryStore()
        self._namespace = ("agent-task", task_id)
        self._counter = 0

    def ingest_event(self, text: str, *, scope: str = "project") -> None:
        super().ingest_event(text, scope=scope)
        self._put(text, kind="event", scope=scope)

    def remember_tool_result(self, text: str) -> None:
        super().remember_tool_result(text)
        self._put(text, kind="tool", scope="project")

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        if self._store is None:
            raise RuntimeError("langgraph_real adapter was not reset")
        items = self._store.search(self._namespace, query=task, limit=self.top_k)
        texts = [_langgraph_item_text(item) for item in items]
        payload = "LANGGRAPH_REAL_STORE_RESULTS:\n" + "\n".join(f"- {item}" for item in texts if item) if texts else ""
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))

    def _put(self, text: str, *, kind: str, scope: str) -> None:
        if self._store is None:
            raise RuntimeError("langgraph_real adapter was not reset")
        self._counter += 1
        self._store.put(
            self._namespace,
            f"{self._counter:06d}",
            {"text": text, "kind": kind, "scope": scope, "index": self._counter},
            index=["text"],
        )


class RealZepAdapter(MemoryAdapter):
    """Real Zep Cloud SDK baseline using the v3 thread and graph APIs."""

    name = "zep_real"

    def __init__(self, top_k: int = 8) -> None:
        super().__init__()
        self.top_k = top_k
        self._client = None
        self._session_id = ""
        self._user_id = "agent-task"
        self._latest_context = ""

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        api_key = os.environ.get("ZEP_API_KEY")
        if not api_key:
            raise RuntimeError("zep_real requires ZEP_API_KEY")
        try:
            from zep_cloud import Zep
        except Exception as exc:
            raise RuntimeError("Install Zep Cloud SDK with: uv pip install zep-cloud") from exc

        run_suffix = f"{int(time.time() * 1000)}"
        self._session_id = f"{task_id}-{run_suffix}"
        self._user_id = f"agent-task-{task_id}-{run_suffix}"
        self._latest_context = ""
        self._client = Zep(api_key=api_key, timeout=120)
        try:
            self._client.user.add(
                user_id=self._user_id,
                metadata={"benchmark": "agent_task_harness", "task_id": task_id},
            )
        except Exception as exc:
            if "409" not in str(exc) and "Conflict" not in exc.__class__.__name__:
                raise RuntimeError(f"zep_real failed to create Zep user: {exc}") from exc
        try:
            self._client.thread.create(
                thread_id=self._session_id,
                user_id=self._user_id,
            )
        except Exception as exc:
            if "409" not in str(exc) and "Conflict" not in exc.__class__.__name__:
                raise RuntimeError(f"zep_real failed to create Zep thread: {exc}") from exc

    def ingest_event(self, text: str, *, scope: str = "project") -> None:
        super().ingest_event(text, scope=scope)
        self._add_message("user", text, scope=scope)

    def remember_tool_result(self, text: str) -> None:
        super().remember_tool_result(text)
        self._add_message("assistant", f"TOOL_RESULT: {text}", scope="tool")

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        if self._client is None:
            raise RuntimeError("zep_real adapter was not reset")
        result = self._client.graph.search(
            query=task,
            user_id=self._user_id,
            scope="auto",
            reranker="rrf",
            limit=self.top_k,
            max_characters=2500,
        )
        payload = result.context or ""
        if not payload:
            try:
                context = self._client.thread.get_user_context(self._session_id)
                payload = context.context or ""
            except Exception:
                payload = ""
        if not payload:
            payload = self._latest_context
        if not payload:
            try:
                messages = self._client.thread.get_messages(self._session_id, lastn=self.top_k)
                payload = "\n".join(message.content for message in messages.messages or [] if message.content)
            except Exception:
                payload = ""
        if payload:
            payload = "ZEP_REAL_CONTEXT:\n" + payload
        return PreparedContext(self.name, payload, estimate_tokens(payload), estimate_tokens(self.raw_context()))

    def _add_message(self, role: str, text: str, *, scope: str) -> None:
        if self._client is None:
            raise RuntimeError("zep_real adapter was not reset")
        from zep_cloud import Message

        response = self._client.thread.add_messages(
            self._session_id,
            messages=[Message(role=role, content=text, metadata={"scope": scope})],
            return_context=True,
        )
        if response.context:
            self._latest_context = response.context


class RealLettaAdapter(MemoryAdapter):
    """Real Letta SDK/server baseline placeholder with strict failure semantics."""

    name = "letta_real"

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        try:
            import letta
        except ImportError as exc:
            raise RuntimeError("Install real Letta SDK/server package before using letta_real") from exc
        exported = [name for name in dir(letta) if not name.startswith("_")]
        if not exported:
            raise RuntimeError(
                "letta_real found package 'letta' but it exposes no usable SDK API in this environment. "
                "Use a modern Letta server/client install and configure LETTA_BASE_URL/LETTA_API_KEY."
            )
        raise RuntimeError("letta_real SDK wiring needs a modern Letta client/server API configuration")

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        raise RuntimeError("letta_real was not initialized")


class SemanticCapsuleAdapter(MemoryAdapter):
    name = "capsule"

    def __init__(self, target_savings: float = 0.70, *, store_path: Path | None = None, token_budget: int = 160) -> None:
        super().__init__()
        self.target_savings = target_savings
        self._token_budget = token_budget
        self._store_path = store_path
        self._tmp: TemporaryDirectory[str] | None = None
        self._memory: AgenticMemoryLayer | None = None
        self._capsule_id = "agent-task"

    def reset(self, task_id: str) -> None:
        super().reset(task_id)
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
        self._capsule_id = task_id
        if self._store_path is not None:
            self._store_path.mkdir(parents=True, exist_ok=True)
            db_path = self._store_path / "memory.sqlite"
        else:
            self._tmp = TemporaryDirectory()
            db_path = Path(self._tmp.name) / "memory.sqlite"
        self._memory = AgenticMemoryLayer(
            db_path,
            default_token_budget=self._token_budget,
            target_reduction=self.target_savings,
            conversation_budget=0,
            lean_threshold=0,
        )

    def ingest_event(self, text: str, *, scope: str = "project") -> None:
        super().ingest_event(text, scope=scope)
        assert self._memory is not None
        self._memory.remember(self._capsule_id, text, scope=scope, label=f"@{self._capsule_id}")

    def remember_tool_result(self, text: str) -> None:
        super().remember_tool_result(text)
        assert self._memory is not None
        self._memory.remember(self._capsule_id, f"TOOL_RESULT: {text}", scope="project", label=f"@{self._capsule_id}")

    def prepare_context(self, task: str, repo_path: Path) -> PreparedContext:
        assert self._memory is not None
        raw = self.raw_context()
        packet = self._memory.prepare_step(self._capsule_id, task, original_context=raw, retrieve_limit=2)
        payload = compact_capsule_prompt(packet.payload)
        token_estimate = estimate_tokens(payload)
        return PreparedContext(
            self.name,
            payload,
            token_estimate,
            estimate_tokens(raw),
            memory_payload_tokens=token_estimate,
        )

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
            self._memory = None


def make_adapter(name: str) -> MemoryAdapter:
    normalized = name.strip().lower().replace("-", "_")
    adapters = {
        "no_memory": NoMemoryAdapter,
        "full_context": FullContextAdapter,
        "sliding_window": SlidingWindowAdapter,
        "summary": SummaryMemoryAdapter,
        "lexical_overlap": LexicalOverlapAdapter,
        "keyword_rag": KeywordRagAdapter,
        "bm25_rag": BM25RagAdapter,
        "openai_embedding_rag": OpenAIEmbeddingRagAdapter,
        "embedding_rag": OpenAIEmbeddingRagAdapter,
        "vector_rag": OpenAIEmbeddingRagAdapter,
        "chroma_rag": ChromaRagAdapter,
        "repo_search": RepoSearchAdapter,
        "langgraph_state": LangGraphStateAdapter,
        "langgraph_checkpoint": LangGraphCheckpointAdapter,
        "langgraph": LangGraphAdapter,
        "mem0_style": Mem0StyleAdapter,
        "mem0": Mem0Adapter,
        "zep_graphiti_style": ZepGraphitiStyleAdapter,
        "zep": ZepAdapter,
        "graphiti": GraphitiAdapter,
        "memgpt_style": MemGPTStyleAdapter,
        "memgpt": MemGPTAdapter,
        "letta": LettaAdapter,
        "mem0_real": RealMem0Adapter,
        "mem0_real_async": RealMem0AsyncAdapter,
        "mem0_real_fast": RealMem0FastAdapter,
        "langgraph_real": RealLangGraphAdapter,
        "zep_real": RealZepAdapter,
        "graphiti_real": RealZepAdapter,
        "letta_real": RealLettaAdapter,
        "memgpt_real": RealLettaAdapter,
        "capsule": SemanticCapsuleAdapter,
    }
    if normalized not in adapters:
        raise ValueError(f"unknown adapter {name!r}; choose one of {', '.join(sorted(adapters))}")
    adapter = adapters[normalized]()
    if normalized == "vector_rag":
        adapter.name = normalized
    return adapter


def _mem0_search_texts(result) -> list[str]:
    if isinstance(result, dict):
        candidates = result.get("results") or result.get("memories") or result.get("data") or []
    else:
        candidates = result or []
    texts: list[str] = []
    for item in candidates:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            text = item.get("memory") or item.get("text") or item.get("content")
            if text:
                texts.append(str(text))
        else:
            text = getattr(item, "memory", None) or getattr(item, "text", None) or getattr(item, "content", None)
            if text:
                texts.append(str(text))
    return texts


def _langgraph_item_text(item) -> str:
    value = getattr(item, "value", None)
    if isinstance(value, dict):
        return str(value.get("text") or value)
    return str(value or item)


def _zep_search_result_text(item) -> str:
    fact = getattr(item, "fact", None)
    if fact is not None:
        text = getattr(fact, "fact", None)
        if text:
            return str(text)
    message = getattr(item, "message", None)
    if message is not None:
        text = getattr(message, "content", None)
        if text:
            return str(text)
    summary = getattr(item, "summary", None)
    if summary is not None:
        text = getattr(summary, "content", None)
        if text:
            return str(text)
    return ""


def _salient_sentence(text: str, max_chars: int = 360) -> str:
    compact = " ".join(text.split())
    if not compact:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        terms = _terms(sentence)
        score = 0
        score += 4 if KEY_VALUE_RE.search(sentence) else 0
        score += 3 if any(term in terms for term in {"current", "final", "required", "must", "owner", "endpoint", "config", "route", "test"}) else 0
        score += min(4, len(terms) // 5)
        scored.append((score, sentence))
    best = max(scored, default=(0, compact))[1]
    if len(best) <= max_chars:
        return best
    return best[:max_chars].rsplit(" ", 1)[0]


def _embedding_input(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit(" ", 1)[0]


def _terms(text: str) -> set[str]:
    return set(_tokens(text))


def _tokens(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[A-Za-z0-9]{2,}", text)]


def _bm25_rank(chunks: list[str], query_terms: list[str], *, k1: float, b: float) -> list[tuple[float, str]]:
    if not chunks or not query_terms:
        return []

    tokenized = [_tokens(chunk) for chunk in chunks]
    doc_count = len(tokenized)
    avg_len = sum(len(tokens) for tokens in tokenized) / max(1, doc_count)
    df = Counter(term for tokens in tokenized for term in set(tokens))
    query_counts = Counter(query_terms)

    scored: list[tuple[float, str]] = []
    for chunk, tokens in zip(chunks, tokenized):
        counts = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for term, query_weight in query_counts.items():
            freq = counts.get(term, 0)
            if not freq:
                continue
            idf = math.log(1.0 + ((doc_count - df[term] + 0.5) / (df[term] + 0.5)))
            denom = freq + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_len)))
            score += query_weight * idf * ((freq * (k1 + 1.0)) / denom)
        if score > 0:
            scored.append((score, chunk))
    return sorted(scored, reverse=True)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
