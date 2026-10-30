"""Working memory engine.

This is the practical core: ingest lossless events, compile tiny memory payloads,
answer grounded questions from active facts, and expand exact evidence when
needed. It is deliberately deterministic so failures are debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .budget import estimate_tokens
from .capsule import Capsule, CapsuleManager
from .compiler import CompiledMemory, MemoryCompiler
from .sketch import tokenize
from .symbolic import Fact


@dataclass(frozen=True)
class MemoryAnswer:
    status: str
    answer: str
    confidence: float
    capsule_id: str
    evidence_sources: tuple[str, ...]
    compiled: CompiledMemory
    retrieved: tuple["EvidenceSnippet", ...] = ()


@dataclass(frozen=True)
class EvidenceSnippet:
    source: str
    text: str
    score: int
    token_estimate: int


class GroundedMemoryEngine:
    """End-to-end memory engine for agent loops."""

    def __init__(self, store: str | Path, *, token_budget: int = 192):
        self.manager = CapsuleManager(store)
        self.compiler = MemoryCompiler(token_budget=token_budget)

    def ingest(self, capsule_id: str, text: str, *, scope: str = "global", label: str | None = None) -> Capsule:
        if capsule_id not in self.manager.capsules:
            self.manager.create(capsule_id, label)
        return self.manager.update(capsule_id, text, scope=scope)

    def compile(self, capsule_id: str, *, question: str = "", original_context: str = "") -> CompiledMemory:
        return self.compiler.compile(
            self.manager.get(capsule_id),
            query=question,
            original_context=original_context,
        )

    def answer(self, capsule_id: str, question: str, *, original_context: str = "") -> MemoryAnswer:
        capsule = self.manager.get(capsule_id)
        compiled = self.compiler.compile(capsule, query=question, original_context=original_context)
        candidates = self._rank_answer_facts(capsule.active_facts(), question)

        if not candidates:
            retrieved = self.retrieve(capsule_id, question, limit=3)
            synthesized = _synthesize_from_snippets(question, retrieved)
            if synthesized:
                return MemoryAnswer(
                    status="answered_from_evidence",
                    answer=synthesized,
                    confidence=0.72,
                    capsule_id=capsule_id,
                    evidence_sources=tuple(snippet.source for snippet in retrieved[:1]),
                    compiled=compiled,
                    retrieved=retrieved,
                )
            return MemoryAnswer(
                status="unknown",
                answer="UNKNOWN",
                confidence=0.0,
                capsule_id=capsule_id,
                evidence_sources=(),
                compiled=compiled,
                retrieved=retrieved,
            )

        top_score = candidates[0][0]
        top = [fact for score, fact in candidates if score == top_score]
        values = {fact.value for fact in top}
        if len(values) > 1:
            return MemoryAnswer(
                status="conflict",
                answer="CONFLICT: " + "; ".join(sorted(values)),
                confidence=0.4,
                capsule_id=capsule_id,
                evidence_sources=tuple(fact.source for fact in top),
                compiled=compiled,
                retrieved=(),
            )

        fact = top[0]
        return MemoryAnswer(
            status="answered",
            answer=fact.value,
            confidence=fact.confidence,
            capsule_id=capsule_id,
            evidence_sources=(fact.source,),
            compiled=compiled,
            retrieved=(),
        )

    def expand(self, capsule_id: str, source: str) -> str:
        capsule = self.manager.get(capsule_id)
        for event in capsule.events:
            if event.source == source:
                return event.text
        raise KeyError(f"Unknown evidence source for {capsule_id}: {source}")

    def retrieve(self, capsule_id: str, query: str, *, limit: int = 3) -> tuple[EvidenceSnippet, ...]:
        """Retrieve exact raw-memory snippets without injecting the full ledger."""

        query_terms = set(tokenize(query.replace("_", " ")))
        if not query_terms:
            return ()

        capsule = self.manager.get(capsule_id)
        snippets: list[EvidenceSnippet] = []
        include_stale = _query_allows_stale_evidence(query)
        for event in capsule.events:
            for sentence in _sentences(event.text):
                if _skip_retrieved_sentence(sentence, include_stale=include_stale):
                    continue
                sentence = _clip_retrieved_sentence(sentence)
                sentence_terms = set(tokenize(sentence.replace("_", " ")))
                score = len(query_terms & sentence_terms)
                if score > 0:
                    snippets.append(
                        EvidenceSnippet(
                            source=event.source,
                            text=sentence,
                            score=score,
                            token_estimate=estimate_tokens(sentence),
                        )
                    )

        return tuple(sorted(snippets, key=lambda item: (item.score, -item.token_estimate), reverse=True)[:limit])

    def stats(self, capsule_id: str, *, compiled: CompiledMemory | None = None) -> dict:
        capsule = self.manager.get(capsule_id)
        raw_tokens = sum(estimate_tokens(event.text) for event in capsule.events)
        compiled_tokens = compiled.token_estimate if compiled else 0
        ratio = float("inf") if compiled_tokens == 0 else raw_tokens / compiled_tokens
        return {
            "capsule_id": capsule_id,
            "events": len(capsule.events),
            "active_facts": len(capsule.active_facts()),
            "raw_tokens": raw_tokens,
            "compiled_tokens": compiled_tokens,
            "compression_ratio": ratio,
        }

    def _rank_answer_facts(self, facts: tuple[Fact, ...], question: str) -> list[tuple[int, Fact]]:
        q = question.lower()
        q_terms = set(tokenize(q.replace("_", " ")))
        wanted: set[str] = set()

        if "board" in q or "device" in q or "hardware" in q or "target" in q:
            wanted.add("board")
        if "language" in q or "python" in q or "code" in q:
            wanted.add("language")
        if "pin" in q:
            wanted.update(fact.key for fact in facts if fact.key.startswith("physical_pin_"))
        if "voltage" in q:
            wanted.update(fact.key for fact in facts if "voltage" in fact.key)
        if "current" in q or "amps" in q or "amp" in q:
            wanted.update(fact.key for fact in facts if "current" in fact.key)
        if "color" in q or "colour" in q:
            wanted.add("color")
        if "avoid" in q or "not use" in q or "don't use" in q or "rejected" in q:
            wanted.add("do_not_use")
            wanted.add("rejected_or_previous")

        ranked: list[tuple[int, Fact]] = []
        for fact in facts:
            fact_terms = set(tokenize(f"{fact.scope} {fact.key} {fact.key.replace('_', ' ')} {fact.value}"))
            overlap = len(q_terms & fact_terms)
            key_match = 4 if fact.key in wanted else 0
            if ("voltage" in q and "voltage" in fact.key) or (("current" in q or "amp" in q) and "current" in fact.key):
                key_match += 8
            score = overlap + key_match
            if score > 0:
                ranked.append((score, fact))

        return sorted(ranked, key=lambda item: (item[0], item[1].confidence, item[1].observed_at), reverse=True)


def _sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _query_allows_stale_evidence(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("avoid", "do not use", "don't use", "rejected", "stale", "obsolete", "deprecated"))


def _skip_retrieved_sentence(sentence: str, *, include_stale: bool) -> bool:
    lowered = sentence.lower()
    if "/generated/filler_" in lowered or "generated/filler_" in lowered:
        return True
    if "copied stale note" in lowered or "no active decision" in lowered:
        return True
    if not include_stale and any(term in lowered for term in ("old docs mention", " is stale", " was rejected", "rejected_", "do not restore")):
        return True
    return False


def _clip_retrieved_sentence(sentence: str, *, limit: int = 360) -> str:
    compact = " ".join(sentence.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _synthesize_from_snippets(question: str, snippets: tuple[EvidenceSnippet, ...]) -> str:
    if not snippets:
        return ""
    q = question.lower()
    text = snippets[0].text

    if "color" in q or "colour" in q:
        match = re.search(r"\b(?:color|colour)[^.!?]*(?:is|:|=|with|ended with|picked|selected)\s+([A-Za-z][A-Za-z -]{2,50})", text, re.I)
        if match:
            return match.group(1).strip(" .,:;")
        match = re.search(r"\b(?:ended with|picked|selected|chose)\s+([^.!?;,]+)", text, re.I)
        if match and re.search(r"\b(blue|red|green|black|white|gray|grey|yellow|purple|orange|silver|gold)\b", match.group(1), re.I):
            return match.group(1).strip(" .,:;")

    if "voltage" in q:
        match = re.search(r"\b(?:voltage|pump voltage)\s*(?:is|:|=)?\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z]+)", text, re.I)
        if match:
            return match.group(1).replace(" ", "")

    if "current" in q or "amp" in q:
        match = re.search(r"\b(?:current|pump current)\s*(?:is|:|=)?\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z]+)", text, re.I)
        if match:
            return match.group(1).replace(" ", "")

    return ""
