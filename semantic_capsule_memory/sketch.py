"""Deterministic dense-ish context sketches.

This is not a trained soft prompt. It is the practical phase-1 stand-in:
compact vectors that route and compare capsule state without injecting the
original text into an LLM prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import sqrt
import re


TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+#-]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _hash_int(value: str, *, digest_size: int = 8) -> int:
    return int.from_bytes(blake2b(value.encode("utf-8"), digest_size=digest_size).digest(), "big")


@dataclass(frozen=True)
class VectorSketch:
    """A small signed feature sketch.

    Shape defaults to 32 vectors x 64 dims: large enough to behave like a
    capsule block, tiny enough to serialize and inspect.
    """

    slots: int = 32
    dims: int = 64
    values: tuple[tuple[float, ...], ...] = ()

    @classmethod
    def empty(cls, slots: int = 32, dims: int = 64) -> "VectorSketch":
        return cls(slots=slots, dims=dims, values=tuple(tuple(0.0 for _ in range(dims)) for _ in range(slots)))

    @classmethod
    def from_text(cls, text: str, slots: int = 32, dims: int = 64) -> "VectorSketch":
        sketch = cls.empty(slots=slots, dims=dims)
        return sketch.add_text(text)

    def add_text(self, text: str, weight: float = 1.0) -> "VectorSketch":
        rows = [list(row) for row in self.values] if self.values else [
            [0.0 for _ in range(self.dims)] for _ in range(self.slots)
        ]
        tokens = tokenize(text)
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            h = _hash_int(feature)
            slot = h % self.slots
            dim = (h >> 8) % self.dims
            sign = 1.0 if ((h >> 16) & 1) else -1.0
            rows[slot][dim] += sign * weight
        return VectorSketch(self.slots, self.dims, tuple(tuple(row) for row in rows)).normalized()

    def normalized(self) -> "VectorSketch":
        rows: list[tuple[float, ...]] = []
        for row in self.values:
            norm = sqrt(sum(v * v for v in row))
            if norm == 0:
                rows.append(tuple(row))
            else:
                rows.append(tuple(round(v / norm, 6) for v in row))
        return VectorSketch(self.slots, self.dims, tuple(rows))

    def similarity(self, other: "VectorSketch") -> float:
        if self.slots != other.slots or self.dims != other.dims:
            raise ValueError("Sketch shapes must match")
        total = 0.0
        for left, right in zip(self.values, other.values):
            total += sum(a * b for a, b in zip(left, right))
        return total / self.slots

    def compact_digest(self) -> str:
        flattened = ",".join(f"{value:.3f}" for row in self.values for value in row)
        return blake2b(flattened.encode("utf-8"), digest_size=12).hexdigest()

    def to_json(self) -> dict:
        return {"slots": self.slots, "dims": self.dims, "values": self.values}

    @classmethod
    def from_json(cls, data: dict) -> "VectorSketch":
        return cls(
            slots=int(data["slots"]),
            dims=int(data["dims"]),
            values=tuple(tuple(float(value) for value in row) for row in data["values"]),
        )

