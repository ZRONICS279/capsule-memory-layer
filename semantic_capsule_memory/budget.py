"""Token accounting.

Every budget, reduction metric, and savings number in this package routes
through :func:`count_tokens` (aliased as :func:`estimate_tokens`). Making this
accurate is what turns the project's efficiency claims from "directional
heuristic" into "honest, verifiable measurement".

Backend selection:

* If ``tiktoken`` is installed, real BPE token counts are used. ``cl100k_base``
  (GPT-4 / 4o / 3.5) is the default encoding and a strong, model-agnostic proxy
  for modern LLM tokenizers. Callers can target their own model/encoding with
  :func:`set_default_encoding` or the per-call ``model=`` argument.
* If ``tiktoken`` is unavailable, a dependency-free heuristic is used that is
  noticeably closer to real BPE than a flat ``len/4`` (it accounts for word and
  punctuation boundaries). The active backend is always inspectable via
  :func:`tokenizer_backend` so reduction numbers can be reproduced.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

# cl100k_base is the BPE used by GPT-4/4o/3.5-turbo and a solid cross-model
# proxy. o200k_base covers GPT-4o / o-series. Switch with set_default_encoding().
_DEFAULT_ENCODING = "cl100k_base"
_active_encoding = _DEFAULT_ENCODING

# Common model name -> encoding hints so users can pass their target model name
# directly. Anything not listed is resolved by tiktoken.encoding_for_model().
_MODEL_ALIASES = {
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o3": "o200k_base",
}

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=16)
def _load_encoder(name: str):
    """Return a cached tiktoken encoder for ``name`` or ``None`` if unavailable."""

    try:
        import tiktoken
    except Exception:  # pragma: no cover - import guard
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        pass
    try:
        return tiktoken.encoding_for_model(name)
    except Exception:
        return None


def _resolve_encoding(model: Optional[str]) -> str:
    if not model:
        return _active_encoding
    return _MODEL_ALIASES.get(model, model)


def set_default_encoding(model_or_encoding: str) -> None:
    """Set the encoding used for all token counts that don't pass ``model=``.

    Accepts either a tiktoken encoding name (e.g. ``"o200k_base"``) or a model
    name (e.g. ``"gpt-4o"``). Affects budgets, reductions, and savings globally,
    so an agent builder can measure against the exact model they deploy on.
    """

    global _active_encoding
    _active_encoding = _MODEL_ALIASES.get(model_or_encoding, model_or_encoding)


def tokenizer_backend(model: Optional[str] = None) -> str:
    """Report the active counting backend for verifiability/reproducibility.

    Returns ``"tiktoken:<encoding>"`` when real BPE counts are in use, or
    ``"heuristic"`` when the dependency-free fallback is active.
    """

    name = _resolve_encoding(model)
    return f"tiktoken:{name}" if _load_encoder(name) is not None else "heuristic"


def _heuristic_tokens(text: str) -> int:
    """Dependency-free token estimate that respects word/punctuation boundaries.

    Closer to real BPE than ``len/4``: most English words are a single token,
    punctuation tends to split, and long tokens get charged extra by length.
    """

    pieces = _WORD_RE.findall(text)
    if not pieces:
        return max(1, round(len(text) / 4))
    total = 0
    for piece in pieces:
        # Subword splitting: BPE rarely keeps long runs as one token.
        total += 1 + (len(piece) - 1) // 6
    return max(1, total)


def count_tokens(text: str, *, model: Optional[str] = None) -> int:
    """Return the token count for ``text`` using the best available backend."""

    if not text:
        return 0
    encoder = _load_encoder(_resolve_encoding(model))
    if encoder is not None:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception:  # pragma: no cover - defensive
            pass
    return _heuristic_tokens(text)


# Backward-compatible alias. Existing callers across the package import this name.
def estimate_tokens(text: str, *, model: Optional[str] = None) -> int:
    """Alias for :func:`count_tokens` (kept for API stability)."""

    return count_tokens(text, model=model)

