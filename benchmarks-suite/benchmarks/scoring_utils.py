"""Shared benchmark scoring helpers."""

from __future__ import annotations

import re


_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def contains_fact(haystack: str, needle: str) -> bool:
    """Return true when a scored fact appears with harmless separator changes.

    Exact case-insensitive matching is checked first. The fallback collapses
    punctuation, whitespace, hyphens, and underscores so natural-language facts
    like "memory loss" still match "memory-loss" without making recall depend
    on cosmetic formatting.
    """

    folded_haystack = haystack.casefold()
    folded_needle = needle.casefold()
    if folded_needle in folded_haystack:
        return True
    normalized_needle = normalize_for_fact_match(needle)
    if not normalized_needle:
        return False
    return normalized_needle in normalize_for_fact_match(haystack)


def normalize_for_fact_match(text: str) -> str:
    split = _CAMEL_BOUNDARY_RE.sub(" ", text)
    return _SEPARATOR_RE.sub(" ", split.casefold()).strip()
