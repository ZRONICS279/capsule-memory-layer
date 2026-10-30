"""Token and input-cost savings helpers for prepared memory packets."""

from __future__ import annotations

from dataclasses import dataclass

from .agentic import MemoryPacket


@dataclass(frozen=True)
class SavingsEstimate:
    raw_tokens_per_turn: int
    memory_tokens_per_turn: int
    tokens_saved_per_turn: int
    token_reduction: float
    turns: int
    raw_tokens_total: int
    memory_tokens_total: int
    tokens_saved_total: int
    input_price_per_1m: float
    baseline_input_cost: float
    memory_input_cost: float
    input_cost_saved: float

    def to_json(self) -> dict:
        return {
            "raw_tokens_per_turn": self.raw_tokens_per_turn,
            "memory_tokens_per_turn": self.memory_tokens_per_turn,
            "tokens_saved_per_turn": self.tokens_saved_per_turn,
            "token_reduction": self.token_reduction,
            "turns": self.turns,
            "raw_tokens_total": self.raw_tokens_total,
            "memory_tokens_total": self.memory_tokens_total,
            "tokens_saved_total": self.tokens_saved_total,
            "input_price_per_1m": self.input_price_per_1m,
            "baseline_input_cost": self.baseline_input_cost,
            "memory_input_cost": self.memory_input_cost,
            "input_cost_saved": self.input_cost_saved,
        }


def estimate_input_savings(packet: MemoryPacket, *, input_price_per_1m: float, turns: int = 1) -> SavingsEstimate:
    """Compare repeated full-context input against a prepared memory packet."""

    if input_price_per_1m < 0:
        raise ValueError("input_price_per_1m must be non-negative")
    if turns <= 0:
        raise ValueError("turns must be positive")

    raw_tokens = packet.raw_tokens
    memory_tokens = packet.memory_tokens
    saved = max(0, raw_tokens - memory_tokens)
    raw_total = raw_tokens * turns
    memory_total = memory_tokens * turns
    saved_total = saved * turns

    baseline_cost = raw_total * input_price_per_1m / 1_000_000
    memory_cost = memory_total * input_price_per_1m / 1_000_000

    return SavingsEstimate(
        raw_tokens_per_turn=raw_tokens,
        memory_tokens_per_turn=memory_tokens,
        tokens_saved_per_turn=saved,
        token_reduction=0.0 if raw_tokens == 0 else saved / raw_tokens,
        turns=turns,
        raw_tokens_total=raw_total,
        memory_tokens_total=memory_total,
        tokens_saved_total=saved_total,
        input_price_per_1m=input_price_per_1m,
        baseline_input_cost=baseline_cost,
        memory_input_cost=memory_cost,
        input_cost_saved=baseline_cost - memory_cost,
    )
