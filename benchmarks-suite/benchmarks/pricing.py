"""Model pricing and cost estimation for benchmark runs."""

from __future__ import annotations

# Prices in USD per 1M tokens.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
}

# Adapters whose ingest path involves LLM calls and their approximate
# token overhead ratios relative to the raw event text.
_INGEST_LLM_OVERHEAD: dict[str, dict[str, float]] = {
    "mem0_real": {"input_ratio": 1.5, "output_ratio": 0.3},
    "mem0_real_async": {"input_ratio": 1.5, "output_ratio": 0.3},
    "mem0_real_fast": {"input_ratio": 0.0, "output_ratio": 0.0},
    "langgraph_real": {"input_ratio": 1.2, "output_ratio": 0.2},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICING.get(model)
    if prices is None:
        for key in PRICING:
            if model.startswith(key.rsplit("-", 1)[0]):
                prices = PRICING[key]
                break
    if prices is None:
        prices = {"input": 0.0, "output": 0.0}
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def estimate_ingest_cost(
    adapter_name: str,
    event_tokens: int,
    model: str = "gpt-4.1-mini",
) -> float:
    overhead = _INGEST_LLM_OVERHEAD.get(adapter_name)
    if not overhead:
        return 0.0
    in_tok = int(event_tokens * overhead["input_ratio"])
    out_tok = int(event_tokens * overhead["output_ratio"])
    return cost_usd(model, in_tok, out_tok)


def cost_breakdown(
    model: str,
    *,
    ingest_tokens: tuple[int, int] = (0, 0),
    prepare_tokens: tuple[int, int] = (0, 0),
    agent_tokens: tuple[int, int] = (0, 0),
) -> dict[str, float]:
    ingest = cost_usd(model, *ingest_tokens)
    prepare = cost_usd(model, *prepare_tokens)
    agent = cost_usd(model, *agent_tokens)
    return {
        "ingest_cost_usd": round(ingest, 6),
        "prepare_cost_usd": round(prepare, 6),
        "agent_cost_usd": round(agent, 6),
        "total_cost_usd": round(ingest + prepare + agent, 6),
    }
