from __future__ import annotations

from dataclasses import asdict, dataclass


DEFAULT_MODEL_PRICING_USD_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4.1": {"input": 2, "output": 8},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
}


@dataclass(slots=True)
class SummaryMetrics:
    duration_ms: float
    input_text_length: int
    output_text_length: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict | None = None,
) -> float:
    pricing_map = pricing or DEFAULT_MODEL_PRICING_USD_PER_1M
    normalized_model = model.split(":", 1)[-1].split("-", 3)
    normalized_model = "-".join(normalized_model[:3]) if len(normalized_model) > 3 else "-".join(normalized_model)

    model_pricing = pricing_map.get(model) or pricing_map.get(normalized_model)
    if not model_pricing:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    return round(input_cost + output_cost, 8)
