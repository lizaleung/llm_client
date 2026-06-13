import warnings
from typing import Optional

from pydantic import BaseModel

PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def _resolve_pricing(model: str) -> Optional[dict[str, float]]:
    """Look up pricing for a model ID, tolerating dated/suffixed variants.

    APIs frequently return a resolved ID like ``gpt-4o-2024-08-06`` or
    ``claude-sonnet-4-20250514`` that differs from the alias the caller
    requested. Try an exact match first, then the longest priced key that is a
    prefix of the model (so ``gpt-4o-2024-08-06`` matches ``gpt-4o``).
    """
    if model in PRICING:
        return PRICING[model]
    candidates = [key for key in PRICING if model.startswith(key)]
    if candidates:
        return PRICING[max(candidates, key=len)]
    return None


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    fallback_model: Optional[str] = None,
) -> float:
    pricing = _resolve_pricing(model)
    if pricing is None and fallback_model:
        pricing = _resolve_pricing(fallback_model)
    if pricing is None:
        warnings.warn(
            f"No pricing entry for model {model!r}"
            + (f" (fallback {fallback_model!r})" if fallback_model else "")
            + "; reporting cost as $0.00. Add it to PRICING to track cost.",
            stacklevel=2,
        )
        pricing = {"input": 0.0, "output": 0.0}
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


class Message(BaseModel):
    role: str
    content: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: Usage
    latency_ms: float
    cached: bool = False
