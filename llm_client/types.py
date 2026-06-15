import warnings
from typing import Callable, Generator, Optional, Tuple

from pydantic import BaseModel

PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
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


class StreamResult:
    """Iterator of text chunks that also exposes usage after full consumption.

    Provider ``stream()`` methods return this so middleware (e.g. ``CostTracker``)
    can record token usage once the caller has drained the stream. ``model`` and
    ``usage`` are ``None`` until iteration completes, and remain ``None`` if the
    provider could not report usage.

    The wrapped generator function should ``yield`` text chunks and ``return`` a
    ``(model, Usage)`` tuple (or ``None``) as its final value.
    """

    def __init__(
        self,
        gen_fn: Callable[[], Generator[str, None, Optional[Tuple[str, "Usage"]]]],
    ):
        self._gen_fn = gen_fn
        self.model: Optional[str] = None
        self.usage: Optional[Usage] = None

    def __iter__(self) -> Generator[str, None, None]:
        result = yield from self._gen_fn()
        if result:
            self.model, self.usage = result
