from pydantic import BaseModel

PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
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
