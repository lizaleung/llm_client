import time
from typing import Iterator, List

import openai

from ..base import BaseLLMClient
from ..types import LLMResponse, Message, Usage, compute_cost

DEFAULT_MODEL = "gpt-4o"


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._model = model
        self._client = openai.OpenAI(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", self._model)

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=[m.model_dump() for m in messages],
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        actual_model = response.model

        return LLMResponse(
            content=response.choices[0].message.content,
            model=actual_model,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=compute_cost(actual_model, input_tokens, output_tokens, fallback_model=model),
            ),
            latency_ms=latency_ms,
        )

    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        model = kwargs.pop("model", self._model)

        stream = self._client.chat.completions.create(
            model=model,
            messages=[m.model_dump() for m in messages],
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
