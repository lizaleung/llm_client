import time
from typing import Iterator, List

import anthropic

from ..base import BaseLLMClient
from ..types import LLMResponse, Message, Usage, compute_cost

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 1024


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        system = kwargs.pop("system", None)

        create_kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[m.model_dump() for m in messages],
            **kwargs,
        )
        if system:
            create_kwargs["system"] = system

        start = time.perf_counter()
        response = self._client.messages.create(**create_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=compute_cost(response.model, input_tokens, output_tokens, fallback_model=model),
            ),
            latency_ms=latency_ms,
        )

    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        system = kwargs.pop("system", None)

        stream_kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[m.model_dump() for m in messages],
            **kwargs,
        )
        if system:
            stream_kwargs["system"] = system

        with self._client.messages.stream(**stream_kwargs) as stream:
            yield from stream.text_stream
