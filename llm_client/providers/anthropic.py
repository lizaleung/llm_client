import time
from typing import List

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised via instantiation error
    anthropic = None

from ..base import BaseLLMClient
from ..types import LLMResponse, Message, StreamResult, Usage, compute_cost

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 1024


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install it with: pip install anthropic"
            )
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

    def stream(self, messages: List[Message], **kwargs) -> StreamResult:
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

        def _gen():
            with self._client.messages.stream(**stream_kwargs) as stream:
                yield from stream.text_stream
                try:
                    final = stream.get_final_message()
                    usage = Usage(
                        input_tokens=final.usage.input_tokens,
                        output_tokens=final.usage.output_tokens,
                        cost_usd=compute_cost(
                            final.model,
                            final.usage.input_tokens,
                            final.usage.output_tokens,
                            fallback_model=model,
                        ),
                    )
                    return final.model, usage
                except Exception:
                    return None

        return StreamResult(_gen)
