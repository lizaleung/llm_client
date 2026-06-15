import time
from typing import List

try:
    import openai
except ImportError:  # pragma: no cover - exercised via instantiation error
    openai = None

from ..base import BaseLLMClient
from ..types import LLMResponse, Message, StreamResult, Usage, compute_cost

DEFAULT_MODEL = "gpt-4o"


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        if openai is None:
            raise ImportError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with: pip install openai"
            )
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

    def stream(self, messages: List[Message], **kwargs) -> StreamResult:
        model = kwargs.pop("model", self._model)

        def _gen():
            stream = self._client.chat.completions.create(
                model=model,
                messages=[m.model_dump() for m in messages],
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
            usage_obj = None
            actual_model = model
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage_obj = chunk_usage
                    actual_model = getattr(chunk, "model", model) or model
            try:
                if usage_obj is not None:
                    usage = Usage(
                        input_tokens=usage_obj.prompt_tokens,
                        output_tokens=usage_obj.completion_tokens,
                        cost_usd=compute_cost(
                            actual_model,
                            usage_obj.prompt_tokens,
                            usage_obj.completion_tokens,
                            fallback_model=model,
                        ),
                    )
                    return actual_model, usage
            except Exception:
                return None
            return None

        return StreamResult(_gen)
