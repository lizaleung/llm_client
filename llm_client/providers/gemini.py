import time
from typing import List, Optional

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised via instantiation error
    genai = None
    genai_types = None

from ..base import BaseLLMClient
from ..types import LLMResponse, Message, StreamResult, Usage, compute_cost

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_TOKENS = 1024

# Gemini uses "model" for the assistant turn; everything else maps to "user".
_ROLE_MAP = {"assistant": "model", "model": "model"}


def _to_contents(messages: List[Message]) -> tuple[list, Optional[str]]:
    """Split messages into Gemini ``contents`` and an optional system instruction.

    ``system`` messages have no place in ``contents``; their text is collected
    into a single system instruction string instead.
    """
    contents = []
    system_parts: list[str] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
            continue
        role = _ROLE_MAP.get(m.role, "user")
        contents.append({"role": role, "parts": [{"text": m.content}]})
    system = "\n".join(system_parts) if system_parts else None
    return contents, system


def _build_config(max_tokens: int, system: Optional[str], extra: dict):
    return genai_types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        system_instruction=system,
        **extra,
    )


def _usage_from(metadata, model: str, requested_model: str) -> Usage:
    input_tokens = metadata.prompt_token_count or 0
    output_tokens = metadata.candidates_token_count or 0
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=compute_cost(model, input_tokens, output_tokens, fallback_model=requested_model),
    )


class GeminiClient(BaseLLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        if genai is None:
            raise ImportError(
                "The 'google-genai' package is required for GeminiClient. "
                "Install it with: pip install google-genai"
            )
        self._model = model
        self._client = genai.Client(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        system = kwargs.pop("system", None)

        contents, system_from_messages = _to_contents(messages)
        config = _build_config(max_tokens, system or system_from_messages, kwargs)

        start = time.perf_counter()
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        actual_model = getattr(response, "model_version", None) or model

        return LLMResponse(
            content=response.text,
            model=actual_model,
            usage=_usage_from(response.usage_metadata, actual_model, model),
            latency_ms=latency_ms,
        )

    def stream(self, messages: List[Message], **kwargs) -> StreamResult:
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        system = kwargs.pop("system", None)

        contents, system_from_messages = _to_contents(messages)
        config = _build_config(max_tokens, system or system_from_messages, kwargs)

        def _gen():
            last_metadata = None
            actual_model = model
            for chunk in self._client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                if getattr(chunk, "text", None):
                    yield chunk.text
                if getattr(chunk, "usage_metadata", None) is not None:
                    last_metadata = chunk.usage_metadata
                actual_model = getattr(chunk, "model_version", None) or actual_model
            try:
                if last_metadata is not None:
                    return actual_model, _usage_from(last_metadata, actual_model, model)
            except Exception:
                return None
            return None

        return StreamResult(_gen)
