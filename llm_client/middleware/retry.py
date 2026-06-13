from typing import Callable, Iterator, List

from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..base import BaseLLMClient
from ..types import LLMResponse, Message


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    if "RateLimitError" in name or "InternalServerError" in name:
        return True
    status = getattr(exc, "status_code", None)
    return status in (429, 500, 502, 503)


class RetryClient(BaseLLMClient):
    def __init__(
        self,
        client: BaseLLMClient,
        max_attempts: int = 3,
        wait_min: float = 4.0,
        wait_max: float = 60.0,
        _wait: Callable | None = None,
    ):
        self._client = client
        self._max_attempts = max_attempts
        self._wait = _wait or wait_exponential(multiplier=1, min=wait_min, max=wait_max)

    @property
    def model(self) -> str:
        return self._client.model

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        for attempt in Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                return self._client.complete(messages, **kwargs)

    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        for attempt in Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                return self._client.stream(messages, **kwargs)
