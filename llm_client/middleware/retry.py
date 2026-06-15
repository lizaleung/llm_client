from typing import Callable, Iterator, List

from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..base import BaseLLMClient
from ..types import LLMResponse, Message


# Exception class-name fragments that indicate a transient failure across the
# anthropic / openai SDKs (and httpx underneath them). Matched as substrings so
# vendor-prefixed variants are covered without importing the SDKs here.
_RETRYABLE_NAME_FRAGMENTS = (
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "OverloadedError",
    "APIConnectionError",
    "APITimeoutError",
    "ConnectTimeout",
    "ReadTimeout",
)

# HTTP status codes worth retrying: 429 (rate limit), 5xx (server errors),
# 529 (Anthropic "overloaded").
_RETRYABLE_STATUS = (429, 500, 502, 503, 504, 529)


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in _RETRYABLE_STATUS:
        return True
    name = type(exc).__name__
    if any(fragment in name for fragment in _RETRYABLE_NAME_FRAGMENTS):
        return True
    # Network-level failures from the stdlib / httpx.
    return isinstance(exc, (ConnectionError, TimeoutError))


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
        # Unreachable: Retrying either returns from the block above or re-raises
        # the last exception (reraise=True). Guards the implicit-None fall-through.
        raise RuntimeError("RetryClient.complete exhausted retries without a result")

    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        # Only stream *establishment* (opening the connection and producing the
        # first chunk) can be safely retried — once chunks have been yielded a
        # mid-stream failure can't be replayed without duplicating output.
        def _establish():
            for attempt in Retrying(
                stop=stop_after_attempt(self._max_attempts),
                wait=self._wait,
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    iterator = iter(self._client.stream(messages, **kwargs))
                    try:
                        first = next(iterator)
                    except StopIteration:
                        return iterator, None, True
                    return iterator, first, False
            # Unreachable: see complete().
            raise RuntimeError("RetryClient.stream exhausted retries without a result")

        def _generator():
            iterator, first, empty = _establish()
            if not empty:
                yield first
                yield from iterator

        return _generator()
