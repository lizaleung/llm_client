from pathlib import Path
from typing import Iterator, List
from unittest.mock import MagicMock

import pytest
from tenacity import wait_none

from llm_client.base import BaseLLMClient
from llm_client.middleware.cache import CachedClient
from llm_client.middleware.cost import CostTracker
from llm_client.middleware.retry import RetryClient
from llm_client.types import LLMResponse, Message, Usage


def make_response(content="Hello", model="gpt-4o-mini", input_tokens=10, output_tokens=5) -> LLMResponse:
    return LLMResponse(
        content=content,
        model=model,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.000002,
        ),
        latency_ms=50.0,
    )


class _FakeClient(BaseLLMClient):
    """Concrete BaseLLMClient backed by a callable for testing."""

    def __init__(self, fn, model_name="test-model"):
        self._fn = fn
        self._model = model_name

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        return self._fn(messages, **kwargs)

    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        yield "chunk"


@pytest.fixture
def messages():
    return [Message(role="user", content="Hi")]


# ---------------------------------------------------------------------------
# RetryClient
# ---------------------------------------------------------------------------


class TestRetryClient:
    def test_passes_through_on_success(self, messages):
        expected = make_response()
        client = _FakeClient(lambda *a, **kw: expected)
        retry = RetryClient(client, _wait=wait_none())

        response = retry.complete(messages)
        assert response is expected

    def test_retries_on_rate_limit_error(self, messages):
        call_count = 0
        expected = make_response()

        class _FakeRateLimitError(Exception):
            pass

        def flaky(msgs, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _FakeRateLimitError("rate limit")
            return expected

        client = _FakeClient(flaky)
        retry = RetryClient(client, max_attempts=3, _wait=wait_none())

        response = retry.complete(messages)
        assert response is expected
        assert call_count == 3

    def test_raises_after_max_attempts(self, messages):
        class _FakeRateLimitError(Exception):
            pass

        client = _FakeClient(lambda *a, **kw: (_ for _ in ()).throw(_FakeRateLimitError()))
        retry = RetryClient(client, max_attempts=2, _wait=wait_none())

        with pytest.raises(_FakeRateLimitError):
            retry.complete(messages)

    def test_does_not_retry_non_retryable(self, messages):
        call_count = 0

        def boom(msgs, **kw):
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        client = _FakeClient(boom)
        retry = RetryClient(client, max_attempts=3, _wait=wait_none())

        with pytest.raises(ValueError):
            retry.complete(messages)

        assert call_count == 1

    def test_delegates_model(self):
        client = _FakeClient(lambda *a, **kw: make_response(), model_name="my-model")
        retry = RetryClient(client)
        assert retry.model == "my-model"


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_records_response(self, messages):
        resp = make_response(input_tokens=100, output_tokens=50)
        client = _FakeClient(lambda *a, **kw: resp)
        tracker = CostTracker(client)

        tracker.complete(messages)

        assert tracker.call_count == 1
        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert tracker.total_cost_usd == resp.usage.cost_usd

    def test_accumulates_across_calls(self, messages):
        client = _FakeClient(lambda *a, **kw: make_response(input_tokens=10, output_tokens=5))
        tracker = CostTracker(client)

        tracker.complete(messages)
        tracker.complete(messages)

        assert tracker.call_count == 2
        assert tracker.total_input_tokens == 20
        assert tracker.total_output_tokens == 10

    def test_reset_clears_records(self, messages):
        client = _FakeClient(lambda *a, **kw: make_response())
        tracker = CostTracker(client)
        tracker.complete(messages)
        tracker.reset()

        assert tracker.call_count == 0
        assert tracker.total_cost_usd == 0.0

    def test_print_summary_empty(self, capsys):
        client = _FakeClient(lambda *a, **kw: make_response())
        tracker = CostTracker(client)
        tracker.print_summary()  # should not raise

    def test_print_summary_with_records(self, messages):
        client = _FakeClient(lambda *a, **kw: make_response())
        tracker = CostTracker(client)
        tracker.complete(messages)
        tracker.print_summary()  # should not raise

    def test_returns_response_unchanged(self, messages):
        resp = make_response(content="unique content")
        client = _FakeClient(lambda *a, **kw: resp)
        tracker = CostTracker(client)

        result = tracker.complete(messages)
        assert result is resp

    def test_delegates_model(self):
        client = _FakeClient(lambda *a, **kw: make_response(), model_name="tracker-model")
        tracker = CostTracker(client)
        assert tracker.model == "tracker-model"


# ---------------------------------------------------------------------------
# CachedClient
# ---------------------------------------------------------------------------


class TestCachedClient:
    def test_cache_disabled_passes_through(self, messages, tmp_path):
        call_count = 0

        def fn(msgs, **kw):
            nonlocal call_count
            call_count += 1
            return make_response()

        client = _FakeClient(fn)
        cached = CachedClient(client, enabled=False, db_path=tmp_path / "c.db")

        cached.complete(messages)
        cached.complete(messages)

        assert call_count == 2

    def test_cache_hit_on_second_call(self, messages, tmp_path):
        call_count = 0

        def fn(msgs, **kw):
            nonlocal call_count
            call_count += 1
            return make_response()

        client = _FakeClient(fn, model_name="gpt-4o-mini")
        cached = CachedClient(client, enabled=True, db_path=tmp_path / "c.db")

        r1 = cached.complete(messages, model="gpt-4o-mini")
        r2 = cached.complete(messages, model="gpt-4o-mini")

        assert call_count == 1
        assert not r1.cached
        assert r2.cached

    def test_different_messages_not_cached(self, tmp_path):
        call_count = 0

        def fn(msgs, **kw):
            nonlocal call_count
            call_count += 1
            return make_response()

        client = _FakeClient(fn, model_name="gpt-4o-mini")
        cached = CachedClient(client, enabled=True, db_path=tmp_path / "c.db")

        cached.complete([Message(role="user", content="Hello")], model="gpt-4o-mini")
        cached.complete([Message(role="user", content="Goodbye")], model="gpt-4o-mini")

        assert call_count == 2

    def test_clear_cache_evicts_entries(self, messages, tmp_path):
        call_count = 0

        def fn(msgs, **kw):
            nonlocal call_count
            call_count += 1
            return make_response()

        client = _FakeClient(fn, model_name="gpt-4o-mini")
        cached = CachedClient(client, enabled=True, db_path=tmp_path / "c.db")

        cached.complete(messages, model="gpt-4o-mini")
        cached.clear_cache()
        cached.complete(messages, model="gpt-4o-mini")

        assert call_count == 2

    def test_delegates_model(self, tmp_path):
        client = _FakeClient(lambda *a, **kw: make_response(), model_name="cache-model")
        cached = CachedClient(client, enabled=False, db_path=tmp_path / "c.db")
        assert cached.model == "cache-model"
