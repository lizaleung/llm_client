from unittest.mock import MagicMock, patch

import pytest

from llm_client.providers.anthropic import AnthropicClient
from llm_client.providers.gemini import GeminiClient
from llm_client.providers.openai import OpenAIClient
from llm_client.types import LLMResponse, Message


@pytest.fixture
def messages():
    return [Message(role="user", content="Hello")]


class TestAnthropicClient:
    @patch("llm_client.providers.anthropic.anthropic.Anthropic")
    def test_complete_returns_llm_response(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Hi there!")]
        mock_msg.model = "claude-haiku-4-5-20251001"
        mock_msg.usage.input_tokens = 10
        mock_msg.usage.output_tokens = 5
        mock_sdk.messages.create.return_value = mock_msg

        client = AnthropicClient(model="claude-haiku-4-5-20251001")
        response = client.complete(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hi there!"
        assert response.model == "claude-haiku-4-5-20251001"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.cost_usd >= 0
        assert response.latency_ms >= 0
        assert response.cached is False

    @patch("llm_client.providers.anthropic.anthropic.Anthropic")
    def test_complete_passes_model_kwarg(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Hello")]
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_msg.usage.input_tokens = 5
        mock_msg.usage.output_tokens = 3
        mock_sdk.messages.create.return_value = mock_msg

        client = AnthropicClient()
        client.complete(messages, model="claude-sonnet-4-20250514")

        call_kwargs = mock_sdk.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @patch("llm_client.providers.anthropic.anthropic.Anthropic")
    def test_stream_yields_text(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_ctx.text_stream = iter(["Hello", " world"])
        mock_sdk.messages.stream.return_value = mock_stream_ctx

        client = AnthropicClient()
        chunks = list(client.stream(messages))

        assert chunks == ["Hello", " world"]

    @patch("llm_client.providers.anthropic.anthropic.Anthropic")
    def test_complete_computes_cost(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Hi")]
        mock_msg.model = "claude-haiku-4-5-20251001"
        mock_msg.usage.input_tokens = 1_000_000
        mock_msg.usage.output_tokens = 1_000_000
        mock_sdk.messages.create.return_value = mock_msg

        client = AnthropicClient(model="claude-haiku-4-5-20251001")
        response = client.complete(messages)

        # 1M input @ $0.80 + 1M output @ $4.00 = $4.80
        assert abs(response.usage.cost_usd - 4.80) < 0.0001

    @patch("llm_client.providers.anthropic.anthropic.Anthropic")
    def test_model_property(self, mock_cls):
        mock_cls.return_value = MagicMock()
        client = AnthropicClient(model="claude-haiku-4-5-20251001")
        assert client.model == "claude-haiku-4-5-20251001"


class TestOpenAIClient:
    @patch("llm_client.providers.openai.openai.OpenAI")
    def test_complete_returns_llm_response(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hello from GPT"
        mock_resp.model = "gpt-4o-mini"
        mock_resp.usage.prompt_tokens = 8
        mock_resp.usage.completion_tokens = 6
        mock_sdk.chat.completions.create.return_value = mock_resp

        client = OpenAIClient(model="gpt-4o-mini")
        response = client.complete(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello from GPT"
        assert response.model == "gpt-4o-mini"
        assert response.usage.input_tokens == 8
        assert response.usage.output_tokens == 6
        assert response.usage.cost_usd >= 0
        assert response.latency_ms >= 0
        assert response.cached is False

    @patch("llm_client.providers.openai.openai.OpenAI")
    def test_complete_passes_model_kwarg(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hi"
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 3
        mock_sdk.chat.completions.create.return_value = mock_resp

        client = OpenAIClient()
        client.complete(messages, model="gpt-4o")

        call_kwargs = mock_sdk.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    @patch("llm_client.providers.openai.openai.OpenAI")
    def test_stream_yields_text(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        def make_chunk(text):
            c = MagicMock()
            c.choices = [MagicMock()]
            c.choices[0].delta.content = text
            return c

        mock_sdk.chat.completions.create.return_value = iter(
            [make_chunk("Hello"), make_chunk(" world"), make_chunk(None)]
        )

        client = OpenAIClient()
        chunks = list(client.stream(messages))

        assert chunks == ["Hello", " world"]

    @patch("llm_client.providers.openai.openai.OpenAI")
    def test_complete_computes_cost(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hi"
        mock_resp.model = "gpt-4o-mini"
        mock_resp.usage.prompt_tokens = 1_000_000
        mock_resp.usage.completion_tokens = 1_000_000
        mock_sdk.chat.completions.create.return_value = mock_resp

        client = OpenAIClient(model="gpt-4o-mini")
        response = client.complete(messages)

        # 1M input @ $0.15 + 1M output @ $0.60 = $0.75
        assert abs(response.usage.cost_usd - 0.75) < 0.0001

    @patch("llm_client.providers.openai.openai.OpenAI")
    def test_complete_costs_dated_model_id(self, mock_cls, messages):
        # APIs return a dated/suffixed model ID that isn't a literal PRICING key.
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hi"
        mock_resp.model = "gpt-4o-2024-08-06"
        mock_resp.usage.prompt_tokens = 1_000_000
        mock_resp.usage.completion_tokens = 1_000_000
        mock_sdk.chat.completions.create.return_value = mock_resp

        client = OpenAIClient(model="gpt-4o")
        response = client.complete(messages)

        # Should resolve to gpt-4o pricing, not silently report $0.00.
        # 1M input @ $2.50 + 1M output @ $10.00 = $12.50
        assert abs(response.usage.cost_usd - 12.50) < 0.0001

    @patch("llm_client.providers.openai.openai.OpenAI")
    def test_model_property(self, mock_cls):
        mock_cls.return_value = MagicMock()
        client = OpenAIClient(model="gpt-4o-mini")
        assert client.model == "gpt-4o-mini"


class TestGeminiClient:
    @patch("llm_client.providers.gemini.genai.Client")
    def test_complete_returns_llm_response(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_resp = MagicMock()
        mock_resp.text = "Hello from Gemini"
        mock_resp.model_version = "gemini-2.5-flash"
        mock_resp.usage_metadata.prompt_token_count = 12
        mock_resp.usage_metadata.candidates_token_count = 7
        mock_sdk.models.generate_content.return_value = mock_resp

        client = GeminiClient(model="gemini-2.5-flash")
        response = client.complete(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello from Gemini"
        assert response.model == "gemini-2.5-flash"
        assert response.usage.input_tokens == 12
        assert response.usage.output_tokens == 7
        assert response.usage.cost_usd >= 0
        assert response.latency_ms >= 0
        assert response.cached is False

    @patch("llm_client.providers.gemini.genai.Client")
    def test_complete_passes_model(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_resp = MagicMock()
        mock_resp.text = "Hi"
        mock_resp.model_version = "gemini-2.5-pro"
        mock_resp.usage_metadata.prompt_token_count = 4
        mock_resp.usage_metadata.candidates_token_count = 2
        mock_sdk.models.generate_content.return_value = mock_resp

        client = GeminiClient()
        client.complete(messages, model="gemini-2.5-pro")

        call_kwargs = mock_sdk.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"

    @patch("llm_client.providers.gemini.genai.Client")
    def test_complete_computes_cost(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        mock_resp = MagicMock()
        mock_resp.text = "Hi"
        mock_resp.model_version = "gemini-2.5-flash"
        mock_resp.usage_metadata.prompt_token_count = 1_000_000
        mock_resp.usage_metadata.candidates_token_count = 1_000_000
        mock_sdk.models.generate_content.return_value = mock_resp

        client = GeminiClient(model="gemini-2.5-flash")
        response = client.complete(messages)

        # 1M input @ $0.30 + 1M output @ $2.50 = $2.80
        assert abs(response.usage.cost_usd - 2.80) < 0.0001

    @patch("llm_client.providers.gemini.genai.Client")
    def test_stream_yields_text(self, mock_cls, messages):
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk

        def make_chunk(text):
            c = MagicMock()
            c.text = text
            c.usage_metadata = None
            c.model_version = "gemini-2.5-flash"
            return c

        mock_sdk.models.generate_content_stream.return_value = iter(
            [make_chunk("Hello"), make_chunk(" world")]
        )

        client = GeminiClient()
        chunks = list(client.stream(messages))

        assert chunks == ["Hello", " world"]

    @patch("llm_client.providers.gemini.genai.Client")
    def test_model_property(self, mock_cls):
        mock_cls.return_value = MagicMock()
        client = GeminiClient(model="gemini-2.0-flash")
        assert client.model == "gemini-2.0-flash"


class TestMissingSdk:
    def test_anthropic_missing_sdk_raises_helpful_error(self):
        with patch("llm_client.providers.anthropic.anthropic", None):
            with pytest.raises(ImportError, match="pip install anthropic"):
                AnthropicClient()

    def test_openai_missing_sdk_raises_helpful_error(self):
        with patch("llm_client.providers.openai.openai", None):
            with pytest.raises(ImportError, match="pip install openai"):
                OpenAIClient()

    def test_gemini_missing_sdk_raises_helpful_error(self):
        with patch("llm_client.providers.gemini.genai", None):
            with pytest.raises(ImportError, match="pip install google-genai"):
                GeminiClient()
