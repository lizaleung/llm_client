from .base import BaseLLMClient
from .middleware.cache import CachedClient
from .middleware.cost import CostTracker
from .middleware.retry import RetryClient
from .providers.anthropic import AnthropicClient
from .providers.gemini import GeminiClient
from .providers.openai import OpenAIClient
from .types import LLMResponse, Message, Usage

__all__ = [
    "get_client",
    "BaseLLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "GeminiClient",
    "RetryClient",
    "CostTracker",
    "CachedClient",
    "Message",
    "Usage",
    "LLMResponse",
]

_PROVIDERS = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def get_client(provider: str, **kwargs) -> BaseLLMClient:
    """Create a provider client by name. Providers: 'anthropic', 'openai', 'gemini'."""
    try:
        return _PROVIDERS[provider](**kwargs)
    except KeyError:
        choices = ", ".join(_PROVIDERS)
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {choices}")
