from .base import BaseLLMClient
from .middleware.cache import CachedClient
from .middleware.cost import CostTracker
from .middleware.retry import RetryClient
from .providers.anthropic import AnthropicClient
from .providers.openai import OpenAIClient
from .types import LLMResponse, Message, Usage

__all__ = [
    "get_client",
    "BaseLLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "RetryClient",
    "CostTracker",
    "CachedClient",
    "Message",
    "Usage",
    "LLMResponse",
]


def get_client(provider: str, **kwargs) -> BaseLLMClient:
    """Create a provider client by name. Providers: 'anthropic', 'openai'."""
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    if provider == "openai":
        return OpenAIClient(**kwargs)
    raise ValueError(f"Unknown provider '{provider}'. Choose from: anthropic, openai")
