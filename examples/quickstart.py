"""
quickstart.py — llm-client basic usage examples.

Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY before running.
"""
from llm_client import get_client
from llm_client.middleware.cache import CachedClient
from llm_client.middleware.cost import CostTracker
from llm_client.middleware.retry import RetryClient
from llm_client.types import Message

# ---------------------------------------------------------------------------
# 1. Basic completion
# ---------------------------------------------------------------------------

client = get_client("anthropic", model="claude-haiku-4-5-20251001")

response = client.complete([Message(role="user", content="What is 2 + 2?")])
print(response.content)
print(f"Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")
print(f"Cost: ${response.usage.cost_usd:.6f}  Latency: {response.latency_ms:.0f}ms")

# ---------------------------------------------------------------------------
# 2. Streaming
# ---------------------------------------------------------------------------

print("\n--- Streaming ---")
for chunk in client.stream([Message(role="user", content="Count to 5.")]):
    print(chunk, end="", flush=True)
print()

# ---------------------------------------------------------------------------
# 3. Middleware composition
#
#   Request
#      │
#      ▼
#   CachedClient  ──► cache hit? return immediately
#      │
#      ▼
#   RetryClient   ──► 429/500? exponential backoff & retry
#      │
#      ▼
#   CostTracker   ──► record tokens, cost, latency
#      │
#      ▼
#   OpenAIClient  ──► actual API call
#      │
#      ▼
#   Response
# ---------------------------------------------------------------------------

base = get_client("openai", model="gpt-4o-mini")
tracked = CostTracker(base)
with_retry = RetryClient(tracked)
# enabled=True turns the cache on regardless of the CACHE_ENABLED env var.
with_cache = CachedClient(with_retry, enabled=True)

print("\n--- With middleware ---")
r = with_cache.complete([Message(role="user", content="Say hello in one word.")])
print(r.content)

# second identical call is served from the cache (no API call)
r2 = with_cache.complete([Message(role="user", content="Say hello in one word.")])
print(f"Cached: {r2.cached}")

tracked.print_summary()

# ---------------------------------------------------------------------------
# 4. Switching providers — same interface, one line changes
# ---------------------------------------------------------------------------

print("\n--- Gemini ---")
gemini = get_client("gemini", model="gemini-2.5-flash")
g = gemini.complete([Message(role="user", content="What is 2 + 2?")])
print(g.content)
print(f"Cost: ${g.usage.cost_usd:.6f}  Latency: {g.latency_ms:.0f}ms")
