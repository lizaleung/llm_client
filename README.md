# llm-client

One library, every LLM provider — swap Anthropic for OpenAI with a single line change.

## Quickstart

```python
from llm_client import get_client
from llm_client.types import Message

client = get_client("anthropic", model="claude-haiku-4-5-20251001")
response = client.complete([Message(role="user", content="Hello!")])

print(response.content)           # "Hello! How can I help you?"
print(f"${response.usage.cost_usd:.6f}")  # "$0.000003"
print(f"{response.latency_ms:.0f}ms")     # "312ms"
```

Switch providers instantly:

```python
client = get_client("openai", model="gpt-4o-mini")
```

## Middleware

Stack middleware using the decorator pattern:

```python
from llm_client.middleware.cost import CostTracker
from llm_client.middleware.retry import RetryClient
from llm_client.middleware.cache import CachedClient

base    = get_client("openai", model="gpt-4o-mini")
tracked = CostTracker(base)
client  = CachedClient(RetryClient(tracked))

response = client.complete([Message(role="user", content="Hi")])
tracked.print_summary()
```

### Middleware pipeline

```
Request
   │
   ▼
┌──────────────┐
│ CachedClient │──► Cache hit → return immediately (cached=True)
└──────┬───────┘
       │ Cache miss
       ▼
┌──────────────┐
│  RetryClient │──► 429 / 500 → exponential backoff, up to 3 attempts
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  CostTracker │──► Records tokens, cost, latency per call
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Provider Client │──► Anthropic / OpenAI API
└──────────────────┘
```

### Cost tracker output

```
            LLM Cost Summary
┏━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Model        ┃ Calls ┃   In  ┃   Out  ┃ Cost (USD) ┃ Avg Lat (ms) ┃
┡━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ gpt-4o-mini  │     3 │   312 │    148 │ $0.000135  │        287.4 │
│ gpt-4o       │     1 │    89 │     42 │ $0.000643  │        521.0 │
├──────────────┼───────┼───────┼────────┼────────────┼──────────────┤
│ TOTAL        │     4 │   401 │    190 │ $0.000778  │              │
└──────────────┴───────┴───────┴────────┴────────────┴──────────────┘
```

## Caching

SQLite cache at `~/.llm_client/cache.db`. Enable with the environment variable:

```bash
CACHE_ENABLED=1 python your_script.py
```

Or programmatically:

```python
client = CachedClient(base, enabled=True)
```

## Supported models and pricing

| Model                        | Input ($/M) | Output ($/M) |
|------------------------------|-------------|--------------|
| claude-sonnet-4-20250514     | $3.00       | $15.00       |
| claude-haiku-4-5-20251001    | $0.80       | $4.00        |
| gpt-4o                       | $2.50       | $10.00       |
| gpt-4o-mini                  | $0.15       | $0.60        |

## Installation

```bash
pip install llm-client
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
