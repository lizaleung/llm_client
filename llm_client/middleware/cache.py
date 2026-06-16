import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterator, List, Optional

from ..base import BaseLLMClient
from ..types import LLMResponse, Message

DEFAULT_DB_PATH = Path.home() / ".llm_client" / "cache.db"


def _cache_key(model: str, messages: List[Message], kwargs: dict) -> str:
    key_data = {
        "model": model,
        "messages": [m.model_dump() for m in messages],
        "kwargs": {k: v for k, v in sorted(kwargs.items())},
    }
    # default=str keeps key construction from raising on values json can't
    # serialize natively (e.g. enums, pydantic models passed as kwargs).
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()


class _SQLiteCache:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[LLMResponse]:
        cursor = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        return LLMResponse.model_validate_json(row[0]) if row else None

    def set(self, key: str, value: LLMResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, value.model_dump_json()),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()


class CachedClient(BaseLLMClient):
    def __init__(
        self,
        client: BaseLLMClient,
        enabled: Optional[bool] = None,
        db_path: Path = DEFAULT_DB_PATH,
    ):
        self._client = client
        if enabled is None:
            enabled = os.environ.get("CACHE_ENABLED", "").lower() in ("1", "true", "yes")
        self._enabled = enabled
        self._cache = _SQLiteCache(db_path) if self._enabled else None

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def enabled(self) -> bool:
        return self._enabled

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        if not self._enabled or self._cache is None:
            return self._client.complete(messages, **kwargs)

        effective_model = kwargs.get("model", self._client.model)
        key = _cache_key(effective_model, messages, kwargs)

        cached = self._cache.get(key)
        if cached is not None:
            cached.cached = True
            return cached

        response = self._client.complete(messages, **kwargs)
        self._cache.set(key, response)
        return response

    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        # Streaming responses are not cached: they are consumed lazily and a
        # partially-drained stream can't be safely stored or replayed.
        return self._client.stream(messages, **kwargs)

    def clear_cache(self) -> None:
        if self._cache:
            self._cache.clear()
