from __future__ import annotations

import json
import os
from typing import Any

import redis.asyncio as redis


class ResumeVersionStore:
    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
            await self._client.ping()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def save_version(self, thread_id: str, payload: dict[str, Any]) -> None:
        await self.connect()
        assert self._client is not None
        key = f"resume:versions:{thread_id}"
        await self._client.lpush(key, json.dumps(payload))
        await self._client.ltrim(key, 0, 9)

    async def list_versions(self, thread_id: str) -> list[dict[str, Any]]:
        await self.connect()
        assert self._client is not None
        key = f"resume:versions:{thread_id}"
        raw_items = await self._client.lrange(key, 0, 9)
        return [json.loads(item) for item in raw_items if item]

    async def save_latex(self, thread_id: str, latex: str) -> None:
        """Store the compiled LaTeX string keyed by thread_id (TTL 24 h)."""
        await self.connect()
        assert self._client is not None
        key = f"resume:latex:{thread_id}"
        await self._client.set(key, latex, ex=86400)

    async def get_latex(self, thread_id: str) -> str | None:
        """Retrieve the stored LaTeX string for the given thread_id."""
        await self.connect()
        assert self._client is not None
        key = f"resume:latex:{thread_id}"
        return await self._client.get(key)
