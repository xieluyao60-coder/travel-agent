from __future__ import annotations

import json
from collections import defaultdict

from redis.asyncio import Redis


class SessionStore:
    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        history_limit: int,
        profile_ttl_seconds: int | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._ttl_seconds = ttl_seconds
        self._profile_ttl_seconds = profile_ttl_seconds if profile_ttl_seconds is not None else ttl_seconds
        self._history_limit = history_limit
        self._redis: Redis | None = None
        self._memory: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._pending_memory: dict[str, dict[str, object]] = {}
        self._profile_memory: dict[str, dict[str, object]] = {}

    async def connect(self) -> None:
        try:
            redis = Redis.from_url(self._redis_url, decode_responses=True)
            await redis.ping()
            self._redis = redis
        except Exception:
            self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        key = self._make_key(session_id)
        if self._redis is None:
            return list(self._memory.get(key, []))[-self._history_limit :]

        items = await self._redis.lrange(key, -self._history_limit, -1)
        return [json.loads(item) for item in items]

    async def append(self, session_id: str, role: str, content: str) -> None:
        key = self._make_key(session_id)
        payload = {"role": role, "content": content}

        if self._redis is None:
            history = self._memory[key]
            history.append(payload)
            self._memory[key] = history[-self._history_limit :]
            return

        await self._redis.rpush(key, json.dumps(payload, ensure_ascii=False))
        await self._redis.ltrim(key, -self._history_limit, -1)
        await self._redis.expire(key, self._ttl_seconds)

    async def get_pending(self, session_id: str) -> dict[str, object] | None:
        key = self._make_pending_key(session_id)
        if self._redis is None:
            payload = self._pending_memory.get(key)
            if payload is None:
                return None
            return dict(payload)

        raw = await self._redis.get(key)
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    async def set_pending(self, session_id: str, payload: dict[str, object]) -> None:
        key = self._make_pending_key(session_id)
        if self._redis is None:
            self._pending_memory[key] = dict(payload)
            return

        await self._redis.set(key, json.dumps(payload, ensure_ascii=False), ex=self._ttl_seconds)

    async def clear_pending(self, session_id: str) -> None:
        key = self._make_pending_key(session_id)
        if self._redis is None:
            self._pending_memory.pop(key, None)
            return
        await self._redis.delete(key)

    async def get_profile(self, scope_key: str) -> dict[str, object] | None:
        key = self._make_profile_key(scope_key)
        if self._redis is None:
            payload = self._profile_memory.get(key)
            if payload is None:
                return None
            return dict(payload)

        raw = await self._redis.get(key)
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    async def set_profile(self, scope_key: str, payload: dict[str, object]) -> None:
        key = self._make_profile_key(scope_key)
        if self._redis is None:
            self._profile_memory[key] = dict(payload)
            return

        await self._redis.set(key, json.dumps(payload, ensure_ascii=False), ex=self._profile_ttl_seconds)

    async def clear_profile(self, scope_key: str) -> None:
        key = self._make_profile_key(scope_key)
        if self._redis is None:
            self._profile_memory.pop(key, None)
            return
        await self._redis.delete(key)

    @staticmethod
    def _make_key(session_id: str) -> str:
        return f"travel_assistant:session:{session_id}"

    @staticmethod
    def _make_pending_key(session_id: str) -> str:
        return f"travel_assistant:pending:{session_id}"

    @staticmethod
    def _make_profile_key(scope_key: str) -> str:
        return f"travel_assistant:profile:{scope_key}"
