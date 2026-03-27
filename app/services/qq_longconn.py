from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from app.errors import ExternalAPIError
from app.providers.qq import QQBotOpenAPIClient
from app.services.orchestrator import ChatOrchestrator

try:
    import websockets
except Exception:  # pragma: no cover - dependency guard
    websockets = None


logger = logging.getLogger(__name__)


QQ_OP_DISPATCH = 0
QQ_OP_HEARTBEAT = 1
QQ_OP_IDENTIFY = 2
QQ_OP_RECONNECT = 7
QQ_OP_INVALID_SESSION = 9
QQ_OP_HELLO = 10
QQ_OP_HEARTBEAT_ACK = 11

QQ_EVENT_GROUP_AT_MESSAGE_CREATE = "GROUP_AT_MESSAGE_CREATE"
QQ_EVENT_C2C_MESSAGE_CREATE = "C2C_MESSAGE_CREATE"

DEFAULT_QQ_INTENTS = (1 << 30) | (1 << 25)
MAX_STREAM_BYTES = 20_480


@dataclass
class QQLongConnConfig:
    app_id: str
    client_secret: str
    intents: int = DEFAULT_QQ_INTENTS
    reconnect_base_delay_seconds: float = 1.0
    reconnect_max_delay_seconds: float = 30.0
    max_missed_heartbeat: int = 2
    max_auth_failure_attempts: int = 5
    event_dedup_ttl_seconds: int = 120
    event_dedup_max_size: int = 5000


class QQLongConnAuthError(RuntimeError):
    pass


class _EventDeduplicator:
    def __init__(self, ttl_seconds: int, max_size: int) -> None:
        self._ttl_seconds = max(1, ttl_seconds)
        self._max_size = max(100, max_size)
        self._seen: OrderedDict[str, float] = OrderedDict()

    def seen_recently(self, event_id: str) -> bool:
        if not event_id:
            return False
        now = time.monotonic()
        self._evict(now)

        existing = self._seen.get(event_id)
        if existing is not None and now - existing <= self._ttl_seconds:
            self._seen.move_to_end(event_id)
            return True

        self._seen[event_id] = now
        self._seen.move_to_end(event_id)
        self._evict(now)
        return False

    def _evict(self, now: float) -> None:
        while self._seen:
            oldest_event_id, oldest_ts = next(iter(self._seen.items()))
            if len(self._seen) <= self._max_size and now - oldest_ts <= self._ttl_seconds:
                break
            self._seen.pop(oldest_event_id, None)


class QQLongConnectionWorker:
    def __init__(
        self,
        *,
        config: QQLongConnConfig,
        orchestrator: ChatOrchestrator,
        qq_client: QQBotOpenAPIClient,
    ) -> None:
        self._config = config
        self._orchestrator = orchestrator
        self._qq_client = qq_client

        self._stop_event = asyncio.Event()
        self._runner_task: asyncio.Task[None] | None = None

        self._connected = False
        self._authenticated = False
        self._reconnect_attempt = 0
        self._auth_failure_attempt = 0
        self._last_error = ""
        self._last_seq: int | None = None
        self._missed_heartbeat = 0
        self._heartbeat_interval_seconds = 30.0
        self._deduplicator = _EventDeduplicator(
            ttl_seconds=self._config.event_dedup_ttl_seconds,
            max_size=self._config.event_dedup_max_size,
        )

    async def start(self) -> None:
        if self._runner_task is not None and not self._runner_task.done():
            return
        if websockets is None:
            raise RuntimeError("websockets dependency is missing, cannot start QQ long connection worker")

        self._stop_event.clear()
        self._runner_task = asyncio.create_task(self._run_forever(), name="qq-long-connection")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._runner_task is None:
            return
        self._runner_task.cancel()
        try:
            await self._runner_task
        except asyncio.CancelledError:
            pass
        self._runner_task = None
        self._connected = False
        self._authenticated = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "authenticated": self._authenticated,
            "reconnect_attempt": self._reconnect_attempt,
            "last_error": self._last_error,
        }

    async def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._run_once()
            except QQLongConnAuthError as exc:
                self._auth_failure_attempt += 1
                self._last_error = str(exc)
                logger.warning(
                    "qq long connection auth failure (%s/%s): %s",
                    self._auth_failure_attempt,
                    self._config.max_auth_failure_attempts,
                    exc,
                )
                if (
                    self._config.max_auth_failure_attempts != -1
                    and self._auth_failure_attempt >= self._config.max_auth_failure_attempts
                ):
                    logger.error("qq long connection stopped after too many auth failures")
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("qq long connection loop error: %s", exc)

            if self._stop_event.is_set():
                break

            self._reconnect_attempt += 1
            backoff = min(
                self._config.reconnect_base_delay_seconds * (2 ** max(0, self._reconnect_attempt - 1)),
                self._config.reconnect_max_delay_seconds,
            )
            logger.info("qq long connection reconnect in %.1fs (attempt=%s)", backoff, self._reconnect_attempt)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass

    async def _run_once(self) -> None:
        if websockets is None:  # pragma: no cover - dependency guard
            raise RuntimeError("websockets dependency is missing")

        gateway_url = await self._qq_client.get_gateway_url()
        logger.info("qq long connection connecting: %s", gateway_url)

        self._connected = False
        self._authenticated = False
        self._missed_heartbeat = 0
        self._last_seq = None

        async with websockets.connect(  # type: ignore[attr-defined]
            gateway_url,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=5,
        ) as ws:
            self._connected = True

            first = await ws.recv()
            hello = self._parse_frame(first)
            if hello is None or int(hello.get("op", -1)) != QQ_OP_HELLO:
                raise QQLongConnAuthError("qq gateway did not return HELLO")
            self._heartbeat_interval_seconds = self._extract_heartbeat_interval(hello)

            identify_payload = await self._build_identify_payload()
            await self._send_frame(ws, identify_payload)

            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws), name="qq-heartbeat")
            try:
                while not self._stop_event.is_set():
                    raw_message = await ws.recv()
                    frame = self._parse_frame(raw_message)
                    if frame is None:
                        continue
                    await self._handle_frame(ws, frame)
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._connected = False
                self._authenticated = False

    async def _build_identify_payload(self) -> dict[str, Any]:
        token = await self._qq_client.get_access_token()
        if self._config.intents <= 0:
            raise QQLongConnAuthError("qq intents is invalid, please configure QQ_WS_INTENTS")

        return {
            "op": QQ_OP_IDENTIFY,
            "d": {
                "token": f"QQBot {token}",
                "intents": self._config.intents,
                "shard": [0, 1],
                "properties": {
                    "$os": "windows",
                    "$sdk": "travel-assistant",
                },
            },
        }

    async def _heartbeat_loop(self, ws: Any) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._heartbeat_interval_seconds)
                return
            except asyncio.TimeoutError:
                pass

            self._missed_heartbeat += 1
            if self._missed_heartbeat > self._config.max_missed_heartbeat:
                logger.warning("qq heartbeat ack timeout, closing websocket")
                await ws.close(code=1011, reason="heartbeat timeout")
                return

            await self._send_frame(
                ws,
                {
                    "op": QQ_OP_HEARTBEAT,
                    "d": self._last_seq,
                },
            )

    async def _handle_frame(self, ws: Any, frame: dict[str, Any]) -> None:
        op = int(frame.get("op", -1))
        seq = frame.get("s")
        if isinstance(seq, int):
            self._last_seq = seq

        if op == QQ_OP_DISPATCH:
            event_type = str(frame.get("t") or "")
            data = frame.get("d")
            if event_type == "READY":
                self._authenticated = True
                self._reconnect_attempt = 0
                self._auth_failure_attempt = 0
                logger.info("qq long connection authenticated")
                return
            if event_type in {QQ_EVENT_GROUP_AT_MESSAGE_CREATE, QQ_EVENT_C2C_MESSAGE_CREATE}:
                if isinstance(data, dict):
                    await self._handle_message_event(event_type=event_type, data=data)
                return
            return

        if op == QQ_OP_HEARTBEAT_ACK:
            self._missed_heartbeat = 0
            return

        if op == QQ_OP_RECONNECT:
            raise RuntimeError("qq gateway requested reconnect")

        if op == QQ_OP_INVALID_SESSION:
            resumable = bool(frame.get("d"))
            reason = "resumable invalid session" if resumable else "invalid session"
            raise QQLongConnAuthError(f"qq gateway returned {reason}")

        if op == QQ_OP_HELLO:
            self._heartbeat_interval_seconds = self._extract_heartbeat_interval(frame)
            return

        logger.debug("qq frame ignored: op=%s frame=%s", op, frame)

    async def _handle_message_event(self, *, event_type: str, data: dict[str, Any]) -> None:
        event_id = str(data.get("id") or data.get("event_id") or "").strip()
        if self._deduplicator.seen_recently(event_id):
            return
        if self._is_self_message(data):
            return

        user_text = self._extract_user_text(data)

        try:
            if event_type == QQ_EVENT_GROUP_AT_MESSAGE_CREATE:
                group_openid = str(data.get("group_openid") or "").strip()
                user_id = self._extract_user_openid(data)
                if not group_openid or not user_id:
                    return
                if not user_text:
                    await self._qq_client.send_group_text(
                        group_openid=group_openid,
                        content="我目前主要处理文字消息，你可以直接问我天气、路线或旅游攻略。",
                        msg_id=event_id or None,
                    )
                    return
                session_id = f"qq-group:{group_openid}:{user_id}"
                reply = await self._orchestrator.handle(user_id=user_id, text=user_text, session_id=session_id)
                await self._qq_client.send_group_text(group_openid=group_openid, content=self._truncate(reply.text), msg_id=event_id or None)
                return

            if event_type == QQ_EVENT_C2C_MESSAGE_CREATE:
                user_openid = self._extract_user_openid(data)
                if not user_openid:
                    return
                if not user_text:
                    await self._qq_client.send_c2c_text(
                        user_openid=user_openid,
                        content="我目前主要处理文字消息，你可以直接问我天气、路线或旅游攻略。",
                        msg_id=event_id or None,
                    )
                    return
                session_id = f"qq-c2c:{user_openid}"
                reply = await self._orchestrator.handle(user_id=user_openid, text=user_text, session_id=session_id)
                await self._qq_client.send_c2c_text(user_openid=user_openid, content=self._truncate(reply.text), msg_id=event_id or None)
                return
        except ExternalAPIError as exc:
            logger.warning("qq send failed: %s", exc)
        except Exception as exc:
            logger.exception("qq long connection event handle failed: %s", exc)

    def _is_self_message(self, data: dict[str, Any]) -> bool:
        author = data.get("author")
        if isinstance(author, dict):
            if bool(author.get("bot")):
                return True
            author_id = str(author.get("id") or "").strip()
            if author_id and author_id == self._config.app_id:
                return True

        bot_appid = str(data.get("bot_appid") or "").strip()
        return bool(bot_appid and bot_appid == self._config.app_id)

    @staticmethod
    def _extract_user_openid(data: dict[str, Any]) -> str:
        author = data.get("author")
        if isinstance(author, dict):
            author_id = str(author.get("id") or "").strip()
            if author_id:
                return author_id
        for key in ("openid", "user_openid", "member_openid"):
            value = str(data.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _extract_user_text(data: dict[str, Any]) -> str:
        content = str(data.get("content") or "").strip()
        if not content:
            return ""

        cleaned = content
        cleaned = re.sub(r"<qqbot-at-user[^>]*>", " ", cleaned)
        cleaned = re.sub(r"</qqbot-at-user>", " ", cleaned)
        cleaned = re.sub(r"<@!?[\w\-]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    async def _send_frame(ws: Any, frame: dict[str, Any]) -> None:
        payload = json.dumps(frame, ensure_ascii=False)
        await ws.send(payload)

    @staticmethod
    def _extract_heartbeat_interval(frame: dict[str, Any]) -> float:
        data = frame.get("d")
        if isinstance(data, dict):
            raw = data.get("heartbeat_interval")
            try:
                interval_ms = int(raw)
                if interval_ms > 0:
                    return interval_ms / 1000.0
            except (TypeError, ValueError):
                pass
        return 30.0

    @staticmethod
    def _parse_frame(raw_message: Any) -> dict[str, Any] | None:
        if isinstance(raw_message, bytes):
            payload = raw_message.decode("utf-8", errors="ignore")
        elif isinstance(raw_message, str):
            payload = raw_message
        else:
            return None

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("qq frame parse failed: invalid json")
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text.encode("utf-8")) <= MAX_STREAM_BYTES:
            return text
        encoded = text.encode("utf-8")[:MAX_STREAM_BYTES]
        return encoded.decode("utf-8", errors="ignore")
