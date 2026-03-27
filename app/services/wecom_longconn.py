from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any

from app.services.orchestrator import ChatOrchestrator

try:
    import websockets
except Exception:  # pragma: no cover - dependency guard
    websockets = None


logger = logging.getLogger(__name__)


CMD_SUBSCRIBE = "aibot_subscribe"
CMD_HEARTBEAT = "ping"
CMD_RESPONSE = "aibot_respond_msg"
CMD_RESPONSE_WELCOME = "aibot_respond_welcome_msg"
CMD_CALLBACK = "aibot_msg_callback"
CMD_EVENT_CALLBACK = "aibot_event_callback"
EVENT_DISCONNECTED = "disconnected_event"
EVENT_ENTER_CHAT = "enter_chat"
MAX_STREAM_BYTES = 20_480


@dataclass
class WeComLongConnConfig:
    bot_id: str
    bot_secret: str
    ws_url: str = "wss://openws.work.weixin.qq.com"
    heartbeat_interval_seconds: int = 30
    max_missed_heartbeat: int = 2
    reconnect_base_delay_seconds: float = 1.0
    reconnect_max_delay_seconds: float = 30.0
    max_auth_failure_attempts: int = 5


class WeComLongConnAuthError(RuntimeError):
    pass


class WeComLongConnectionWorker:
    def __init__(self, config: WeComLongConnConfig, orchestrator: ChatOrchestrator) -> None:
        self._config = config
        self._orchestrator = orchestrator
        self._stop_event = asyncio.Event()
        self._runner_task: asyncio.Task[None] | None = None
        self._connected = False
        self._authenticated = False
        self._missed_heartbeat = 0
        self._reconnect_attempt = 0
        self._auth_failure_attempt = 0
        self._last_error = ""

    async def start(self) -> None:
        if self._runner_task is not None and not self._runner_task.done():
            return
        if websockets is None:
            raise RuntimeError("websockets dependency is missing, cannot start WeCom long connection worker")
        self._stop_event.clear()
        self._runner_task = asyncio.create_task(self._run_forever(), name="wecom-long-connection")

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
            except WeComLongConnAuthError as exc:
                self._auth_failure_attempt += 1
                self._last_error = str(exc)
                logger.warning(
                    "wecom long connection auth failure (%s/%s): %s",
                    self._auth_failure_attempt,
                    self._config.max_auth_failure_attempts,
                    exc,
                )
                if (
                    self._config.max_auth_failure_attempts != -1
                    and self._auth_failure_attempt >= self._config.max_auth_failure_attempts
                ):
                    logger.error("wecom long connection stopped after too many auth failures")
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("wecom long connection loop error: %s", exc)

            if self._stop_event.is_set():
                break

            self._reconnect_attempt += 1
            backoff = min(
                self._config.reconnect_base_delay_seconds * (2 ** max(0, self._reconnect_attempt - 1)),
                self._config.reconnect_max_delay_seconds,
            )
            logger.info("wecom long connection reconnect in %.1fs (attempt=%s)", backoff, self._reconnect_attempt)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass

    async def _run_once(self) -> None:
        if websockets is None:  # pragma: no cover - dependency guard
            raise RuntimeError("websockets dependency is missing")

        logger.info("wecom long connection connecting: %s", self._config.ws_url)
        self._connected = False
        self._authenticated = False
        self._missed_heartbeat = 0

        async with websockets.connect(  # type: ignore[attr-defined]
            self._config.ws_url,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=5,
        ) as ws:
            self._connected = True
            await self._send_auth(ws)

            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws), name="wecom-heartbeat")
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

    async def _heartbeat_loop(self, ws: Any) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._config.heartbeat_interval_seconds)
                return
            except asyncio.TimeoutError:
                pass

            self._missed_heartbeat += 1
            if self._missed_heartbeat > self._config.max_missed_heartbeat:
                logger.warning("wecom heartbeat ack timeout, closing websocket")
                await ws.close(code=1011, reason="heartbeat timeout")
                return

            await self._send_frame(
                ws,
                {
                    "cmd": CMD_HEARTBEAT,
                    "headers": {"req_id": self._generate_req_id(CMD_HEARTBEAT)},
                },
            )

    async def _send_auth(self, ws: Any) -> None:
        await self._send_frame(
            ws,
            {
                "cmd": CMD_SUBSCRIBE,
                "headers": {"req_id": self._generate_req_id(CMD_SUBSCRIBE)},
                "body": {
                    "bot_id": self._config.bot_id,
                    "secret": self._config.bot_secret,
                },
            },
        )

    async def _handle_frame(self, ws: Any, frame: dict[str, Any]) -> None:
        cmd = str(frame.get("cmd") or "")
        req_id = self._extract_req_id(frame)

        if cmd in (CMD_CALLBACK, CMD_EVENT_CALLBACK):
            await self._handle_callback_frame(ws, frame)
            return

        if req_id.startswith(CMD_SUBSCRIBE):
            errcode = int(frame.get("errcode", -1))
            if errcode != 0:
                errmsg = str(frame.get("errmsg") or "unknown error")
                raise WeComLongConnAuthError(f"wecom long connection auth failed: {errmsg} (code={errcode})")
            self._authenticated = True
            self._reconnect_attempt = 0
            self._auth_failure_attempt = 0
            logger.info("wecom long connection authenticated")
            return

        if req_id.startswith(CMD_HEARTBEAT):
            errcode = int(frame.get("errcode", -1))
            if errcode == 0:
                self._missed_heartbeat = 0
            return

        errcode = frame.get("errcode")
        if isinstance(errcode, int) and errcode != 0:
            logger.warning("wecom frame ack error: req_id=%s errcode=%s errmsg=%s", req_id, errcode, frame.get("errmsg"))

    async def _handle_callback_frame(self, ws: Any, frame: dict[str, Any]) -> None:
        req_id = self._extract_req_id(frame)
        if not req_id:
            logger.warning("wecom callback frame missing req_id")
            return

        body = frame.get("body")
        if not isinstance(body, dict):
            logger.warning("wecom callback frame missing body")
            return

        cmd = str(frame.get("cmd") or "")
        if cmd == CMD_EVENT_CALLBACK:
            event_type = str(((body.get("event") or {}) if isinstance(body.get("event"), dict) else {}).get("eventtype") or "")
            if event_type == EVENT_DISCONNECTED:
                raise RuntimeError("wecom server sent disconnected_event")
            if event_type == EVENT_ENTER_CHAT:
                try:
                    await self._send_welcome_reply(ws, req_id)
                    logger.warning("wecom welcome reply sent req_id=%s", req_id)
                except Exception as exc:
                    logger.exception("wecom welcome reply failed req_id=%s: %s", req_id, exc)
            return

        user_id = self._extract_user_id(body)
        chat_id = str(body.get("chatid") or user_id or "unknown")
        msg_type = str(body.get("msgtype") or "")
        logger.warning(
            "wecom callback received req_id=%s msgtype=%s chatid=%s user=%s",
            req_id,
            msg_type,
            chat_id,
            user_id,
        )

        user_text = self._extract_user_text(body)
        if not user_text:
            logger.warning("wecom callback empty text req_id=%s msgtype=%s body_keys=%s", req_id, msg_type, list(body.keys()))
            try:
                await self._send_stream_reply(
                    ws,
                    req_id=req_id,
                    content=(
                        "我目前主要处理文字问题。你可以直接问我“上海天气”"
                        "或“从虹桥站到外滩怎么走”。"
                    ),
                )
                logger.warning("wecom empty-text hint sent req_id=%s", req_id)
            except Exception as exc:
                logger.exception("wecom empty-text hint send failed req_id=%s: %s", req_id, exc)
            return

        session_id = f"wecom-longconn:{chat_id}"
        try:
            reply = await self._orchestrator.handle(user_id=user_id, text=user_text, session_id=session_id)
            reply_text = reply.text
        except Exception as exc:
            logger.exception("wecom long connection orchestrator failed: %s", exc)
            reply_text = "抱歉，这次查询我没稳住。你可以稍后再试，或把问题说得更具体一点。"

        try:
            await self._send_stream_reply(ws, req_id=req_id, content=reply_text)
            logger.warning("wecom reply sent req_id=%s chatid=%s user=%s text_len=%s", req_id, chat_id, user_id, len(reply_text))
        except Exception as exc:
            logger.exception("wecom reply send failed req_id=%s chatid=%s user=%s: %s", req_id, chat_id, user_id, exc)
            raise

    async def _send_welcome_reply(self, ws: Any, req_id: str) -> None:
        await self._send_frame(
            ws,
            {
                "cmd": CMD_RESPONSE_WELCOME,
                "headers": {"req_id": req_id},
                "body": {
                    "msgtype": "text",
                    "text": {
                        "content": (
                            "你好，我是你的旅行规划助手。"
                            "我可以帮你查天气、规划路线、做联网搜索。"
                            "你可以直接说需求，比如“上海天气”或“从A到B怎么走”。"
                        ),
                    },
                },
            },
        )

    async def _send_stream_reply(self, ws: Any, req_id: str, content: str) -> None:
        safe_content = self._truncate_utf8(content, MAX_STREAM_BYTES)
        body = {
            "msgtype": "stream",
            "stream": {
                "id": self._generate_stream_id(),
                "finish": True,
                "content": safe_content,
            },
        }
        await self._send_frame(
            ws,
            {
                "cmd": CMD_RESPONSE,
                "headers": {"req_id": req_id},
                "body": body,
            },
        )

    @staticmethod
    async def _send_frame(ws: Any, frame: dict[str, Any]) -> None:
        payload = json.dumps(frame, ensure_ascii=False)
        await ws.send(payload)

    @staticmethod
    def _extract_req_id(frame: dict[str, Any]) -> str:
        headers = frame.get("headers")
        if not isinstance(headers, dict):
            return ""
        req_id = headers.get("req_id")
        return str(req_id or "")

    @staticmethod
    def _extract_user_id(body: dict[str, Any]) -> str:
        from_data = body.get("from")
        if isinstance(from_data, dict):
            user_id = from_data.get("userid")
            if user_id:
                return str(user_id)
        return "unknown"

    @staticmethod
    def _extract_user_text(body: dict[str, Any]) -> str:
        msg_type = str(body.get("msgtype") or "")
        if msg_type == "text":
            text_obj = body.get("text")
            if isinstance(text_obj, dict):
                return str(text_obj.get("content") or "").strip()
            return ""

        if msg_type == "markdown":
            markdown_obj = body.get("markdown")
            if isinstance(markdown_obj, dict):
                return str(markdown_obj.get("content") or "").strip()
            return str(body.get("content") or "").strip()

        if msg_type == "voice":
            voice_obj = body.get("voice")
            if isinstance(voice_obj, dict):
                return str(voice_obj.get("content") or "").strip()
            return ""

        if msg_type == "mixed":
            mixed_obj = body.get("mixed")
            if not isinstance(mixed_obj, dict):
                return str(body.get("content") or "").strip()
            items = mixed_obj.get("msg_item")
            if not isinstance(items, list):
                return str(body.get("content") or "").strip()
            texts: list[str] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("msgtype") or "") != "text":
                    continue
                text_obj = item.get("text")
                if not isinstance(text_obj, dict):
                    continue
                content = str(text_obj.get("content") or "").strip()
                if content:
                    texts.append(content)
            merged = "\n".join(texts).strip()
            if merged:
                return merged
            return str(body.get("content") or "").strip()

        return str(body.get("content") or "").strip()

    @staticmethod
    def _truncate_utf8(text: str, max_bytes: int) -> str:
        if len(text.encode("utf-8")) <= max_bytes:
            return text
        encoded = text.encode("utf-8")[:max_bytes]
        return encoded.decode("utf-8", errors="ignore")

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
            logger.warning("wecom frame parse failed: invalid json")
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _generate_req_id(prefix: str) -> str:
        return f"{prefix}_{int(time.time() * 1000)}_{secrets.token_hex(4)}"

    @staticmethod
    def _generate_stream_id() -> str:
        return f"stream_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
