from __future__ import annotations

import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response

from app.config import Settings
from app.container import AppContainer, build_container
from app.errors import UserInputError
from app.providers.qq import QQBotOpenAPIClient
from app.services.qq_longconn import QQLongConnConfig, QQLongConnectionWorker
from app.services.wecom_longconn import WeComLongConnConfig, WeComLongConnectionWorker
from app.services.wecom import (
    WeComCrypto,
    build_encrypted_reply_xml,
    build_text_reply_xml,
    build_wecom_signature,
    extract_encrypt_text,
    parse_wecom_message,
    verify_wecom_signature,
)

logger = logging.getLogger(__name__)


class _ContainerProxy:
    def __init__(self) -> None:
        self.container: AppContainer | None = None
        self.wecom_long_worker: WeComLongConnectionWorker | None = None
        self.qq_long_worker: QQLongConnectionWorker | None = None


def _is_placeholder(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    return normalized.startswith("replace-with-")


def _wecom_secure_ready(settings: Settings) -> bool:
    aes_key = (settings.wecom_encoding_aes_key or "").strip()
    corp_id = (settings.wecom_corp_id or "").strip()
    if not aes_key or not corp_id:
        return False
    if _is_placeholder(aes_key) or _is_placeholder(corp_id):
        return False
    # WeCom EncodingAESKey is a 43-char Base64 string (without trailing '=').
    return len(aes_key) == 43


def _wecom_long_mode(settings: Settings) -> bool:
    mode = (settings.wecom_connection_mode or "").strip().lower()
    return mode in {"long_connection", "long-connection", "long", "ws", "websocket"}


def _qq_long_mode(settings: Settings) -> bool:
    return bool(settings.qq_enabled)


def create_app(
    settings_override: Settings | None = None,
    container_override: AppContainer | None = None,
) -> FastAPI:
    settings = settings_override or Settings()
    proxy = _ContainerProxy()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if container_override is not None:
            proxy.container = container_override
            yield
            return

        built_container = await build_container(settings)
        proxy.container = built_container

        if _wecom_long_mode(settings):
            if not settings.wecom_bot_id or not settings.wecom_bot_secret:
                logger.error("wecom long connection mode enabled but WECOM_BOT_ID / WECOM_BOT_SECRET is missing")
            else:
                proxy.wecom_long_worker = WeComLongConnectionWorker(
                    config=WeComLongConnConfig(
                        bot_id=settings.wecom_bot_id,
                        bot_secret=settings.wecom_bot_secret,
                        ws_url=settings.wecom_ws_url,
                        heartbeat_interval_seconds=settings.wecom_ws_heartbeat_interval_seconds,
                        max_missed_heartbeat=settings.wecom_ws_max_missed_heartbeat,
                        reconnect_base_delay_seconds=settings.wecom_ws_reconnect_base_delay_seconds,
                        reconnect_max_delay_seconds=settings.wecom_ws_reconnect_max_delay_seconds,
                        max_auth_failure_attempts=settings.wecom_ws_max_auth_failure_attempts,
                    ),
                    orchestrator=built_container.orchestrator,
                )
                await proxy.wecom_long_worker.start()
                logger.info("wecom long connection worker started")

        if _qq_long_mode(settings):
            if not settings.qq_bot_app_id or not settings.qq_bot_client_secret:
                logger.error("qq long connection mode enabled but QQ_BOT_APP_ID / QQ_BOT_CLIENT_SECRET is missing")
            else:
                try:
                    qq_client = QQBotOpenAPIClient(
                        app_id=settings.qq_bot_app_id,
                        client_secret=settings.qq_bot_client_secret,
                        api_base_url=settings.qq_api_base_url,
                        auth_base_url=settings.qq_auth_base_url,
                        client=built_container.http_client,
                    )
                    proxy.qq_long_worker = QQLongConnectionWorker(
                        config=QQLongConnConfig(
                            app_id=settings.qq_bot_app_id,
                            client_secret=settings.qq_bot_client_secret,
                            intents=settings.qq_ws_intents,
                            reconnect_base_delay_seconds=settings.qq_ws_reconnect_base_delay_seconds,
                            reconnect_max_delay_seconds=settings.qq_ws_reconnect_max_delay_seconds,
                            max_missed_heartbeat=settings.qq_ws_max_missed_heartbeat,
                            max_auth_failure_attempts=settings.qq_ws_max_auth_failure_attempts,
                            event_dedup_ttl_seconds=settings.qq_event_dedup_ttl_seconds,
                            event_dedup_max_size=settings.qq_event_dedup_max_size,
                        ),
                        orchestrator=built_container.orchestrator,
                        qq_client=qq_client,
                    )
                    await proxy.qq_long_worker.start()
                    logger.info("qq long connection worker started")
                except Exception as exc:
                    proxy.qq_long_worker = None
                    logger.exception("qq long connection worker failed to start: %s", exc)
        try:
            yield
        finally:
            if proxy.wecom_long_worker is not None:
                await proxy.wecom_long_worker.stop()
                proxy.wecom_long_worker = None
            if proxy.qq_long_worker is not None:
                await proxy.qq_long_worker.stop()
                proxy.qq_long_worker = None
            await built_container.aclose()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ok",
            "env": settings.app_env,
            "service": settings.app_name,
            "wecom_mode": settings.wecom_connection_mode,
            "qq_enabled": settings.qq_enabled,
        }
        if proxy.wecom_long_worker is not None:
            payload["wecom_longconn"] = proxy.wecom_long_worker.snapshot()
        if proxy.qq_long_worker is not None:
            payload["qq_longconn"] = proxy.qq_long_worker.snapshot()
        return payload

    @app.get("/webhook/wecom")
    async def wecom_verify(
        signature: str | None = Query(default=None),
        msg_signature: str | None = Query(default=None),
        timestamp: str | None = Query(default=None),
        nonce: str | None = Query(default=None),
        echostr: str | None = Query(default=None),
    ) -> Response:
        if _wecom_long_mode(settings):
            raise HTTPException(status_code=409, detail="webhook disabled in long connection mode")

        sign = msg_signature or signature
        if not sign or not timestamp or not nonce or echostr is None:
            raise HTTPException(status_code=400, detail="missing verify params")

        is_secure_verify = bool(msg_signature) and _wecom_secure_ready(settings)
        if is_secure_verify:
            if not verify_wecom_signature(settings.wecom_token, timestamp, nonce, sign, payload=echostr):
                raise HTTPException(status_code=401, detail="invalid signature")

            try:
                crypto = WeComCrypto(
                    token=settings.wecom_token,
                    encoding_aes_key=settings.wecom_encoding_aes_key,
                    receive_id=settings.wecom_corp_id,
                )
                plain_echo = crypto.decrypt(echostr)
            except UserInputError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            return Response(content=plain_echo, media_type="text/plain")

        plain_ok = verify_wecom_signature(settings.wecom_token, timestamp, nonce, sign)
        payload_ok = verify_wecom_signature(settings.wecom_token, timestamp, nonce, sign, payload=echostr)
        if not plain_ok and not payload_ok:
            raise HTTPException(status_code=401, detail="invalid signature")

        return Response(content=echostr, media_type="text/plain")

    @app.post("/webhook/wecom")
    async def wecom_webhook(
        request: Request,
        signature: str | None = Query(default=None),
        msg_signature: str | None = Query(default=None),
        timestamp: str | None = Query(default=None),
        nonce: str | None = Query(default=None),
    ) -> Response:
        if _wecom_long_mode(settings):
            raise HTTPException(status_code=409, detail="webhook disabled in long connection mode")

        sign = msg_signature or signature
        if not sign or not timestamp or not nonce:
            raise HTTPException(status_code=400, detail="missing signature params")

        if proxy.container is None:
            raise HTTPException(status_code=503, detail="service not ready")

        raw_body = await request.body()
        encrypted_text = extract_encrypt_text(raw_body)
        use_secure_mode = encrypted_text is not None

        if use_secure_mode:
            if not _wecom_secure_ready(settings):
                raise HTTPException(
                    status_code=500,
                    detail="missing or invalid wecom encryption config: WECOM_ENCODING_AES_KEY / WECOM_CORP_ID",
                )
            if not verify_wecom_signature(settings.wecom_token, timestamp, nonce, sign, payload=encrypted_text):
                raise HTTPException(status_code=401, detail="invalid signature")

            try:
                crypto = WeComCrypto(
                    token=settings.wecom_token,
                    encoding_aes_key=settings.wecom_encoding_aes_key,
                    receive_id=settings.wecom_corp_id,
                )
                plain_xml = crypto.decrypt(encrypted_text)
                incoming = parse_wecom_message(plain_xml.encode("utf-8"))
            except UserInputError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            if not verify_wecom_signature(settings.wecom_token, timestamp, nonce, sign):
                raise HTTPException(status_code=401, detail="invalid signature")
            try:
                incoming = parse_wecom_message(raw_body)
            except UserInputError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        if incoming.msg_type == "text":
            user_input = incoming.content
        elif incoming.msg_type == "event" and incoming.event.lower() == "subscribe":
            user_input = "你好"
        else:
            user_input = ""

        if not user_input:
            reply_text = (
                "我目前主要处理文字消息。你可以直接发文字给我，"
                "比如“从虹桥站到外滩怎么走”或“上海明天天气怎么样”。"
            )
        else:
            session_id = f"wecom:{incoming.from_user_name}:{incoming.agent_id or settings.wecom_agent_id}"
            try:
                reply = await proxy.container.orchestrator.handle(
                    user_id=incoming.from_user_name,
                    text=user_input,
                    session_id=session_id,
                )
                reply_text = reply.text
            except Exception as exc:
                logger.exception("orchestrator failed: %s", exc)
                reply_text = (
                    "抱歉，这次查询我没稳住。你可以稍后再试，"
                    "或者把问题说得更具体一点，我继续帮你处理。"
                )

        plain_reply_xml = build_text_reply_xml(incoming, reply_text)
        if not use_secure_mode:
            return Response(content=plain_reply_xml, media_type="application/xml")

        try:
            crypto = WeComCrypto(
                token=settings.wecom_token,
                encoding_aes_key=settings.wecom_encoding_aes_key,
                receive_id=settings.wecom_corp_id,
            )
            encrypted_reply = crypto.encrypt(plain_reply_xml)
        except UserInputError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        response_timestamp = str(int(time.time()))
        response_nonce = secrets.token_hex(8)
        response_signature = build_wecom_signature(
            token=settings.wecom_token,
            timestamp=response_timestamp,
            nonce=response_nonce,
            payload=encrypted_reply,
        )
        encrypted_xml = build_encrypted_reply_xml(
            encrypt=encrypted_reply,
            signature=response_signature,
            timestamp=response_timestamp,
            nonce=response_nonce,
        )
        return Response(content=encrypted_xml, media_type="application/xml")

    return app


app = create_app()
