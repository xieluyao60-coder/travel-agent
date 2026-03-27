from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from app.errors import ExternalAPIError, UserInputError
from app.providers.common import request_json


class QQBotOpenAPIClient:
    TOKEN_PATH = "/app/getAppAccessToken"
    GATEWAY_BOT_PATH = "/gateway/bot"

    def __init__(
        self,
        app_id: str,
        client_secret: str,
        api_base_url: str,
        auth_base_url: str,
        client: httpx.AsyncClient,
    ) -> None:
        self._app_id = app_id.strip()
        self._client_secret = client_secret.strip()
        self._api_base_url = api_base_url.rstrip("/")
        self._auth_base_url = auth_base_url.rstrip("/")
        self._client = client

        self._access_token: str | None = None
        self._token_expire_at_monotonic = 0.0
        self._token_lock = asyncio.Lock()

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        if not force_refresh and self._token_available():
            return str(self._access_token)

        async with self._token_lock:
            if not force_refresh and self._token_available():
                return str(self._access_token)

            if not self._app_id or not self._client_secret:
                raise ExternalAPIError("未配置 QQ 机器人凭据（QQ_BOT_APP_ID / QQ_BOT_CLIENT_SECRET）")

            payload = {
                "appId": self._app_id,
                "clientSecret": self._client_secret,
            }
            response = await self._client.post(
                f"{self._auth_base_url}{self.TOKEN_PATH}",
                json=payload,
            )
            data = await request_json(response, "QQ access token")

            access_token = str(data.get("access_token") or data.get("accessToken") or "").strip()
            if not access_token:
                raise ExternalAPIError("QQ access token 返回为空")

            expires_in_raw = data.get("expires_in", data.get("expiresIn", 0))
            try:
                expires_in = int(expires_in_raw)
            except (TypeError, ValueError):
                expires_in = 0
            if expires_in <= 0:
                expires_in = 7200

            # Refresh early to avoid edge-of-expiry failures.
            refresh_before_expiry = min(300, max(60, expires_in // 10))
            self._access_token = access_token
            self._token_expire_at_monotonic = time.monotonic() + max(1, expires_in - refresh_before_expiry)
            return access_token

    async def get_gateway_url(self) -> str:
        data = await self._request_openapi("GET", self.GATEWAY_BOT_PATH)
        url = str(data.get("url") or "").strip()
        if not url:
            raise ExternalAPIError("QQ 网关地址为空（/gateway/bot）")
        return url

    async def send_group_text(self, *, group_openid: str, content: str, msg_id: str | None = None) -> dict[str, Any]:
        group_openid = group_openid.strip()
        if not group_openid:
            raise UserInputError("QQ 群消息缺少 group_openid")

        body: dict[str, Any] = {"content": content}
        if msg_id:
            body["msg_id"] = msg_id
        return await self._request_openapi("POST", f"/v2/groups/{group_openid}/messages", json_body=body)

    async def send_c2c_text(self, *, user_openid: str, content: str, msg_id: str | None = None) -> dict[str, Any]:
        user_openid = user_openid.strip()
        if not user_openid:
            raise UserInputError("QQ 单聊消息缺少 user_openid")

        body: dict[str, Any] = {"content": content}
        if msg_id:
            body["msg_id"] = msg_id
        return await self._request_openapi("POST", f"/v2/users/{user_openid}/messages", json_body=body)

    async def _request_openapi(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for attempt in range(2):
            token = await self.get_access_token(force_refresh=attempt > 0)
            headers = {
                "Authorization": f"QQBot {token}",
                "X-Union-Appid": self._app_id,
            }
            response = await self._client.request(
                method=method,
                url=f"{self._api_base_url}{path}",
                headers=headers,
                json=json_body,
            )
            if response.status_code in {401, 403} and attempt == 0:
                continue

            data = await request_json(response, f"QQ OpenAPI {path}")
            code = data.get("code")
            if code not in (None, 0, "0", 200, "200"):
                message = str(data.get("message") or data.get("msg") or data.get("error") or "unknown")
                raise ExternalAPIError(f"QQ OpenAPI 调用失败: path={path} code={code} message={message}")
            return data

        raise ExternalAPIError(f"QQ OpenAPI 调用失败: path={path}，鉴权未通过")

    def _token_available(self) -> bool:
        return bool(self._access_token and time.monotonic() < self._token_expire_at_monotonic)

