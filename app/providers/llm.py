from __future__ import annotations

import httpx

from app.errors import ExternalAPIError


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        timeout_seconds: float,
        client: httpx.AsyncClient,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def chat(self, messages: list[dict[str, str]]) -> str:
        if not self._api_key:
            raise ExternalAPIError("未配置 LLM API Key")

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "temperature": self._temperature,
                "messages": messages,
            },
            timeout=self._timeout_seconds,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalAPIError(f"LLM 调用失败: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalAPIError("LLM 返回了无效 JSON") from exc

        choices = data.get("choices") or []
        if not choices:
            raise ExternalAPIError("LLM 返回为空")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            merged = "".join(text_parts).strip()
            if merged:
                return merged

        raise ExternalAPIError("LLM 未返回可用文本")
