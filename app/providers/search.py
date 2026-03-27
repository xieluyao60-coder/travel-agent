from __future__ import annotations

import httpx

from app.errors import ExternalAPIError, UserInputError
from app.providers.common import request_json
from app.schemas import SearchResult


class SerpApiProvider:
    SEARCH_URL = "https://serpapi.com/search.json"

    def __init__(self, api_key: str, client: httpx.AsyncClient) -> None:
        self._api_key = api_key
        self._client = client

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._api_key:
            raise ExternalAPIError("未配置 SerpAPI Key")

        query = query.strip()
        if not query:
            raise UserInputError("请提供搜索关键词")

        response = await self._client.get(
            self.SEARCH_URL,
            params={
                "api_key": self._api_key,
                "engine": "google",
                "q": query,
                "hl": "zh-cn",
                "gl": "cn",
            },
        )
        data = await request_json(response, "SerpAPI")
        if data.get("error"):
            raise ExternalAPIError(f"SerpAPI 错误: {data['error']}")

        organic = data.get("organic_results") or []
        results: list[SearchResult] = []
        for item in organic[: max(top_k, 1)]:
            title = item.get("title")
            link = item.get("link")
            if not title or not link:
                continue
            results.append(
                SearchResult(
                    title=title,
                    link=link,
                    snippet=item.get("snippet") or item.get("snippet_highlighted_words", [""])[0] or None,
                )
            )

        if not results:
            raise ExternalAPIError("搜索结果为空")

        return results
