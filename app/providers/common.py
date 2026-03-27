from __future__ import annotations

import httpx

from app.errors import ExternalAPIError


async def request_json(response: httpx.Response, provider_name: str) -> dict:
    try:
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ExternalAPIError(f"{provider_name} 请求失败: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise ExternalAPIError(f"{provider_name} 返回了无效 JSON") from exc

    if not isinstance(data, dict):
        raise ExternalAPIError(f"{provider_name} 返回结构异常")

    return data


def to_float(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
