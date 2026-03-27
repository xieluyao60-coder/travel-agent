from __future__ import annotations

from datetime import date

import httpx

from app.errors import ExternalAPIError, UserInputError, WeatherForecastRangeError
from app.providers.common import to_float, to_int
from app.schemas import WeatherForecastResult, WeatherResult


class QWeatherProvider:
    GEO_PATH = "/geo/v2/city/lookup"
    NOW_PATH = "/v7/weather/now"
    FORECAST_PATHS = ("/v7/weather/7d", "/v7/weather/3d")

    def __init__(self, api_key: str, api_host: str, client: httpx.AsyncClient) -> None:
        self._api_key = api_key
        self._api_host = api_host.rstrip("/")
        self._client = client

    async def now(self, location_text: str) -> WeatherResult:
        location_id, chosen = await self._resolve_location(location_text)
        weather_data = await self._request_qweather(
            url=f"{self._api_host}{self.NOW_PATH}",
            params={"location": location_id, "key": self._api_key},
            action="实时天气",
        )

        if weather_data.get("code") != "200":
            raise ExternalAPIError(f"和风天气-实时天气失败: {weather_data.get('code')}")

        now = weather_data.get("now") or {}
        temperature = to_float(now.get("temp"))
        if temperature is None:
            raise ExternalAPIError("和风天气返回温度字段异常")

        location_name = self._display_name(chosen, fallback=location_text)
        return WeatherResult(
            location=location_name,
            condition=now.get("text") or "未知",
            temperature_c=temperature,
            feels_like_c=to_float(now.get("feelsLike")),
            humidity=to_int(now.get("humidity")),
            wind_direction=now.get("windDir"),
            wind_scale=now.get("windScale"),
            observed_at=now.get("obsTime"),
        )

    async def forecast(
        self,
        location_text: str,
        *,
        target_date: date | None = None,
        days_ahead: int | None = None,
    ) -> WeatherForecastResult:
        location_id, chosen = await self._resolve_location(location_text)
        daily = await self._get_daily_forecasts(location_id)
        if not daily:
            raise ExternalAPIError("和风天气预报返回为空")

        selected: dict | None = None
        if target_date is not None:
            target_iso = target_date.isoformat()
            for item in daily:
                if str(item.get("fxDate") or "") == target_iso:
                    selected = item
                    break
            if selected is None:
                first = str(daily[0].get("fxDate") or "")
                last = str(daily[-1].get("fxDate") or "")
                raise WeatherForecastRangeError(first, last)
        elif days_ahead is not None:
            if days_ahead < 0 or days_ahead >= len(daily):
                first = str(daily[0].get("fxDate") or "")
                last = str(daily[-1].get("fxDate") or "")
                raise WeatherForecastRangeError(first, last)
            selected = daily[days_ahead]
        else:
            selected = daily[0]

        fx_date_text = str(selected.get("fxDate") or "")
        fx_date = date.fromisoformat(fx_date_text)
        location_name = self._display_name(chosen, fallback=location_text)
        return WeatherForecastResult(
            location=location_name,
            forecast_date=fx_date,
            condition_day=str(selected.get("textDay") or "未知"),
            condition_night=(str(selected.get("textNight") or "").strip() or None),
            temp_min_c=to_float(selected.get("tempMin")),
            temp_max_c=to_float(selected.get("tempMax")),
            humidity=to_int(selected.get("humidity")),
            wind_direction=str(selected.get("windDirDay") or selected.get("windDir") or "") or None,
            wind_scale=str(selected.get("windScaleDay") or selected.get("windScale") or "") or None,
            sunrise=str(selected.get("sunrise") or "") or None,
            sunset=str(selected.get("sunset") or "") or None,
        )

    async def _resolve_location(self, location_text: str) -> tuple[str, dict]:
        if not self._api_key:
            raise ExternalAPIError("未配置和风天气 API Key")
        if not self._api_host:
            raise ExternalAPIError("未配置和风天气 API Host，请在 .env 设置 QWEATHER_API_HOST")

        location_text = location_text.strip()
        if not location_text:
            raise UserInputError("请提供城市名称，例如“上海”")

        geo_data = await self._request_qweather(
            url=f"{self._api_host}{self.GEO_PATH}",
            params={"location": location_text, "key": self._api_key},
            action="城市查询",
        )
        if geo_data.get("code") != "200":
            raise ExternalAPIError(f"和风天气-城市查询失败: {geo_data.get('code')}")

        candidates = geo_data.get("location") or []
        if not candidates:
            raise UserInputError(f"没有找到“{location_text}”对应城市")

        chosen = candidates[0]
        location_id = chosen.get("id")
        if not location_id:
            raise ExternalAPIError("和风天气返回城市 ID 缺失")
        return str(location_id), chosen

    async def _get_daily_forecasts(self, location_id: str) -> list[dict]:
        last_error: str | None = None
        for path in self.FORECAST_PATHS:
            data = await self._request_qweather(
                url=f"{self._api_host}{path}",
                params={"location": location_id, "key": self._api_key},
                action=f"天气预报({path})",
            )
            if data.get("code") == "200":
                daily = data.get("daily") or []
                if isinstance(daily, list) and daily:
                    return daily
                last_error = "预报结果为空"
                continue
            last_error = str(data.get("code") or "unknown")

        raise ExternalAPIError(f"和风天气-天气预报失败: {last_error or 'unknown'}")

    async def _request_qweather(self, url: str, params: dict[str, str], action: str) -> dict:
        try:
            response = await self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise ExternalAPIError(f"和风天气-{action} 请求失败: {exc}") from exc

        try:
            data = response.json() if response.text else {}
        except ValueError as exc:
            raise ExternalAPIError(f"和风天气-{action} 返回了无效 JSON") from exc

        if not response.is_success:
            detail = self._extract_error_detail(data) or response.text or f"HTTP {response.status_code}"
            if "invalid host" in detail.lower():
                raise ExternalAPIError(
                    "和风天气 API Host 未授权。请在和风控制台复制项目专属 API Host，填入 QWEATHER_API_HOST。"
                )
            raise ExternalAPIError(f"和风天气-{action} 请求失败: {detail}")

        if not isinstance(data, dict):
            raise ExternalAPIError(f"和风天气-{action} 返回结构异常")

        return data

    @staticmethod
    def _display_name(chosen: dict, fallback: str) -> str:
        city = (chosen.get("name") or "").strip()
        district = (chosen.get("adm2") or "").strip()
        if city and district and city != district:
            return f"{district}{city}"
        if city:
            return city
        return fallback

    @staticmethod
    def _extract_error_detail(data: dict | object) -> str | None:
        if not isinstance(data, dict):
            return None

        error = data.get("error")
        if isinstance(error, dict):
            detail = error.get("detail")
            title = error.get("title")
            if detail and title:
                return f"{title}: {detail}"
            if detail:
                return str(detail)
            if title:
                return str(title)

        for key in ("message", "msg", "detail", "code"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return None
