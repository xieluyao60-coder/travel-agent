from __future__ import annotations

import re

from app.schemas import IntentDecision, IntentType


class IntentRouter:
    _route_from_to = re.compile(
        r"从(?P<origin>.+?)到(?P<destination>.+?)(?:怎么|如何|怎样|路线|通勤|去|走|要多久|多长时间|[?？]|$)"
    )
    _route_general = re.compile(
        r"(?P<origin>[\u4e00-\u9fa5A-Za-z0-9·\-]+?)到(?P<destination>[\u4e00-\u9fa5A-Za-z0-9·\-]+?)(?:怎么|如何|路线|通勤|多久)"
    )
    _weather_location = re.compile(r"(?P<location>[\u4e00-\u9fa5A-Za-z0-9·\-]{2,})(?:天气|气温)")

    _weather_keywords = ("天气", "气温", "下雨", "降温", "体感")
    _search_keywords = ("搜索", "搜", "查一下", "查询", "攻略", "推荐", "景点")
    _weather_time_prefixes = (
        "今天",
        "明天",
        "后天",
        "今晚",
        "今早",
        "今晨",
        "本周",
        "这周",
        "下周",
        "周末",
        "明日",
        "昨日",
    )

    def detect(self, text: str) -> IntentDecision:
        content = text.strip()
        if not content:
            return IntentDecision(intent=IntentType.CHAT)

        route_match = self._route_from_to.search(content) or self._route_general.search(content)
        if route_match:
            origin = route_match.group("origin").strip(" ，,。")
            destination = route_match.group("destination").strip(" ，,。")
            return IntentDecision(intent=IntentType.ROUTE, origin=origin, destination=destination)

        if any(keyword in content for keyword in self._weather_keywords):
            location_match = self._weather_location.search(content)
            location = self._normalize_weather_location(location_match.group("location")) if location_match else None
            return IntentDecision(intent=IntentType.WEATHER, location=location)

        if any(keyword in content for keyword in self._search_keywords):
            query = re.sub(r"^(帮我)?(搜索|搜|查一下|查询|查)\s*", "", content).strip()
            return IntentDecision(intent=IntentType.SEARCH, query=query or content)

        return IntentDecision(intent=IntentType.CHAT)

    def _normalize_weather_location(self, location: str) -> str | None:
        normalized = location.strip()
        for prefix in self._weather_time_prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                break
        return normalized or None
