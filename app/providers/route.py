from __future__ import annotations

import re

import httpx

from app.errors import ExternalAPIError, UserInputError
from app.providers.common import request_json, to_float, to_int
from app.schemas import NearbyPlace, RoutePlan


class AmapRouteProvider:
    GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
    PLACE_TEXT_URL = "https://restapi.amap.com/v3/place/text"
    PLACE_AROUND_URL = "https://restapi.amap.com/v3/place/around"
    WALKING_URL = "https://restapi.amap.com/v3/direction/walking"
    DRIVING_URL = "https://restapi.amap.com/v3/direction/driving"
    TRANSIT_URL = "https://restapi.amap.com/v3/direction/transit/integrated"
    CITY_ENDING_CHARS = {
        "州",
        "京",
        "海",
        "津",
        "庆",
        "门",
        "宁",
        "昌",
        "沙",
        "阳",
        "川",
        "安",
        "圳",
        "汉",
        "肥",
        "岛",
        "滨",
        "原",
        "林",
        "口",
        "关",
        "山",
        "溪",
        "德",
        "江",
        "宾",
        "兴",
        "坊",
        "溪",
        "台",
        "明",
    }

    def __init__(self, api_key: str, default_city: str, client: httpx.AsyncClient) -> None:
        self._api_key = api_key
        self._default_city = default_city
        self._client = client

    async def plan(self, origin: str, destination: str, mode: str | None = None) -> list[RoutePlan]:
        if not self._api_key:
            raise ExternalAPIError("未配置高德 API Key")

        origin = origin.strip()
        destination = destination.strip()
        if not origin or not destination:
            raise UserInputError("请同时提供起点和终点，例如“从虹桥站到外滩怎么走”")

        origin_lnglat, origin_city = await self._geocode(origin, preferred_city=None)
        destination_lnglat, destination_city = await self._geocode(destination, preferred_city=origin_city)

        transit_city = self._resolve_transit_city(origin_city, destination_city)

        plans: list[RoutePlan] = []
        selected_mode = (mode or "").lower().strip()
        if not selected_mode or selected_mode == "transit":
            plans.extend(await self._transit(origin_lnglat, destination_lnglat, city=transit_city))
        if not selected_mode or selected_mode == "driving":
            driving = await self._driving(origin_lnglat, destination_lnglat)
            if driving:
                plans.append(driving)
        if not selected_mode or selected_mode == "walking":
            walking = await self._walking(origin_lnglat, destination_lnglat)
            if walking:
                plans.append(walking)
        if selected_mode == "transit" and not plans:
            driving = await self._driving(origin_lnglat, destination_lnglat)
            if driving:
                plans.append(driving)
            walking = await self._walking(origin_lnglat, destination_lnglat)
            if walking:
                plans.append(walking)

        if not plans:
            raise ExternalAPIError("未查询到可用通勤方案")
        if origin != destination and self._looks_like_same_point_result(plans):
            raise UserInputError("地点解析结果疑似重合，请补充更具体地名（如区县、站点全称）")

        priority = {"transit": 0, "driving": 1, "walking": 2}
        plans.sort(key=lambda p: (priority.get(p.mode, 99), p.duration_minutes))
        return plans[:3]

    async def nearby(
        self,
        *,
        location_text: str,
        keyword: str,
        radius_m: int = 1000,
        limit: int = 5,
    ) -> list[NearbyPlace]:
        if not self._api_key:
            raise ExternalAPIError("未配置高德 API Key")

        location = location_text.strip()
        query = keyword.strip()
        if not location:
            raise UserInputError("请补充中心地点，例如“温州站附近有咖啡店吗”。")
        if not query:
            raise UserInputError("请补充要找的类型，例如“咖啡店”“便利店”。")

        radius = max(100, min(int(radius_m or 1000), 5000))
        page_size = max(1, min(int(limit or 5), 20))

        center_lnglat, _ = await self._geocode(location, preferred_city=None)
        response = await self._client.get(
            self.PLACE_AROUND_URL,
            params={
                "key": self._api_key,
                "location": center_lnglat,
                "keywords": query,
                "radius": radius,
                "sortrule": "distance",
                "offset": page_size,
                "page": 1,
                "extensions": "base",
            },
        )
        data = await request_json(response, "高德周边检索")
        if data.get("status") != "1":
            raise ExternalAPIError(f"高德周边检索失败: {data.get('info', '未知错误')}")

        pois = data.get("pois") or []
        places: list[NearbyPlace] = []
        for poi in pois[:page_size]:
            name = str(poi.get("name") or "").strip()
            if not name:
                continue
            address = str(poi.get("address") or "").strip() or None
            category = str(poi.get("type") or "").split(";")[0].strip() or None
            places.append(
                NearbyPlace(
                    name=name,
                    address=address,
                    distance_m=to_int(poi.get("distance")),
                    category=category,
                )
            )
        return places

    async def _geocode(self, location_text: str, preferred_city: str | None = None) -> tuple[str, str | None]:
        city_candidates = self._city_candidates(location_text=location_text, preferred_city=preferred_city)

        # First: city-scoped POI search.
        for city in city_candidates:
            place_location, place_city = await self._query_place_text(
                location_text=location_text,
                city=city,
                city_limit=True,
            )
            if place_location:
                return place_location, self._normalize_city_name(place_city) or city

        # Second: global POI search.
        global_place_location, global_place_city = await self._query_place_text(
            location_text=location_text,
            city=None,
            city_limit=False,
        )
        if global_place_location:
            return global_place_location, self._normalize_city_name(global_place_city)

        # Third: geocode fallback.
        for city in city_candidates:
            scoped_geocodes = await self._query_geocode(
                location_text=location_text,
                city=city,
                city_limit=True,
                strict=False,
            )
            matched = self._pick_city_matched_geocode(scoped_geocodes, city)
            if matched and matched.get("location"):
                return matched["location"], self._extract_city_from_geocode(matched)

        geocodes = await self._query_geocode(location_text=location_text, city=None, city_limit=False, strict=True)
        if not geocodes:
            raise UserInputError(f"无法识别地点“{location_text}”，请换个更具体的地名")

        chosen = geocodes[0]
        if self._is_low_confidence_geocode(chosen, location_text):
            raise UserInputError(f"地点“{location_text}”存在歧义，请补充更完整地名")

        lnglat = chosen.get("location")
        if not lnglat:
            raise ExternalAPIError("高德地理编码返回坐标为空")
        return lnglat, self._extract_city_from_geocode(chosen)

    async def _query_geocode(
        self,
        location_text: str,
        city: str | None,
        city_limit: bool,
        strict: bool = True,
    ) -> list[dict]:
        params = {"key": self._api_key, "address": location_text}
        if city:
            params["city"] = city
            if city_limit:
                params["citylimit"] = "true"

        response = await self._client.get(self.GEOCODE_URL, params=params)
        data = await request_json(response, "高德地理编码")
        if data.get("status") != "1":
            if not strict:
                return []
            raise ExternalAPIError(f"高德地理编码失败: {data.get('info', '未知错误')}")

        return data.get("geocodes") or []

    @staticmethod
    def _pick_city_matched_geocode(geocodes: list[dict], city_hint: str) -> dict | None:
        normalized_hint = city_hint.replace("市", "").strip()
        if not geocodes or not normalized_hint:
            return None

        for item in geocodes:
            city = (item.get("city") or "").replace("市", "").strip()
            province = (item.get("province") or "").replace("市", "").strip()
            district = (item.get("district") or "").replace("市", "").strip()
            combined = f"{province}{city}{district}"
            if normalized_hint and normalized_hint in combined:
                return item

        return None

    async def _query_place_text(
        self,
        location_text: str,
        city: str | None,
        city_limit: bool,
    ) -> tuple[str | None, str | None]:
        params = {
            "key": self._api_key,
            "keywords": location_text,
            "offset": 10,
            "page": 1,
            "extensions": "base",
        }
        if city:
            params["city"] = city
        if city_limit:
            params["citylimit"] = "true"

        response = await self._client.get(self.PLACE_TEXT_URL, params=params)
        data = await request_json(response, "高德POI检索")
        if data.get("status") != "1":
            return None, None

        pois = data.get("pois") or []
        if not pois:
            return None, None

        candidate = self._pick_relevant_poi(pois, location_text=location_text, city_hint=city if city_limit else None)
        if candidate is None:
            return None, None

        location = candidate.get("location")
        if not location:
            return None, None
        return location, (candidate.get("cityname") or candidate.get("pname") or city)

    def _pick_relevant_poi(self, pois: list[dict], location_text: str, city_hint: str | None) -> dict | None:
        normalized_city_hint = self._normalize_city_name(city_hint)
        query_city_hint = self._normalize_city_name(self._extract_city_hint_from_location(location_text))
        city_hint_conflicts = bool(normalized_city_hint and query_city_hint and normalized_city_hint != query_city_hint)
        tokens = self._extract_query_tokens(location_text)

        for poi in pois:
            location = poi.get("location")
            if not location:
                continue

            if normalized_city_hint and not city_hint_conflicts:
                city_name = self._normalize_city_name(poi.get("cityname"))
                province_name = self._normalize_city_name(poi.get("pname"))
                if normalized_city_hint not in {city_name, province_name}:
                    continue

            combined_text = self._normalize_text(
                f"{poi.get('name') or ''}{poi.get('address') or ''}{poi.get('adname') or ''}{poi.get('cityname') or ''}"
            )
            if tokens and not any(token in combined_text for token in tokens):
                continue
            return poi

        if normalized_city_hint and not city_hint_conflicts:
            for poi in pois:
                location = poi.get("location")
                if not location:
                    continue
                city_name = self._normalize_city_name(poi.get("cityname"))
                province_name = self._normalize_city_name(poi.get("pname"))
                if normalized_city_hint in {city_name, province_name}:
                    return poi

        return pois[0] if pois else None

    @staticmethod
    def _contains_city_hint(location_text: str, city_hint: str) -> bool:
        normalized_city = city_hint.replace("市", "").strip().lower()
        normalized_location = location_text.replace("市", "").strip().lower()
        return bool(normalized_city and normalized_city in normalized_location)

    def _city_candidates(self, location_text: str, preferred_city: str | None) -> list[str]:
        candidates: list[str] = []
        extracted = self._extract_city_hint_from_location(location_text)
        for raw in (extracted, preferred_city, self._default_city):
            normalized = self._normalize_city_name(raw)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        return candidates

    @staticmethod
    def _extract_city_hint_from_location(location_text: str) -> str | None:
        normalized = re.sub(r"\s+", "", location_text.strip())
        normalized = re.sub(r"[()（）\[\]【】]", "/", normalized)
        match_city_suffix = re.search(r"(?P<city>[\u4e00-\u9fa5]{2,8})市", normalized)
        if match_city_suffix:
            return match_city_suffix.group("city")

        # e.g. 温州站 / 上海虹桥站 / 杭州东站 / 汉庭酒店(温州车站店)
        station_like_iter = re.finditer(
            r"(?<![\u4e00-\u9fa5])(?P<city>[\u4e00-\u9fa5]{2,3}?)(?:虹桥)?(?:东|南|西|北)?(?:车站|火车站|高铁站|站|机场)",
            normalized,
        )
        for match_station_like in station_like_iter:
            city = match_station_like.group("city")
            if city and city[-1] in AmapRouteProvider.CITY_ENDING_CHARS:
                return city

        # e.g. 温州汉庭酒店 / 杭州西湖 / 北京大学
        if len(normalized) >= 4 and re.match(r"^[\u4e00-\u9fa5]+$", normalized):
            for size in (2, 3, 4):
                if len(normalized) <= size:
                    continue
                city = normalized[:size]
                if city[-1] in AmapRouteProvider.CITY_ENDING_CHARS:
                    return city

        # e.g. 温州五马街 / 杭州西湖景区
        match_poi_like = re.match(
            r"(?P<city>[\u4e00-\u9fa5]{2,3}?)[\u4e00-\u9fa5]{1,8}(?:街|路|大道|老街|古镇|景区|广场|公园)$",
            normalized,
        )
        if match_poi_like:
            city = match_poi_like.group("city")
            if city and city[-1] in AmapRouteProvider.CITY_ENDING_CHARS:
                return city

        return None

    def _extract_query_tokens(self, location_text: str) -> list[str]:
        text = self._normalize_text(location_text)
        text = re.sub(r"[()（）\[\]【】,，。；;:：/\\\-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        stop_phrases = (
            "火车站",
            "高铁站",
            "地铁站",
            "公交站",
            "大学",
            "学院",
            "校区",
            "步行街",
            "老街",
            "大道",
            "机场",
            "车站",
            "广场",
            "公园",
            "景区",
            "景点",
            "街道",
            "街",
            "路",
            "站",
            "市",
            "区",
            "县",
            "镇",
            "村",
        )
        tokenized = text
        for phrase in stop_phrases:
            tokenized = tokenized.replace(phrase, " ")
        chunks = [chunk for chunk in tokenized.split(" ") if len(chunk) >= 2]
        if not chunks and len(text) >= 2:
            chunks = [text[:2]]
        elif len(text) >= 2:
            chunks.append(text[:2])
        unique: list[str] = []
        for chunk in chunks:
            if chunk not in unique:
                unique.append(chunk)
        return unique[:3]

    def _is_low_confidence_geocode(self, geocode: dict, location_text: str) -> bool:
        if not re.search(r"[\u4e00-\u9fa5]", location_text):
            return False
        combined = self._normalize_text(
            f"{geocode.get('formatted_address') or ''}{geocode.get('province') or ''}"
            f"{geocode.get('city') or ''}{geocode.get('district') or ''}"
        )
        tokens = self._extract_query_tokens(location_text)
        if not tokens:
            return False
        return not any(token in combined for token in tokens)

    @staticmethod
    def _looks_like_same_point_result(plans: list[RoutePlan]) -> bool:
        driving = next((plan for plan in plans if plan.mode == "driving"), None)
        walking = next((plan for plan in plans if plan.mode == "walking"), None)
        if driving is None or walking is None:
            return False

        driving_distance = driving.distance_km if driving.distance_km is not None else 0.0
        walking_distance = walking.distance_km if walking.distance_km is not None else 0.0
        return (
            driving.duration_minutes <= 2
            and walking.duration_minutes <= 2
            and driving_distance <= 0.1
            and walking_distance <= 0.1
        )

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        if not text:
            return ""
        normalized = str(text).strip().lower()
        normalized = re.sub(r"\s+", "", normalized)
        return normalized

    async def _walking(self, origin_lnglat: str, destination_lnglat: str) -> RoutePlan | None:
        response = await self._client.get(
            self.WALKING_URL,
            params={
                "key": self._api_key,
                "origin": origin_lnglat,
                "destination": destination_lnglat,
            },
        )
        data = await request_json(response, "高德步行路线")
        if data.get("status") != "1":
            return None

        paths = (data.get("route") or {}).get("paths") or []
        if not paths:
            return None

        first = paths[0]
        duration_minutes = max(1, round((to_int(first.get("duration")) or 0) / 60))
        distance_km = round((to_float(first.get("distance")) or 0.0) / 1000, 1)
        return RoutePlan(
            mode="walking",
            duration_minutes=duration_minutes,
            distance_km=distance_km,
            summary="步行可达，适合短距离通勤。",
        )

    async def _driving(self, origin_lnglat: str, destination_lnglat: str) -> RoutePlan | None:
        response = await self._client.get(
            self.DRIVING_URL,
            params={
                "key": self._api_key,
                "origin": origin_lnglat,
                "destination": destination_lnglat,
                "strategy": 0,
            },
        )
        data = await request_json(response, "高德驾车路线")
        if data.get("status") != "1":
            return None

        paths = (data.get("route") or {}).get("paths") or []
        if not paths:
            return None

        first = paths[0]
        duration_minutes = max(1, round((to_int(first.get("duration")) or 0) / 60))
        distance_km = round((to_float(first.get("distance")) or 0.0) / 1000, 1)
        tolls = to_float(first.get("tolls"))
        toll_text = f"，预计过路费约 {tolls:.0f} 元" if tolls else ""
        return RoutePlan(
            mode="driving",
            duration_minutes=duration_minutes,
            distance_km=distance_km,
            summary=f"驾车路线较直接{toll_text}。",
        )

    async def _transit(self, origin_lnglat: str, destination_lnglat: str, city: str | None) -> list[RoutePlan]:
        if not city:
            return []

        response = await self._client.get(
            self.TRANSIT_URL,
            params={
                "key": self._api_key,
                "origin": origin_lnglat,
                "destination": destination_lnglat,
                "city": city,
            },
        )
        data = await request_json(response, "高德公交路线")
        if data.get("status") != "1":
            return []

        transits = (data.get("route") or {}).get("transits") or []
        results: list[RoutePlan] = []
        for transit in transits[:2]:
            duration_minutes = max(1, round((to_int(transit.get("duration")) or 0) / 60))
            walk_distance = round((to_float(transit.get("walking_distance")) or 0.0) / 1000, 1)
            cost = to_float(transit.get("cost"))

            line_names: list[str] = []
            for segment in transit.get("segments") or []:
                bus = segment.get("bus") or {}
                for line in bus.get("buslines") or []:
                    name = (line.get("name") or "").split("(")[0].strip()
                    if name:
                        line_names.append(name)

            deduped_lines = list(dict.fromkeys(line_names))
            lines_text = " -> ".join(deduped_lines[:3]) if deduped_lines else "公交换乘"
            cost_text = f"，票价约 {cost:.0f} 元" if cost else ""

            results.append(
                RoutePlan(
                    mode="transit",
                    duration_minutes=duration_minutes,
                    distance_km=walk_distance if walk_distance > 0 else None,
                    summary=f"公交优先：{lines_text}{cost_text}。",
                )
            )

        return results

    def _resolve_transit_city(self, origin_city: str | None, destination_city: str | None) -> str | None:
        normalized_origin = self._normalize_city_name(origin_city)
        normalized_destination = self._normalize_city_name(destination_city)
        default_city = self._normalize_city_name(self._default_city)

        if normalized_origin and normalized_destination:
            if normalized_origin == normalized_destination:
                return normalized_origin
            return None
        if normalized_origin:
            return normalized_origin
        if normalized_destination:
            return normalized_destination
        return default_city

    @staticmethod
    def _extract_city_from_geocode(geocode: dict) -> str | None:
        city = geocode.get("city")
        if isinstance(city, str) and city.strip():
            return city.strip()
        province = geocode.get("province")
        if isinstance(province, str) and province.strip():
            return province.strip()
        return None

    @staticmethod
    def _normalize_city_name(city: str | None) -> str | None:
        if not city:
            return None
        normalized = city.replace("市", "").strip()
        return normalized or None
