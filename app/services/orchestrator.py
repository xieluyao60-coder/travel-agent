from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import date, timedelta
from typing import Any

from app.errors import ExternalAPIError, ServiceError, UserInputError, WeatherForecastRangeError
from app.schemas import (
    AssistantReply,
    IntentType,
    PlannerAction,
    PlannerOutput,
    PlannerToolName,
)
from app.services.formatter import (
    format_nearby_reply,
    format_route_reply,
    format_search_reply,
    format_unavailable_reply,
    format_weather_forecast_reply,
    format_weather_reply,
    format_weather_search_fallback_reply,
)
from app.services.session_store import SessionStore

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    REPLY_AGENT_TIMEOUT_SECONDS = 8.0
    SYSTEM_PROMPT = (
        "你是叫真由理，一个可爱元气、但非常靠谱的旅行助手少女。"
        "你说话轻快温暖、清楚直接，默认用简洁自然中文。"
        "实用性第一，先给结论再补充细节；用户着急时更简洁。"
        "口头禅是“嘟嘟噜”，默认自然出现一次。"
        "不要夸张卖萌，不要编造实时数据。"
        "当问题依赖实时信息时，优先引导用户使用天气、路线、搜索能力。"
    )

    SLOT_LABELS = {
        "origin": "起点",
        "destination": "终点",
        "location": "城市",
        "keyword": "周边类型",
        "query": "搜索主题",
        "travel_city": "旅行城市",
        "hotel_location": "酒店位置",
    }
    HOTEL_ALIASES = ("酒店", "住处", "宾馆", "旅馆")
    CITY_SUFFIXES = ("市", "州", "县", "区", "镇", "乡", "旗", "盟")

    def __init__(
        self,
        planner,
        reply_agent,
        session_store: SessionStore,
        weather_provider,
        route_provider,
        search_provider,
        llm_provider,
        history_limit: int,
        memory_enabled: bool = True,
    ) -> None:
        self._planner = planner
        self._reply_agent = reply_agent
        self._session_store = session_store
        self._weather_provider = weather_provider
        self._route_provider = route_provider
        self._search_provider = search_provider
        self._llm_provider = llm_provider
        self._history_limit = history_limit
        self._memory_enabled = memory_enabled

    async def handle(self, user_id: str, text: str, session_id: str) -> AssistantReply:
        start = time.perf_counter()
        plan: PlannerOutput | None = None
        reply: AssistantReply | None = None

        original_text = text.strip()
        scope_key = self._profile_scope_key(session_id=session_id, user_id=user_id)
        profile = await self._load_profile(scope_key) if self._memory_enabled else {}

        history = await self._session_store.get_history(session_id)
        pending = await self._session_store.get_pending(session_id)
        plan = await self._planner.plan(
            text=original_text,
            history=history,
            pending=pending,
            memory_hints=self._build_memory_hints(profile),
        )
        if self._memory_enabled:
            plan = self._apply_memory_to_plan(plan=plan, profile=profile)

        try:
            if plan.action == PlannerAction.CLARIFY:
                pending_payload = self._build_pending_payload(plan=plan, previous_pending=pending)
                await self._session_store.set_pending(session_id, pending_payload)
                reply = AssistantReply(intent=plan.intent, text=self._format_clarify_reply(plan))
            elif plan.action == PlannerAction.CALL_TOOL:
                await self._session_store.clear_pending(session_id)
                reply = await self._execute_tool(
                    plan=plan,
                    user_text=original_text,
                    scope_key=scope_key,
                    profile=profile,
                )
                if plan.intent == IntentType.NEARBY:
                    nearby_context = self._build_pending_payload(plan=plan, previous_pending=pending)
                    await self._session_store.set_pending(session_id, nearby_context)
            else:
                await self._session_store.clear_pending(session_id)
                reply = await self._handle_chat_fallback(
                    text=original_text,
                    session_id=session_id,
                    history=history,
                    profile=profile,
                )
        except UserInputError as exc:
            reply = AssistantReply(intent=plan.intent, text=format_unavailable_reply(str(exc)))
        except (ExternalAPIError, ServiceError) as exc:
            reply = AssistantReply(intent=plan.intent, text=format_unavailable_reply(str(exc)))
        except Exception:
            reply = AssistantReply(intent=plan.intent, text=format_unavailable_reply("上游服务暂时不可达"))

        await self._session_store.append(session_id, "user", text)
        await self._session_store.append(session_id, "assistant", reply.text)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "orchestrator done session=%s user=%s action=%s intent=%s tool=%s latency_ms=%s",
            session_id,
            user_id,
            plan.action.value if plan else "unknown",
            reply.intent.value if reply else "unknown",
            plan.tool_name.value if plan and plan.tool_name else "-",
            elapsed_ms,
        )
        return reply

    async def _execute_tool(
        self,
        *,
        plan: PlannerOutput,
        user_text: str,
        scope_key: str,
        profile: dict[str, Any],
    ) -> AssistantReply:
        tool_name = plan.tool_name or self._default_tool_name(plan.intent)
        if tool_name == PlannerToolName.WEATHER_NOW:
            return await self._execute_weather(
                plan=plan,
                user_text=user_text,
                scope_key=scope_key,
                profile=profile,
            )
        if tool_name == PlannerToolName.ROUTE_PLAN:
            return await self._execute_route(
                plan=plan,
                user_text=user_text,
                scope_key=scope_key,
                profile=profile,
            )
        if tool_name == PlannerToolName.SEARCH_WEB:
            return await self._execute_search(
                plan=plan,
                user_text=user_text,
                scope_key=scope_key,
                profile=profile,
            )
        if tool_name == PlannerToolName.NEARBY_SEARCH:
            return await self._execute_nearby(
                plan=plan,
                user_text=user_text,
                scope_key=scope_key,
                profile=profile,
            )
        if tool_name == PlannerToolName.MEMORY_UPDATE:
            return await self._execute_memory(
                plan=plan,
                user_text=user_text,
                scope_key=scope_key,
                profile=profile,
            )
        return await self._handle_chat_fallback(text=user_text, session_id="stateless", history=[])

    async def _load_profile(self, scope_key: str) -> dict[str, Any]:
        raw = await self._session_store.get_profile(scope_key)
        if not isinstance(raw, dict):
            return {}
        return self._normalize_profile_dict(raw)

    @staticmethod
    def _normalize_profile_dict(profile: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        travel_city = ChatOrchestrator._clean_memory_city(str(profile.get("travel_city") or ""))
        hotel_location = ChatOrchestrator._clean_memory_place(str(profile.get("hotel_location") or ""))
        updated_at = str(profile.get("updated_at") or "").strip()
        route_preference = profile.get("route_preference")
        route_pref_payload: dict[str, str] = {}
        if isinstance(route_preference, dict):
            mode = str(route_preference.get("mode") or "").strip().lower()
            goal = str(route_preference.get("goal") or "").strip().lower()
            if mode in {"transit", "driving", "walking"}:
                route_pref_payload["mode"] = mode
            if goal in {"fastest", "cheapest", "least_walking", "balanced"}:
                route_pref_payload["goal"] = goal

        if travel_city:
            normalized["travel_city"] = travel_city
        if hotel_location:
            normalized["hotel_location"] = hotel_location
        if route_pref_payload:
            normalized["route_preference"] = route_pref_payload
        if updated_at:
            normalized["updated_at"] = updated_at
        return normalized

    async def _save_profile(self, scope_key: str, profile: dict[str, Any]) -> None:
        if not self._memory_enabled:
            return
        payload = self._normalize_profile_dict(profile)
        if not payload:
            await self._session_store.clear_profile(scope_key)
            return
        payload["updated_at"] = str(int(time.time()))
        await self._session_store.set_profile(scope_key, payload)

    @staticmethod
    def _build_memory_hints(profile: dict[str, Any]) -> dict[str, Any]:
        hints: dict[str, Any] = {}
        travel_city = str(profile.get("travel_city") or "").strip()
        hotel_location = str(profile.get("hotel_location") or "").strip()
        route_preference = profile.get("route_preference")
        if travel_city:
            hints["travel_city"] = travel_city
        if hotel_location:
            hints["hotel_location"] = hotel_location
        if isinstance(route_preference, dict) and route_preference:
            hints["route_preference"] = route_preference
        return hints

    @staticmethod
    def _profile_scope_key(*, session_id: str, user_id: str) -> str:
        session = (session_id or "").strip().lower()
        if session.startswith("qq-") or session.startswith("qq:"):
            return f"qq:{user_id}"
        if session.startswith("wecom"):
            return f"wecom:{user_id}"

        platform = "default"
        if ":" in session:
            platform = session.split(":", 1)[0] or "default"
        return f"{platform}:{user_id}"

    async def _try_handle_memory_command(
        self,
        *,
        scope_key: str,
        text: str,
        profile: dict[str, Any],
    ) -> tuple[AssistantReply | None, dict[str, Any]]:
        if not self._memory_enabled:
            return None, profile

        content = (text or "").strip()
        if not content:
            return None, profile
        if re.search(r"[?？]|(在哪|哪里|多少|几[点个]|怎么|为何|为什么|是否)", content):
            return None, profile

        if re.fullmatch(r"(?:忘记酒店|清除酒店|删除酒店)\s*", content):
            updated = dict(profile)
            updated.pop("hotel_location", None)
            await self._save_profile(scope_key, updated)
            return AssistantReply(intent=IntentType.CHAT, text="已帮你忘记“酒店”位置，之后不会再自动替换啦。"), updated

        if re.fullmatch(r"(?:重置记忆|清空记忆|忘记我的偏好|忘记我偏好)\s*", content):
            await self._session_store.clear_profile(scope_key)
            return AssistantReply(intent=IntentType.CHAT, text="记忆已重置。我们可以从当前行程重新开始设置。"), {}

        updated = dict(profile)
        updates: list[str] = []
        had_command = False

        hotel_location = self._extract_hotel_location_from_command(content)
        if hotel_location is not None:
            had_command = True
            if not hotel_location:
                return AssistantReply(intent=IntentType.CHAT, text="酒店位置我还没听清，可以再说得具体一点吗？"), profile
            updated["hotel_location"] = hotel_location
            updates.append(f"酒店位置记为“{hotel_location}”")

        travel_city = self._extract_travel_city_from_command(content)
        if travel_city is not None:
            had_command = True
            if not travel_city:
                return AssistantReply(intent=IntentType.CHAT, text="你当前所在城市我还没听清，可以说成“我在温州旅行”。"), profile
            updated["travel_city"] = travel_city
            updates.append(f"默认出行城市记为“{travel_city}”")

        if updates:
            await self._save_profile(scope_key, updated)
            return (
                AssistantReply(intent=IntentType.CHAT, text=self._format_memory_update_reply(updates)),
                self._normalize_profile_dict(updated),
            )
        if had_command:
            return AssistantReply(intent=IntentType.CHAT, text="我明白你想更新记忆，但信息还不够完整。可以再说具体一点吗？"), profile

        return None, profile

    @staticmethod
    def _clean_memory_place(value: str) -> str:
        cleaned = re.sub(r"[，。！？,.!?；;：:\s]+", "", (value or "").strip())
        cleaned = re.sub(r"(这边|这里|那边|那里)$", "", cleaned)
        if cleaned in {"哪", "哪里", "在哪"}:
            return ""
        if re.fullmatch(r"\d{3,}", cleaned):
            return ""
        return cleaned.strip()

    @classmethod
    def _clean_memory_city(cls, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""
        raw = re.sub(r"^[在于]", "", raw)
        raw = re.split(r"[，,。；;！？?\s]", raw, maxsplit=1)[0]
        raw = re.sub(r"(旅行|旅游|出差|游玩|玩)$", "", raw)
        raw = re.sub(r"(这边|这里|那边|那里)$", "", raw)
        raw = raw.strip("（）()")
        if not raw:
            return ""
        if len(raw) > 12:
            return ""
        if re.search(r"(酒店|宾馆|旅馆|车站|机场|地铁|公交|大道|路|街|景区|公园|校区)", raw):
            return ""
        if not re.search(r"[\u4e00-\u9fa5]", raw):
            return ""
        return raw

    def _extract_hotel_location_from_command(self, content: str) -> str | None:
        direct_match = re.search(r"把(?P<place>[^，。；;！？?]+?)记为酒店", content)
        if direct_match:
            return self._clean_memory_place(direct_match.group("place"))

        hotel_match = re.search(r"(?:我的酒店在|我住在)\s*(?P<place>[^，。；;！？?]+)", content)
        if not hotel_match:
            return None
        return self._clean_memory_place(hotel_match.group("place"))

    def _extract_travel_city_from_command(self, content: str) -> str | None:
        patterns = (
            r"(?:^|[，,。；;！？?])\s*我(?:现在)?在(?P<city>[^\s，,。；;！？?]{2,12}?)(?:旅行|旅游|出差|游玩|玩)(?:$|[，,。；;！？?])",
            r"^\s*我(?:现在)?在(?P<city>[^\s，,。；;！？?]{2,12})\s*$",
        )
        for pattern in patterns:
            match = re.search(pattern, content)
            if not match:
                continue
            city = self._clean_memory_city(match.group("city"))
            return city
        return None

    @staticmethod
    def _format_memory_update_reply(updates: list[str]) -> str:
        if not updates:
            return "记忆已更新。"
        if len(updates) == 1:
            return f"记住啦，{updates[0]}。"
        return f"记住啦，{updates[0]}，同时{updates[1]}。"

    def _inject_memory_aliases(self, text: str, profile: dict[str, Any]) -> str:
        hotel_location = str(profile.get("hotel_location") or "").strip()
        if not hotel_location:
            return text
        content = text or ""
        for alias in self.HOTEL_ALIASES:
            content = re.sub(alias, hotel_location, content)
        return content.strip()

    def _apply_memory_to_plan(self, *, plan: PlannerOutput, profile: dict[str, Any]) -> PlannerOutput:
        if not self._memory_enabled:
            return plan
        travel_city = self._clean_memory_city(str(profile.get("travel_city") or ""))
        hotel_location = self._clean_memory_place(str(profile.get("hotel_location") or ""))
        route_preference = profile.get("route_preference")

        if plan.intent == IntentType.WEATHER:
            if not (plan.weather.location or "").strip() and travel_city:
                plan.weather.location = travel_city

        if plan.intent == IntentType.ROUTE:
            origin = (plan.route.origin or "").strip()
            destination = (plan.route.destination or "").strip()
            if hotel_location:
                if origin in self.HOTEL_ALIASES:
                    plan.route.origin = hotel_location
                    origin = hotel_location
                if destination in self.HOTEL_ALIASES:
                    plan.route.destination = hotel_location
                    destination = hotel_location

            inferred_city = self._extract_city_hint(origin) or self._extract_city_hint(destination)
            city_for_route = inferred_city or travel_city
            if city_for_route:
                if origin and not self._looks_city_qualified(origin, city_for_route):
                    plan.route.origin = f"{city_for_route}{origin}"
                if destination and not self._looks_city_qualified(destination, city_for_route):
                    plan.route.destination = f"{city_for_route}{destination}"

            if isinstance(route_preference, dict):
                pref_mode = str(route_preference.get("mode") or "").strip().lower()
                pref_goal = str(route_preference.get("goal") or "").strip().lower()
                if not plan.route.mode and pref_mode in {"transit", "driving", "walking"}:
                    plan.route.mode = pref_mode  # type: ignore[assignment]
                if not plan.route.goal and pref_goal in {"fastest", "cheapest", "least_walking", "balanced"}:
                    plan.route.goal = pref_goal  # type: ignore[assignment]

        if plan.intent == IntentType.NEARBY:
            location = (plan.nearby.location or "").strip()
            keyword = (plan.nearby.keyword or "").strip()
            if hotel_location and location in self.HOTEL_ALIASES:
                plan.nearby.location = hotel_location
                location = hotel_location
            if not location and hotel_location:
                plan.nearby.location = hotel_location
                location = hotel_location
            if not location and travel_city:
                plan.nearby.location = travel_city
                location = travel_city
            if location and travel_city and not self._looks_city_qualified(location, travel_city):
                plan.nearby.location = f"{travel_city}{location}"
            if keyword:
                plan.nearby.keyword = keyword
            if plan.nearby.radius_m is None:
                plan.nearby.radius_m = 1000

        if plan.action == PlannerAction.CLARIFY:
            if plan.intent == IntentType.WEATHER and (plan.weather.location or "").strip():
                plan.action = PlannerAction.CALL_TOOL
                plan.missing_slots = []
                plan.clarification_question = None
            elif plan.intent == IntentType.ROUTE:
                if (plan.route.origin or "").strip() and (plan.route.destination or "").strip():
                    plan.action = PlannerAction.CALL_TOOL
                    plan.missing_slots = []
                    plan.clarification_question = None
            elif plan.intent == IntentType.NEARBY:
                if (plan.nearby.location or "").strip() and (plan.nearby.keyword or "").strip():
                    plan.action = PlannerAction.CALL_TOOL
                    plan.missing_slots = []
                    plan.clarification_question = None
        return plan

    @staticmethod
    def _looks_city_qualified(place: str, city: str) -> bool:
        candidate = (place or "").strip()
        if not candidate:
            return True
        if city and city in candidate:
            return True
        if candidate.startswith(("上海", "北京", "天津", "重庆", "香港", "澳门")):
            return True
        if any(token in candidate for token in ("站", "机场", "火车站", "高铁站", "大学", "校区")):
            return True
        if re.match(r"^[\u4e00-\u9fa5]{2,4}(?:市|州|县|区)", candidate):
            return True
        if re.search(r"[省市区县州]", candidate[:4]):
            return True
        return False

    @staticmethod
    def _extract_city_hint(place: str) -> str | None:
        candidate = re.sub(r"\s+", "", (place or "").strip())
        if not candidate:
            return None

        for municipality in ("上海", "北京", "天津", "重庆", "香港", "澳门"):
            if candidate.startswith(municipality):
                return municipality

        city_match = re.match(r"^(?P<city>[\u4e00-\u9fa5]{2,4}?)(?:市|州|县|区)", candidate)
        if city_match:
            return city_match.group("city")

        station_like = re.match(r"^(?P<city>[\u4e00-\u9fa5]{2,3}?)(?:[\u4e00-\u9fa5]{0,4})(?:站|机场|火车站|高铁站)", candidate)
        if station_like:
            return station_like.group("city")
        return None

    async def _update_profile_after_weather(self, *, scope_key: str, profile: dict[str, Any], location: str) -> None:
        if not self._memory_enabled:
            return
        if str(profile.get("travel_city") or "").strip():
            return

        city_hint = self._extract_city_hint(location)
        if not city_hint:
            return
        updated = dict(profile)
        updated["travel_city"] = city_hint
        await self._save_profile(scope_key, updated)

    async def _update_profile_after_route(
        self,
        *,
        scope_key: str,
        profile: dict[str, Any],
        plan: PlannerOutput,
        origin: str,
        destination: str,
    ) -> None:
        if not self._memory_enabled:
            return

        updated = dict(profile)
        changed = False

        route_preference = updated.get("route_preference")
        route_pref_payload: dict[str, str] = dict(route_preference) if isinstance(route_preference, dict) else {}
        if plan.route.mode and route_pref_payload.get("mode") != plan.route.mode:
            route_pref_payload["mode"] = plan.route.mode
            changed = True
        if plan.route.goal and route_pref_payload.get("goal") != plan.route.goal:
            route_pref_payload["goal"] = plan.route.goal
            changed = True
        if route_pref_payload:
            updated["route_preference"] = route_pref_payload

        if not str(updated.get("travel_city") or "").strip():
            inferred_city = self._extract_city_hint(origin) or self._extract_city_hint(destination)
            if inferred_city:
                updated["travel_city"] = inferred_city
                changed = True

        if changed:
            await self._save_profile(scope_key, updated)

    @staticmethod
    def _build_profile_hint_for_chat(profile: dict[str, Any]) -> str:
        travel_city = str(profile.get("travel_city") or "").strip()
        hotel_location = str(profile.get("hotel_location") or "").strip()
        route_preference = profile.get("route_preference")
        bits: list[str] = []
        if travel_city:
            bits.append(f"用户当前默认旅行城市：{travel_city}")
        if hotel_location:
            bits.append(f"用户“酒店”代称指向：{hotel_location}")
        if isinstance(route_preference, dict):
            mode = str(route_preference.get("mode") or "").strip()
            goal = str(route_preference.get("goal") or "").strip()
            if mode:
                bits.append(f"用户常用通勤方式偏好：{mode}")
            if goal:
                bits.append(f"用户通勤目标偏好：{goal}")
        if not bits:
            return ""
        return "可参考的用户记忆：\n- " + "\n- ".join(bits)

    async def _execute_weather(
        self,
        *,
        plan: PlannerOutput,
        user_text: str,
        scope_key: str,
        profile: dict[str, Any],
    ) -> AssistantReply:
        location = (plan.weather.location or "").strip()
        if not location:
            raise UserInputError("请告诉我要查询天气的城市，例如“上海天气”。")

        when = (plan.weather.when or "realtime").strip()
        if when == "week":
            raise UserInputError("请给出具体日期，例如“上海3月24日天气”或“上海明天天气”。")

        if when == "realtime":
            weather = await self._weather_provider.now(location)
            fallback_text = format_weather_reply(weather)
            tool_payload = {
                "tool_name": PlannerToolName.WEATHER_NOW.value,
                "scenario": "weather_realtime",
                "location": location,
                "weather": weather.model_dump(mode="json"),
            }
            final_text = await self._compose_reply(
                user_text=user_text,
                plan=plan,
                tool_result=tool_payload,
                fallback_text=fallback_text,
            )
            await self._update_profile_after_weather(scope_key=scope_key, profile=profile, location=location)
            return AssistantReply(intent=IntentType.WEATHER, text=self._sanitize_search_like_reply(final_text, fallback_text=fallback_text))

        default_target_date = date.today() + timedelta(days=1) if when == "tomorrow" else None
        if when == "tomorrow":
            try:
                forecast = await self._weather_provider.forecast(location, days_ahead=1)
                fallback_text = format_weather_forecast_reply(forecast)
                tool_payload = {
                    "tool_name": PlannerToolName.WEATHER_NOW.value,
                    "scenario": "weather_forecast",
                    "location": location,
                    "when": "tomorrow",
                    "forecast": forecast.model_dump(mode="json"),
                }
                final_text = await self._compose_reply(
                    user_text=user_text,
                    plan=plan,
                    tool_result=tool_payload,
                    fallback_text=fallback_text,
                )
                await self._update_profile_after_weather(scope_key=scope_key, profile=profile, location=location)
                return AssistantReply(intent=IntentType.WEATHER, text=self._sanitize_search_like_reply(final_text, fallback_text=fallback_text))
            except WeatherForecastRangeError as exc:
                return await self._execute_weather_search_fallback(
                    user_text=user_text,
                    location=location,
                    target_date=default_target_date,
                    range_error=exc,
                )

        target_date: date | None = None
        if plan.weather.target_date:
            try:
                target_date = date.fromisoformat(plan.weather.target_date)
            except ValueError as exc:
                raise UserInputError("日期格式无法识别，请改成“3.24”或“2026-03-24”。") from exc

        if when == "date":
            if target_date is None:
                raise UserInputError("请给出具体日期，例如“上海3月24日天气”。")
            try:
                forecast = await self._weather_provider.forecast(location, target_date=target_date)
                fallback_text = format_weather_forecast_reply(forecast)
                tool_payload = {
                    "tool_name": PlannerToolName.WEATHER_NOW.value,
                    "scenario": "weather_forecast",
                    "location": location,
                    "when": "date",
                    "target_date": target_date.isoformat(),
                    "forecast": forecast.model_dump(mode="json"),
                }
                final_text = await self._compose_reply(
                    user_text=user_text,
                    plan=plan,
                    tool_result=tool_payload,
                    fallback_text=fallback_text,
                )
                await self._update_profile_after_weather(scope_key=scope_key, profile=profile, location=location)
                return AssistantReply(intent=IntentType.WEATHER, text=self._sanitize_search_like_reply(final_text, fallback_text=fallback_text))
            except WeatherForecastRangeError as exc:
                return await self._execute_weather_search_fallback(
                    user_text=user_text,
                    location=location,
                    target_date=target_date,
                    range_error=exc,
                )

        weather = await self._weather_provider.now(location)
        fallback_text = format_weather_reply(weather)
        tool_payload = {
            "tool_name": PlannerToolName.WEATHER_NOW.value,
            "scenario": "weather_realtime",
            "location": location,
            "weather": weather.model_dump(mode="json"),
        }
        final_text = await self._compose_reply(
            user_text=user_text,
            plan=plan,
            tool_result=tool_payload,
            fallback_text=fallback_text,
        )
        await self._update_profile_after_weather(scope_key=scope_key, profile=profile, location=location)
        return AssistantReply(intent=IntentType.WEATHER, text=self._sanitize_search_like_reply(final_text, fallback_text=fallback_text))

    async def _execute_weather_search_fallback(
        self,
        *,
        user_text: str,
        location: str,
        target_date: date | None,
        range_error: WeatherForecastRangeError,
    ) -> AssistantReply:
        target_label = target_date.isoformat() if target_date else "目标日期"
        query = f"{location} {target_label} 天气预报 15天 40天"
        results = await self._search_provider.search(query=query, top_k=5)
        reliable_results = self._filter_weather_reliable_results(
            results=results,
            location=location,
            target_date=target_date,
        )
        detail_results = self._filter_results_with_weather_detail(reliable_results)
        focus = self._infer_weather_focus(user_text)
        temp_samples = self._extract_temperature_samples(detail_results) if focus == "temperature" else []
        temperature_estimate = self._summarize_temperature_samples(temp_samples)

        fallback_text, references = format_weather_search_fallback_reply(
            location=location,
            target_date_text=target_label,
            available_start=range_error.start_date,
            available_end=range_error.end_date,
            reliable_results=detail_results,
            focus=focus,
            temperature_estimate=temperature_estimate,
        )

        tool_payload = {
            "tool_name": PlannerToolName.SEARCH_WEB.value,
            "scenario": "weather_over_range",
            "query": query,
            "location": location,
            "target_date": target_label,
            "focus": focus,
            "temperature_estimate": temperature_estimate,
            "temperature_samples": [{"low": low, "high": high} for low, high in temp_samples],
            "has_weather_detail": bool(detail_results),
            "available_range": {
                "start": range_error.start_date,
                "end": range_error.end_date,
            },
            "result_count": len(detail_results),
            "results": self._compact_search_results(detail_results, top_k=3),
        }
        final_text = await self._compose_reply(
            user_text=user_text,
            plan=PlannerOutput(
                action=PlannerAction.CALL_TOOL,
                intent=IntentType.WEATHER,
                tool_name=PlannerToolName.SEARCH_WEB,
                normalized_query=user_text,
                confidence=1.0,
            ),
            tool_result=tool_payload,
            fallback_text=fallback_text,
        )
        final_text = self._sanitize_search_like_reply(final_text, fallback_text=fallback_text)
        final_text = self._ensure_weather_detail_guard(
            text=final_text,
            location=location,
            target_label=target_label,
            focus=focus,
            has_weather_detail=bool(detail_results),
        )
        final_text = self._ensure_weather_over_range_note(
            final_text,
            available_start=range_error.start_date,
            available_end=range_error.end_date,
        )
        final_text = self._ensure_weather_over_range_focus_guard(
            text=final_text,
            location=location,
            target_label=target_label,
            focus=focus,
            temperature_estimate=temperature_estimate,
        )
        safe_references = references if detail_results else []
        return AssistantReply(intent=IntentType.WEATHER, text=final_text, references=safe_references)

    async def _execute_route(
        self,
        *,
        plan: PlannerOutput,
        user_text: str,
        scope_key: str,
        profile: dict[str, Any],
    ) -> AssistantReply:
        origin = (plan.route.origin or "").strip()
        destination = (plan.route.destination or "").strip()
        if not origin or not destination:
            raise UserInputError("请明确起点和终点，例如“从虹桥站到外滩怎么走”。")

        plans = await self._route_provider.plan(origin, destination, mode=plan.route.mode)
        fallback_text = format_route_reply(origin, destination, plans, goal=plan.route.goal)
        tool_payload = {
            "tool_name": PlannerToolName.ROUTE_PLAN.value,
            "origin": origin,
            "destination": destination,
            "goal": plan.route.goal or "balanced",
            "plans": [item.model_dump(mode="json") for item in plans],
        }
        final_text = await self._compose_reply(
            user_text=user_text,
            plan=plan,
            tool_result=tool_payload,
            fallback_text=fallback_text,
        )
        await self._update_profile_after_route(
            scope_key=scope_key,
            profile=profile,
            plan=plan,
            origin=origin,
            destination=destination,
        )
        return AssistantReply(intent=IntentType.ROUTE, text=final_text)

    async def _execute_search(
        self,
        *,
        plan: PlannerOutput,
        user_text: str,
        scope_key: str,
        profile: dict[str, Any],
    ) -> AssistantReply:
        query = (plan.search.query or plan.normalized_query or user_text).strip()
        if not query:
            raise UserInputError("请补充你想搜索的主题。")

        top_k = max(1, min(int(plan.search.top_k or 5), 5))
        results = await self._search_provider.search(query=query, top_k=top_k)
        fallback_text, references = format_search_reply(query, results)
        tool_payload = {
            "tool_name": PlannerToolName.SEARCH_WEB.value,
            "query": query,
            "result_count": len(results),
            "results": self._compact_search_results(results, top_k=3),
        }
        final_text = await self._compose_reply(
            user_text=user_text,
            plan=plan,
            tool_result=tool_payload,
            fallback_text=fallback_text,
        )
        final_text = self._sanitize_search_like_reply(final_text, fallback_text=fallback_text)
        return AssistantReply(intent=IntentType.SEARCH, text=final_text, references=references)

    async def _execute_nearby(
        self,
        *,
        plan: PlannerOutput,
        user_text: str,
        scope_key: str,
        profile: dict[str, Any],
    ) -> AssistantReply:
        location = (plan.nearby.location or "").strip()
        keyword = (plan.nearby.keyword or "").strip()
        radius_m = max(100, min(int(plan.nearby.radius_m or 1000), 5000))
        if not location:
            raise UserInputError("请告诉我要在哪个地点附近查，比如“温州站附近”。")
        if not keyword:
            raise UserInputError("请补充你要找的类型，例如“咖啡店”“便利店”。")

        places = await self._route_provider.nearby(
            location_text=location,
            keyword=keyword,
            radius_m=radius_m,
            limit=5,
        )
        fallback_text = format_nearby_reply(location=location, keyword=keyword, radius_m=radius_m, places=places)
        tool_payload = {
            "tool_name": PlannerToolName.NEARBY_SEARCH.value,
            "location": location,
            "keyword": keyword,
            "radius_m": radius_m,
            "result_count": len(places),
            "places": [item.model_dump(mode="json") for item in places],
        }
        final_text = await self._compose_reply(
            user_text=user_text,
            plan=plan,
            tool_result=tool_payload,
            fallback_text=fallback_text,
        )
        await self._update_profile_after_weather(scope_key=scope_key, profile=profile, location=location)
        return AssistantReply(intent=IntentType.NEARBY, text=final_text)

    async def _execute_memory(
        self,
        *,
        plan: PlannerOutput,
        user_text: str,
        scope_key: str,
        profile: dict[str, Any],
    ) -> AssistantReply:
        operation = (plan.memory.operation or "").strip()
        updated = dict(profile)
        if operation == "clear_hotel":
            updated.pop("hotel_location", None)
            await self._save_profile(scope_key, updated)
            return AssistantReply(intent=IntentType.MEMORY, text="记住啦，我已经把你的酒店位置从记忆里清除了。")
        if operation == "reset_profile":
            await self._session_store.clear_profile(scope_key)
            return AssistantReply(intent=IntentType.MEMORY, text="记忆已重置，我们可以从当前行程重新开始。")
        if operation == "set_hotel":
            place = self._clean_memory_place(str(plan.memory.hotel_location or ""))
            if not place or place in {"哪", "哪里", "在哪"}:
                raise UserInputError("酒店位置我还没听清，可以再说得具体一点吗？")
            updated["hotel_location"] = place
            city = self._clean_memory_city(str(plan.memory.travel_city or ""))
            if city:
                updated["travel_city"] = city
            await self._save_profile(scope_key, updated)
            if city:
                return AssistantReply(
                    intent=IntentType.MEMORY,
                    text=f"记住啦，酒店记为“{place}”，默认出行城市也更新为“{city}”。",
                )
            return AssistantReply(intent=IntentType.MEMORY, text=f"记住啦，后面提到“酒店”我会按“{place}”来理解。")
        if operation == "set_city":
            city = self._clean_memory_city(str(plan.memory.travel_city or ""))
            if not city:
                raise UserInputError("你当前所在城市我还没听清，可以说成“我在温州旅行”。")
            updated["travel_city"] = city
            await self._save_profile(scope_key, updated)
            return AssistantReply(intent=IntentType.MEMORY, text=f"收到，我先把你的默认出行城市记为“{city}”。")

        # 兜底：解析结果未给出 operation 时，再用旧解析器兜底识别。
        memory_reply, _ = await self._try_handle_memory_command(scope_key=scope_key, text=user_text, profile=profile)
        if memory_reply is not None:
            return AssistantReply(intent=IntentType.MEMORY, text=memory_reply.text)
        raise UserInputError("这条消息不像记忆写入命令，我先不改你的记忆。")

    async def _handle_chat_fallback(
        self,
        text: str,
        session_id: str,
        history: list[dict[str, str]] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> AssistantReply:
        if self._should_force_search(text):
            try:
                results = await self._search_provider.search(query=text, top_k=3)
                fallback_text, references = format_search_reply(text, results)
                reply_text = self._sanitize_search_like_reply(fallback_text, fallback_text=fallback_text)
                return AssistantReply(intent=IntentType.SEARCH, text=reply_text, references=references)
            except Exception:
                pass

        usable_history = history if history is not None else await self._session_store.get_history(session_id)
        llm_messages = self._build_llm_messages(usable_history, text, profile=profile or {})
        fallback_text = "我在呢。你可以直接告诉我想查天气、路线、附近地点，或者更新你的旅行记忆。"
        try:
            fallback_text = await self._llm_provider.chat(llm_messages)
        except ExternalAPIError:
            try:
                results = await self._search_provider.search(query=text, top_k=3)
                fallback_text, references = format_search_reply(text, results)
                reply_text = self._sanitize_search_like_reply(fallback_text, fallback_text=fallback_text)
                return AssistantReply(intent=IntentType.SEARCH, text=reply_text, references=references)
            except Exception as exc:
                raise ExternalAPIError(str(exc)) from exc

        plan = PlannerOutput(
            action=PlannerAction.CHAT,
            intent=IntentType.CHAT,
            tool_name=None,
            normalized_query=text,
            confidence=1.0,
        )
        tool_payload = {
            "tool_name": "none",
            "scenario": "no_tool_direct_chat",
            "can_answer_directly": True,
            "has_tool_result": False,
            "memory_hints": self._build_memory_hints(profile or {}),
        }
        final_text = await self._compose_reply(
            user_text=text,
            plan=plan,
            tool_result=tool_payload,
            fallback_text=fallback_text,
        )
        return AssistantReply(intent=IntentType.CHAT, text=final_text)

    def _build_llm_messages(
        self,
        history: list[dict[str, str]],
        user_text: str,
        *,
        profile: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if self._memory_enabled and profile:
            memory_hint = self._build_profile_hint_for_chat(profile)
            if memory_hint:
                messages.append({"role": "system", "content": memory_hint})
        messages.extend(history[-self._history_limit :])
        messages.append({"role": "user", "content": user_text})
        return messages

    async def _compose_reply(
        self,
        *,
        user_text: str,
        plan: PlannerOutput,
        tool_result: dict[str, Any],
        fallback_text: str,
    ) -> str:
        try:
            candidate = await asyncio.wait_for(
                self._reply_agent.compose(
                    user_text=user_text,
                    plan=plan,
                    tool_result=tool_result,
                    fallback_text=fallback_text,
                ),
                timeout=self.REPLY_AGENT_TIMEOUT_SECONDS,
            )
            return self._guard_low_quality_reply(candidate, tool_result=tool_result, fallback_text=fallback_text)
        except asyncio.TimeoutError:
            logger.warning("reply agent timeout, fallback used")
            return fallback_text
        except Exception as exc:
            logger.warning("reply agent failed, fallback used: %s", exc)
            return fallback_text

    def _build_pending_payload(
        self,
        plan: PlannerOutput,
        previous_pending: dict[str, Any] | None,
    ) -> dict[str, object]:
        known_slots: dict[str, str] = {}
        if isinstance(previous_pending, dict):
            previous_known = previous_pending.get("known_slots")
            if isinstance(previous_known, dict):
                for key, value in previous_known.items():
                    if isinstance(key, str) and isinstance(value, str) and value.strip():
                        known_slots[key] = value.strip()

        if plan.intent == IntentType.ROUTE:
            if (plan.route.origin or "").strip():
                known_slots["origin"] = plan.route.origin.strip()
            if (plan.route.destination or "").strip():
                known_slots["destination"] = plan.route.destination.strip()
            if plan.route.mode:
                known_slots["mode"] = plan.route.mode
            if plan.route.goal:
                known_slots["goal"] = plan.route.goal
            missing_slots = self._missing_route_slots(known_slots)
        elif plan.intent == IntentType.WEATHER:
            if (plan.weather.location or "").strip():
                known_slots["location"] = plan.weather.location.strip()
            if (plan.weather.target_date or "").strip():
                known_slots["target_date"] = plan.weather.target_date.strip()
            missing_slots = [] if known_slots.get("location") else ["location"]
        elif plan.intent == IntentType.SEARCH:
            if (plan.search.query or "").strip():
                known_slots["query"] = plan.search.query.strip()
            missing_slots = [] if known_slots.get("query") else ["query"]
        elif plan.intent == IntentType.NEARBY:
            if (plan.nearby.location or "").strip():
                known_slots["location"] = plan.nearby.location.strip()
            if (plan.nearby.keyword or "").strip():
                known_slots["keyword"] = plan.nearby.keyword.strip()
            if plan.nearby.radius_m is not None:
                known_slots["radius_m"] = str(max(100, min(int(plan.nearby.radius_m), 5000)))
            missing_slots = []
            if not known_slots.get("location"):
                missing_slots.append("location")
            if not known_slots.get("keyword"):
                missing_slots.append("keyword")
        elif plan.intent == IntentType.MEMORY:
            if plan.memory.travel_city:
                known_slots["travel_city"] = plan.memory.travel_city
            if plan.memory.hotel_location:
                known_slots["hotel_location"] = plan.memory.hotel_location
            missing_slots = []
            if plan.memory.operation == "set_city" and not known_slots.get("travel_city"):
                missing_slots.append("travel_city")
            if plan.memory.operation == "set_hotel" and not known_slots.get("hotel_location"):
                missing_slots.append("hotel_location")
        else:
            missing_slots = list(plan.missing_slots)

        if plan.missing_slots:
            merged_missing = list(dict.fromkeys([*plan.missing_slots, *missing_slots]))
        else:
            merged_missing = missing_slots

        return {
            "intent": plan.intent.value,
            "tool_name": (plan.tool_name.value if plan.tool_name else None),
            "known_slots": known_slots,
            "missing_slots": merged_missing,
            "clarification_question": plan.clarification_question or "",
        }

    @staticmethod
    def _missing_route_slots(known_slots: dict[str, str]) -> list[str]:
        missing: list[str] = []
        if not (known_slots.get("origin") or "").strip():
            missing.append("origin")
        if not (known_slots.get("destination") or "").strip():
            missing.append("destination")
        return missing

    def _format_clarify_reply(self, plan: PlannerOutput) -> str:
        missing_labels = [self.SLOT_LABELS.get(slot, slot) for slot in plan.missing_slots]
        missing_text = "、".join(missing_labels) if missing_labels else "关键信息不足"
        question = (plan.clarification_question or "").strip() or "请再补充一点信息，我马上继续。"
        return f"我还差一点信息才能继续处理：目前缺少 {missing_text}。{question}"

    @staticmethod
    def _should_force_search(text: str) -> bool:
        search_signals = ("最新", "实时", "新闻", "突发", "今天", "近期")
        return any(signal in text for signal in search_signals)

    @staticmethod
    def _default_tool_name(intent: IntentType) -> PlannerToolName | None:
        if intent == IntentType.WEATHER:
            return PlannerToolName.WEATHER_NOW
        if intent == IntentType.ROUTE:
            return PlannerToolName.ROUTE_PLAN
        if intent == IntentType.SEARCH:
            return PlannerToolName.SEARCH_WEB
        if intent == IntentType.NEARBY:
            return PlannerToolName.NEARBY_SEARCH
        if intent == IntentType.MEMORY:
            return PlannerToolName.MEMORY_UPDATE
        return None

    @staticmethod
    def _sanitize_search_like_reply(text: str, *, fallback_text: str) -> str:
        clean = re.sub(r"https?://\S+", "", (text or "")).strip()
        if not clean:
            clean = fallback_text.strip()
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        return clean.strip()

    @staticmethod
    def _ensure_weather_over_range_note(text: str, *, available_start: str, available_end: str) -> str:
        range_prefix = f"和风天气当前可预报范围是 {available_start} 到 {available_end}"
        range_hint = f"{range_prefix}。"
        note = "超过7天，天气数据来自于网络，不一定准确。"
        clean = text.strip()
        additions: list[str] = []
        if range_prefix not in clean:
            additions.append(range_hint)
        if note not in clean:
            additions.append(note)
        if additions:
            clean = f"{clean}\n\n{' '.join(additions)}" if clean else " ".join(additions)
        return clean.strip()

    @staticmethod
    def _filter_weather_reliable_results(
        *,
        results: list,
        location: str,
        target_date: date | None,
    ) -> list:
        date_tokens = ChatOrchestrator._build_weather_date_tokens(target_date)
        require_target_date = target_date is not None
        matched: list[tuple[int, Any]] = []

        for item in results:
            text = f"{getattr(item, 'title', '')} {getattr(item, 'snippet', '')}"
            date_hit = ChatOrchestrator._has_target_date_token(text=text, date_tokens=date_tokens)
            if require_target_date and not date_hit:
                continue
            score = ChatOrchestrator._weather_result_score(text=text, location=location, date_tokens=date_tokens)
            if score >= 6:
                matched.append((score, item))

        matched.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in matched]

    @staticmethod
    def _filter_results_with_weather_detail(results: list) -> list:
        filtered: list = []
        for item in results:
            text = f"{getattr(item, 'title', '')} {getattr(item, 'snippet', '')}"
            if ChatOrchestrator._has_weather_detail(text):
                filtered.append(item)
        return filtered

    @staticmethod
    def _weather_result_score(*, text: str, location: str, date_tokens: set[str]) -> int:
        normalized = " ".join((text or "").split()).lower()
        location_hit = location.lower() in normalized if location else False
        date_hit = ChatOrchestrator._has_target_date_token(text=normalized, date_tokens=date_tokens)
        weather_hit = any(token in normalized for token in ("天气", "预报", "气温", "降雨", "温度"))
        long_range_hit = any(token in normalized for token in ("15天", "30天", "40天", "未来", "长期"))
        history_hit = any(token in normalized for token in ("历史天气", "历史", "往年", "回顾"))

        score = 0
        if location_hit:
            score += 3
        if date_hit:
            score += 3
        if weather_hit:
            score += 2
        if long_range_hit:
            score += 1
        if history_hit:
            score -= 3
        return score

    @staticmethod
    def _has_weather_detail(text: str) -> bool:
        content = " ".join((text or "").split())
        if not content:
            return False

        range_pattern = re.compile(r"(-?\d{1,2})\s*(?:~|-|至|到|/)\s*(-?\d{1,2})\s*(?:°?\s*[cC]|℃|度)?")
        temp_pattern = re.compile(r"-?\d{1,2}\s*(?:°?\s*[cC]|℃|度)")
        date_pattern = re.compile(r"(?:\d{4}[-/.])?\d{1,2}[-/.月]\d{1,2}(?:日|号)?")

        has_range = bool(range_pattern.search(content))
        has_temp = bool(temp_pattern.search(content))
        has_date = bool(date_pattern.search(content))
        has_condition = any(token in content for token in ("晴", "阴", "多云", "小雨", "中雨", "大雨", "阵雨", "雷", "雪", "风"))

        has_high_low = bool(re.search(r"最高(?:气温)?\D{0,4}-?\d{1,2}", content)) and bool(
            re.search(r"最低(?:气温)?\D{0,4}-?\d{1,2}", content)
        )

        if has_high_low:
            return True
        if has_range and (has_date or has_condition):
            return True
        if has_temp and has_date and has_condition:
            return True
        return False

    @staticmethod
    def _has_target_date_token(*, text: str, date_tokens: set[str]) -> bool:
        normalized = (text or "").lower()
        return any(token and token.lower() in normalized for token in date_tokens)

    @staticmethod
    def _build_weather_date_tokens(target_date: date | None) -> set[str]:
        if target_date is None:
            return set()

        year = target_date.year
        month = target_date.month
        day = target_date.day
        mm = f"{month:02d}"
        dd = f"{day:02d}"

        return {
            f"{year}-{mm}-{dd}",
            f"{year}/{mm}/{dd}",
            f"{year}.{mm}.{dd}",
            f"{month}/{day}",
            f"{mm}/{dd}",
            f"{month}.{day}",
            f"{mm}.{dd}",
            f"{month}月{day}日",
            f"{month}月{day}号",
        }

    @staticmethod
    def _infer_weather_focus(user_text: str) -> str:
        text = user_text.lower()
        if any(token in text for token in ("温度", "气温", "最高温", "最低温", "几度", "多少度")):
            return "temperature"
        return "general"

    @staticmethod
    def _extract_temperature_samples(results: list) -> list[tuple[int, int]]:
        patterns = (
            re.compile(r"(-?\d{1,2})\s*[~\-至到]\s*(-?\d{1,2})\s*(?:°?\s*[cC]|℃|度)"),
            re.compile(r"(-?\d{1,2})\s*/\s*(-?\d{1,2})\s*(?:°?\s*[cC]|℃)"),
            re.compile(r"(-?\d{1,2})\s*(?:°?\s*[cC]|℃)\s*/\s*(-?\d{1,2})\s*(?:°?\s*[cC]|℃)"),
        )
        unique: set[tuple[int, int]] = set()

        for item in results:
            text = f"{getattr(item, 'title', '')} {getattr(item, 'snippet', '')}"
            for pattern in patterns:
                for match in pattern.finditer(text):
                    left = int(match.group(1))
                    right = int(match.group(2))
                    low = min(left, right)
                    high = max(left, right)
                    if -40 <= low <= 60 and -40 <= high <= 60:
                        unique.add((low, high))

            high_match = re.search(r"最高(?:气温)?\s*(-?\d{1,2})\s*(?:°?\s*[cC]|℃|度)", text)
            low_match = re.search(r"最低(?:气温)?\s*(-?\d{1,2})\s*(?:°?\s*[cC]|℃|度)", text)
            if high_match and low_match:
                high = int(high_match.group(1))
                low = int(low_match.group(1))
                if -40 <= low <= 60 and -40 <= high <= 60:
                    unique.add((min(low, high), max(low, high)))

        return sorted(unique)

    @staticmethod
    def _summarize_temperature_samples(samples: list[tuple[int, int]]) -> str | None:
        if not samples:
            return None
        low = min(item[0] for item in samples)
        high = max(item[1] for item in samples)
        if low > high:
            return None
        return f"{low}~{high}°C"

    @staticmethod
    def _ensure_weather_over_range_focus_guard(
        *,
        text: str,
        location: str,
        target_label: str,
        focus: str,
        temperature_estimate: str | None,
    ) -> str:
        if focus != "temperature":
            return text
        clean = text.strip()
        if temperature_estimate:
            if temperature_estimate not in clean:
                prefix = f"按联网长周期预报看，{location}{target_label}温度大致在 {temperature_estimate}（仅供参考）。"
                clean = f"{prefix}\n{clean}" if clean else prefix
            if "仅供参考" not in clean:
                clean = f"{clean}\n仅供参考。"
            return clean.strip()

        if all(token not in clean for token in ("无法给出", "没法给出", "不确定", "不能给出")):
            prefix = f"目前我没法给出{location}{target_label}的可靠具体温度。"
            clean = f"{prefix}\n{clean}" if clean else prefix
        if "未给出该日期明确温度数值" not in clean:
            clean = f"{clean}\n这次检索结果未给出该日期明确温度数值。".strip()
        return clean.strip()

    @staticmethod
    def _ensure_weather_detail_guard(
        *,
        text: str,
        location: str,
        target_label: str,
        focus: str,
        has_weather_detail: bool,
    ) -> str:
        if has_weather_detail:
            return text
        if focus == "temperature":
            prefix = f"目前我没法给出{location}{target_label}的可靠具体温度。"
            if prefix in text or "明确温度数值" in text:
                return text
            return f"{prefix}\n{text}".strip()

        prefix = f"目前还没检索到与{location}{target_label}强相关、可直接采用的天气细节。"
        if prefix in text or any(token in text for token in ("强相关", "匹配度不够", "没检索到")):
            return text
        return f"{prefix}\n{text}".strip()

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        normalized = (text or "").lower()
        normalized = normalized.replace("，", ",").replace("。", ".").replace("！", "!").replace("？", "?")
        normalized = re.sub(r"\s+", "", normalized)
        return normalized

    @staticmethod
    def _extract_minutes_from_reply(text: str) -> list[int]:
        values: list[int] = []
        seen: set[int] = set()
        pattern = re.compile(r"(?:(\d{1,2})\s*小时(?:\s*(\d{1,2})\s*分(?:钟)?)?|(\d{1,3})\s*分钟)")
        for match in pattern.finditer(text):
            minutes: int | None = None
            if match.group(1):
                hours = int(match.group(1))
                mins = int(match.group(2) or 0)
                minutes = hours * 60 + mins
            elif match.group(3):
                minutes = int(match.group(3))

            if minutes is None:
                continue
            if minutes < 1 or minutes > 24 * 60:
                continue
            if minutes in seen:
                continue
            seen.add(minutes)
            values.append(minutes)
        return values

    @staticmethod
    def _extract_distance_km_from_reply(text: str) -> list[float]:
        values: list[float] = []
        seen: set[float] = set()
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:km|公里)", text.lower()):
            value = round(float(match.group(1)), 1)
            if value < 0 or value > 5000:
                continue
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
        return values

    @staticmethod
    def _compact_search_results(results: list[Any], *, top_k: int = 3) -> list[dict[str, str]]:
        compact: list[dict[str, str]] = []
        for item in results[: max(1, top_k)]:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
            else:
                title = str(getattr(item, "title", "") or "").strip()
                snippet = str(getattr(item, "snippet", "") or "").strip()

            if not title and not snippet:
                continue
            if len(snippet) > 140:
                snippet = f"{snippet[:140]}..."
            compact.append({"title": title, "snippet": snippet})
        return compact

    @staticmethod
    def _is_route_reply_grounded(text: str, tool_result: dict[str, Any]) -> bool:
        plans = tool_result.get("plans") if isinstance(tool_result.get("plans"), list) else []
        if not plans:
            return False

        durations: list[int] = []
        distances: list[float] = []
        modes: set[str] = set()
        for item in plans:
            if not isinstance(item, dict):
                continue
            duration = item.get("duration_minutes")
            if isinstance(duration, int):
                durations.append(duration)
            elif isinstance(duration, (float, str)):
                try:
                    durations.append(int(float(duration)))
                except ValueError:
                    pass

            distance = item.get("distance_km")
            if isinstance(distance, (float, int)):
                distances.append(float(distance))
            elif isinstance(distance, str):
                try:
                    distances.append(float(distance))
                except ValueError:
                    pass

            mode = str(item.get("mode") or "").strip().lower()
            if mode:
                modes.add(mode)

        if not durations:
            return False

        mentioned_minutes = ChatOrchestrator._extract_minutes_from_reply(text)
        if mentioned_minutes:
            for value in mentioned_minutes:
                if not any(abs(value - expected) <= 8 for expected in durations):
                    return False
            if any(token in text for token in ("最快", "最短", "最省时")):
                fastest = min(durations)
                if min(abs(value - fastest) for value in mentioned_minutes) > 8:
                    return False

        mentioned_distances = ChatOrchestrator._extract_distance_km_from_reply(text)
        if mentioned_distances and distances:
            rounded_distances = [round(item, 1) for item in distances]
            for value in mentioned_distances:
                if not any(abs(value - expected) <= 1.0 for expected in rounded_distances):
                    return False

        lowered = text.lower()
        mode_keywords = {
            "transit": ("transit", "公交", "地铁"),
            "driving": ("driving", "驾车", "开车", "打车"),
            "walking": ("walking", "步行", "走路"),
        }
        for mode, tokens in mode_keywords.items():
            if any(token in lowered or token in text for token in tokens) and mode not in modes:
                return False

        if not mentioned_minutes and not mentioned_distances and not any(
            token in text for token in ("路线", "方案", "公交", "地铁", "驾车", "步行", "通勤", "小时", "分钟")
        ):
            return False

        return True

    @staticmethod
    def _guard_low_quality_reply(text: str, *, tool_result: dict[str, Any], fallback_text: str) -> str:
        clean = (text or "").strip()
        if not clean:
            return fallback_text

        generic_patterns = (
            r"抱歉[，,]?(我)?(还|暂时).{0,8}(没有学会|不会|无法|不能).{0,8}(回答|处理)",
            r"如果你有其他问题.{0,12}(乐意|愿意).{0,8}(帮助|服务)",
            r"请换(个|一个)?问题",
            r"请告诉我(您|你)?的?需求",
            r"想去哪里(旅行|旅游)",
            r"旅行时间(多长|多久)",
            r"偏好什么类型的体验",
        )
        normalized = ChatOrchestrator._normalize_for_match(clean)
        if any(re.search(pattern, normalized) for pattern in generic_patterns):
            return fallback_text

        tool_name = str(tool_result.get("tool_name") or "")
        lowered = clean.lower()

        if tool_name == PlannerToolName.ROUTE_PLAN.value:
            if not ChatOrchestrator._is_route_reply_grounded(clean, tool_result):
                return fallback_text

        if tool_name == PlannerToolName.WEATHER_NOW.value:
            location = str(tool_result.get("location") or "")
            if location and location not in clean and "天气" not in clean and "气温" not in clean and "温度" not in clean:
                return fallback_text

        if tool_name == PlannerToolName.SEARCH_WEB.value:
            query = str(tool_result.get("query") or "")
            key_tokens = [token for token in re.split(r"\s+", query) if token][:3]
            has_token = any(token and token in clean for token in key_tokens)
            if not has_token and "检索" not in clean and "搜索" not in clean and "查询" not in clean:
                return fallback_text

        if tool_name == PlannerToolName.NEARBY_SEARCH.value:
            places = tool_result.get("places") if isinstance(tool_result.get("places"), list) else []
            if places:
                names = [str(item.get("name") or "").strip() for item in places if isinstance(item, dict)]
                if names and not any(name and name in clean for name in names[:3]):
                    if not any(token in clean for token in ("附近", "米", "公里", "家")):
                        return fallback_text
            else:
                if not any(token in clean for token in ("没找到", "未找到", "暂无", "没有")):
                    return fallback_text

        if any(sym in lowered for sym in ("🌍", "✈️", "💫", "✨")) and len(clean) > 120:
            return fallback_text
        return clean
