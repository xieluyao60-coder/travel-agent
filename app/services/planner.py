from __future__ import annotations

import asyncio
import json
import re
from datetime import date, timedelta
from typing import Any

from app.schemas import IntentType, PlannerAction, PlannerOutput, PlannerToolName


class QueryPlanner:
    LLM_TIMEOUT_SECONDS = 1.5
    SYSTEM_PROMPT = (
        "You are the parser agent for a Chinese travel assistant.\n"
        "You do NOT answer the user directly. You only output one JSON object.\n"
        "Your job: choose action/intent/tool and extract tool parameters.\n\n"
        "Tool catalog:\n"
        "1) weather.now(location, when, target_date?): weather query (realtime/tomorrow/date/week).\n"
        "2) route.plan(origin, destination, mode?): route planning.\n"
        "3) search.search(query, top_k<=5): web search for攻略/新闻等开放信息.\n\n"
        "4) nearby.search(location, keyword, radius_m<=5000): nearby POI search around a location.\n\n"
        "5) memory.update(operation, travel_city?, hotel_location?): update user memory.\n\n"
        "Output schema:\n"
        '{'
        '"action":"call_tool|clarify|chat",'
        '"intent":"weather|route|search|nearby|memory|chat",'
        '"tool_name":"weather.now|route.plan|search.search|nearby.search|memory.update|null",'
        '"normalized_query":"...",'
        '"confidence":0.0,'
        '"weather":{"location":null,"when":null,"target_date":null},'
        '"route":{"origin":null,"destination":null,"mode":null,"goal":null},'
        '"search":{"query":null,"top_k":5},'
        '"nearby":{"location":null,"keyword":null,"radius_m":1000},'
        '"memory":{"operation":null,"travel_city":null,"hotel_location":null},'
        '"missing_slots":[],'
        '"clarification_question":null'
        '}\n\n'
        "Rules:\n"
        "- For weather date queries (tomorrow/specific date), still use weather.now tool with when/target_date.\n"
        "- Route extraction should tolerate patterns like 从A到B / A到B / 从A道B / 从A去B.\n"
        "- Nearby extraction should tolerate forms like 酒店附近有麻将馆吗 / 温州站周边咖啡店 / 附近有便利店吗.\n"
        "- Memory update is only for explicit commands like 我在X旅行 / 我的酒店在X / 把X记为酒店 / 忘记酒店 / 重置记忆.\n"
        "- Questions like 我的酒店在哪 are chat, not memory.update.\n"
        "- If required slots are missing, return clarify with missing_slots and clarification_question.\n"
        "- Return JSON only."
    )

    _route_patterns = (
        re.compile(
            r"(?:从|由)\s*(?P<origin>.+?)\s*(?:到|道|至)\s*(?P<destination>.+?)"
            r"(?:怎么走|怎么去|怎么坐|怎么坐车|如何|最快|最短|多久|多长时间|多少时间|路程|路线|通勤|要多久|需要多久|[?？]|$)"
        ),
        re.compile(
            r"(?P<origin>.+?)\s*(?:到|道|至)\s*(?P<destination>.+?)"
            r"(?:怎么走|怎么去|怎么坐|怎么坐车|如何|最快|最短|多久|多长时间|多少时间|路程|路线|通勤|要多久|需要多久|[?？]|$)"
        ),
        re.compile(
            r"(?:从|由)\s*(?P<origin>.+?)\s*(?:去|前往|往)\s*(?P<destination>.+?)"
            r"(?:怎么走|怎么去|怎么坐|怎么坐车|如何|最快|最短|多久|多长时间|多少时间|路程|路线|通勤|要多久|需要多久|[?？]|$)"
        ),
    )
    _route_cues = (
        "怎么走",
        "怎么去",
        "怎么坐",
        "怎么坐车",
        "如何",
        "路线",
        "路程",
        "通勤",
        "最快",
        "最短",
        "多久",
        "多长时间",
        "多少时间",
    )
    _weather_keywords = ("天气", "气温", "温度", "最高温", "最低温", "降雨", "下雨", "体感")
    _weather_realtime_cues = ("现在", "当前", "实时", "此刻", "此时", "今天")
    _weather_tomorrow_cues = ("明天", "明日")
    _weather_after_tomorrow_cues = ("后天",)
    _weather_week_cues = ("本周", "这周", "下周", "周末")
    _date_full_pattern = re.compile(r"(?P<year>\d{4})[./\-年](?P<month>\d{1,2})[./\-月](?P<day>\d{1,2})(?:日|号)?")
    _date_md_pattern = re.compile(r"(?P<month>\d{1,2})[./月](?P<day>\d{1,2})(?:日|号)?")
    _weather_location_pattern = re.compile(r"(?P<location>[\u4e00-\u9fa5A-Za-z0-9\-./]{2,32}?)(?:的)?(?:天气|气温|温度)")
    _search_keywords = ("搜索", "搜", "查一下", "查询", "攻略", "推荐", "景点", "新闻")
    _nearby_cues = ("附近", "周边", "周围")
    _nearby_prefix = re.compile(r"^(?:在)?(?P<location>.+?)(?:附近|周边|周围)(?:有|有没有|有无|有什么)?(?P<keyword>.+)$")
    _nearby_suffix = re.compile(r"^(?:附近|周边|周围)(?:有|有没有|有无|有什么)?(?P<keyword>.+)$")
    _nearby_radius_km_pattern = re.compile(r"(?P<km>\d+(?:\.\d+)?)\s*公里")
    _nearby_radius_m_pattern = re.compile(r"(?P<m>\d{2,5})\s*米")
    _nearby_expand_cues = ("扩大", "扩一下", "放大", "更大", "远一点", "搜索范围", "扩大范围")
    _nearby_shrink_cues = ("缩小", "小一点", "近一点", "缩范围")
    _nearby_followup_cues = ("再找", "再搜", "再看看", "继续找", "继续搜", "还有吗")
    _nearby_range_cues = ("范围", "半径")
    _search_prefix_pattern = re.compile(r"^(?:帮我|请)?(?:搜索|搜|查一下|查询)\s*")
    _time_prefix_pattern = re.compile(r"^(?:今天|明天|后天|今晚|今早|本周|这周|下周|周末|明日|昨日)")
    _time_inline_pattern = re.compile(r"(?:今天|明天|后天|今晚|今早|本周|这周|下周|周末|明日|昨日)")
    _trailing_particle_pattern = re.compile(r"(?:的|呢|啊|呀|吗)$")
    _question_mark_pattern = re.compile(r"[?？]|(在哪|哪里|多少|几[点个]|怎么|为何|为什么|是否)")

    def __init__(self, llm_provider) -> None:
        self._llm_provider = llm_provider

    async def plan(
        self,
        text: str,
        history: list[dict[str, str]] | None = None,
        pending: dict[str, Any] | None = None,
        memory_hints: dict[str, Any] | None = None,
    ) -> PlannerOutput:
        content = text.strip()
        pending_state = pending or {}
        memory_state = memory_hints or {}

        fast_plan = self._fallback_plan(content, pending_state)
        if fast_plan.action in {PlannerAction.CALL_TOOL, PlannerAction.CLARIFY} and (
            not pending_state or fast_plan.intent == IntentType.NEARBY
        ):
            return self._post_process(plan=fast_plan, text=content, pending=pending_state)

        payload = {
            "user_text": content,
            "history": (history or [])[-8:],
            "pending": pending_state,
            "memory_hints": memory_state,
        }
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        try:
            raw = await asyncio.wait_for(self._llm_provider.chat(messages), timeout=self.LLM_TIMEOUT_SECONDS)
            parsed = self._parse_json_object(raw)
            plan = PlannerOutput.model_validate(parsed)
        except Exception:
            plan = self._fallback_plan(content, pending_state)

        return self._post_process(plan=plan, text=content, pending=pending_state)

    def _post_process(self, plan: PlannerOutput, text: str, pending: dict[str, Any]) -> PlannerOutput:
        normalized_query = (plan.normalized_query or "").strip()
        plan.normalized_query = normalized_query or text
        plan.confidence = max(0.0, min(1.0, float(plan.confidence or 0.0)))
        plan.search.top_k = max(1, min(int(plan.search.top_k or 5), 5))
        if plan.nearby.radius_m is not None:
            plan.nearby.radius_m = max(100, min(int(plan.nearby.radius_m), 5000))

        plan = self._merge_pending(plan, pending)

        if plan.action == PlannerAction.CALL_TOOL:
            self._normalize_tool_defaults(plan, text)
            missing_slots = self._infer_missing_slots(plan.intent, plan)
            if missing_slots:
                plan.action = PlannerAction.CLARIFY
                plan.missing_slots = missing_slots
                plan.clarification_question = self._default_clarify_question(plan.intent, missing_slots)

        if plan.action == PlannerAction.CLARIFY:
            if plan.intent == IntentType.CHAT:
                plan.intent = self._pending_intent(pending) or IntentType.CHAT
            if plan.intent != IntentType.CHAT and plan.tool_name is None:
                plan.tool_name = self._default_tool_for_intent(plan.intent)
            if not plan.missing_slots:
                plan.missing_slots = self._infer_missing_slots(plan.intent, plan)
            if not (plan.clarification_question or "").strip():
                plan.clarification_question = self._default_clarify_question(plan.intent, plan.missing_slots)

        if plan.action == PlannerAction.CHAT:
            plan.intent = IntentType.CHAT
            plan.tool_name = None
            plan.missing_slots = []
            plan.clarification_question = None

        return plan

    def _normalize_tool_defaults(self, plan: PlannerOutput, text: str) -> None:
        if plan.intent == IntentType.ROUTE:
            plan.tool_name = PlannerToolName.ROUTE_PLAN
            if not plan.route.mode:
                plan.route.mode = self._infer_route_mode(text)
            if not plan.route.goal:
                plan.route.goal = self._infer_route_goal(text)
            return

        if plan.intent == IntentType.WEATHER:
            plan.tool_name = PlannerToolName.WEATHER_NOW
            location = (plan.weather.location or "").strip()
            if location:
                plan.weather.location = self._normalize_weather_location(location)
            if not plan.weather.when:
                when, target_date = self._infer_weather_when_and_date(text)
                plan.weather.when = when
                plan.weather.target_date = target_date
            return

        if plan.intent == IntentType.SEARCH:
            plan.tool_name = PlannerToolName.SEARCH_WEB
            if not (plan.search.query or "").strip():
                plan.search.query = plan.normalized_query or text
            return

        if plan.intent == IntentType.NEARBY:
            plan.tool_name = PlannerToolName.NEARBY_SEARCH
            location = (plan.nearby.location or "").strip()
            keyword = (plan.nearby.keyword or "").strip()
            if location:
                plan.nearby.location = self._clean_place_text(location)
            if keyword:
                plan.nearby.keyword = self._normalize_nearby_keyword(keyword)
            if plan.nearby.radius_m is None:
                plan.nearby.radius_m = self._infer_nearby_radius(text)
            return

        if plan.intent == IntentType.MEMORY:
            plan.tool_name = PlannerToolName.MEMORY_UPDATE
            if plan.memory.travel_city:
                plan.memory.travel_city = self._normalize_memory_city(plan.memory.travel_city)
            if plan.memory.hotel_location:
                plan.memory.hotel_location = self._normalize_memory_place(plan.memory.hotel_location)
            return

        plan.tool_name = None

    def _merge_pending(self, plan: PlannerOutput, pending: dict[str, Any]) -> PlannerOutput:
        pending_intent = self._pending_intent(pending)
        if pending_intent is None:
            return plan

        known_slots = pending.get("known_slots")
        if not isinstance(known_slots, dict):
            return plan

        if plan.intent == IntentType.CHAT and plan.action in {PlannerAction.CALL_TOOL, PlannerAction.CLARIFY}:
            plan.intent = pending_intent

        if plan.intent != pending_intent:
            return plan

        if plan.intent == IntentType.ROUTE:
            if not plan.route.origin and isinstance(known_slots.get("origin"), str):
                plan.route.origin = known_slots["origin"].strip()
            if not plan.route.destination and isinstance(known_slots.get("destination"), str):
                plan.route.destination = known_slots["destination"].strip()
            if not plan.route.mode and isinstance(known_slots.get("mode"), str):
                mode = known_slots["mode"].strip().lower()
                if mode in {"transit", "driving", "walking"}:
                    plan.route.mode = mode  # type: ignore[assignment]
            if not plan.route.goal and isinstance(known_slots.get("goal"), str):
                goal = known_slots["goal"].strip().lower()
                if goal in {"fastest", "cheapest", "least_walking", "balanced"}:
                    plan.route.goal = goal  # type: ignore[assignment]
        elif plan.intent == IntentType.WEATHER:
            if not plan.weather.location and isinstance(known_slots.get("location"), str):
                plan.weather.location = known_slots["location"].strip()
            if not plan.weather.target_date and isinstance(known_slots.get("target_date"), str):
                plan.weather.target_date = known_slots["target_date"].strip()
        elif plan.intent == IntentType.SEARCH:
            if not plan.search.query and isinstance(known_slots.get("query"), str):
                plan.search.query = known_slots["query"].strip()
        elif plan.intent == IntentType.NEARBY:
            if not plan.nearby.location and isinstance(known_slots.get("location"), str):
                plan.nearby.location = known_slots["location"].strip()
            if not plan.nearby.keyword and isinstance(known_slots.get("keyword"), str):
                plan.nearby.keyword = known_slots["keyword"].strip()
        elif plan.intent == IntentType.MEMORY:
            if not plan.memory.travel_city and isinstance(known_slots.get("travel_city"), str):
                plan.memory.travel_city = known_slots["travel_city"].strip()
            if not plan.memory.hotel_location and isinstance(known_slots.get("hotel_location"), str):
                plan.memory.hotel_location = known_slots["hotel_location"].strip()

        if plan.tool_name is None and plan.intent != IntentType.CHAT:
            plan.tool_name = self._default_tool_for_intent(plan.intent)

        return plan

    def _fallback_plan(self, text: str, pending: dict[str, Any]) -> PlannerOutput:
        content = text.strip()

        memory_plan = self._extract_memory_plan(content)
        if memory_plan is not None:
            return memory_plan

        nearby_followup_plan = self._extract_nearby_followup_plan(content, pending)
        if nearby_followup_plan is not None:
            return nearby_followup_plan

        route_slots = self._extract_route_slots(content)
        if route_slots:
            origin, destination = route_slots
            return PlannerOutput(
                action=PlannerAction.CALL_TOOL,
                intent=IntentType.ROUTE,
                tool_name=PlannerToolName.ROUTE_PLAN,
                normalized_query=content,
                confidence=0.68,
                route={
                    "origin": origin,
                    "destination": destination,
                    "mode": self._infer_route_mode(content),
                    "goal": self._infer_route_goal(content),
                },
            )

        if self._looks_like_nearby_query(content):
            location, keyword, radius_m = self._extract_nearby_slots(content)
            return PlannerOutput(
                action=PlannerAction.CALL_TOOL if keyword else PlannerAction.CLARIFY,
                intent=IntentType.NEARBY,
                tool_name=PlannerToolName.NEARBY_SEARCH,
                normalized_query=content,
                confidence=0.64 if keyword else 0.5,
                nearby={
                    "location": location,
                    "keyword": keyword,
                    "radius_m": radius_m,
                },
                missing_slots=[] if keyword else ["keyword"],
                clarification_question=(None if keyword else self._default_clarify_question(IntentType.NEARBY, ["keyword"])),
            )

        if self._looks_like_weather_query(content):
            location = self._extract_weather_location(content)
            when, target_date = self._infer_weather_when_and_date(content)
            return PlannerOutput(
                action=PlannerAction.CALL_TOOL if location else PlannerAction.CLARIFY,
                intent=IntentType.WEATHER,
                tool_name=PlannerToolName.WEATHER_NOW,
                normalized_query=content,
                confidence=0.62 if location else 0.5,
                weather={
                    "location": location,
                    "when": when,
                    "target_date": target_date,
                },
                missing_slots=[] if location else ["location"],
                clarification_question=(None if location else self._default_clarify_question(IntentType.WEATHER, ["location"])),
            )

        if self._looks_like_search_query(content):
            query = self._search_prefix_pattern.sub("", content).strip() or content
            return PlannerOutput(
                action=PlannerAction.CALL_TOOL,
                intent=IntentType.SEARCH,
                tool_name=PlannerToolName.SEARCH_WEB,
                normalized_query=content,
                confidence=0.58,
                search={"query": query, "top_k": 5},
            )

        pending_intent = self._pending_intent(pending)
        if pending_intent in {IntentType.ROUTE, IntentType.WEATHER, IntentType.SEARCH, IntentType.NEARBY}:
            missing_slots = list(pending.get("missing_slots") or [])
            if not missing_slots:
                return PlannerOutput(
                    action=PlannerAction.CHAT,
                    intent=IntentType.CHAT,
                    tool_name=None,
                    normalized_query=content,
                    confidence=0.45,
                )
            return PlannerOutput(
                action=PlannerAction.CLARIFY,
                intent=pending_intent,
                tool_name=self._default_tool_for_intent(pending_intent),
                normalized_query=content,
                confidence=0.42,
                missing_slots=missing_slots,
                clarification_question=self._default_clarify_question(pending_intent, missing_slots),
            )

        return PlannerOutput(
            action=PlannerAction.CHAT,
            intent=IntentType.CHAT,
            tool_name=None,
            normalized_query=content,
            confidence=0.45,
        )

    def _extract_nearby_followup_plan(self, text: str, pending: dict[str, Any]) -> PlannerOutput | None:
        if self._pending_intent(pending) != IntentType.NEARBY:
            return None
        if not self._looks_like_nearby_followup(text):
            return None

        known_slots = pending.get("known_slots")
        if not isinstance(known_slots, dict):
            return None

        raw_location = known_slots.get("location")
        raw_keyword = known_slots.get("keyword")
        location = raw_location.strip() if isinstance(raw_location, str) else ""
        keyword = raw_keyword.strip() if isinstance(raw_keyword, str) else ""
        if not location or not keyword:
            return None

        radius_m = self._derive_nearby_followup_radius(text=text, known_slots=known_slots)
        return PlannerOutput(
            action=PlannerAction.CALL_TOOL,
            intent=IntentType.NEARBY,
            tool_name=PlannerToolName.NEARBY_SEARCH,
            normalized_query=text,
            confidence=0.7,
            nearby={
                "location": location,
                "keyword": keyword,
                "radius_m": radius_m,
            },
        )

    def _extract_memory_plan(self, text: str) -> PlannerOutput | None:
        content = (text or "").strip()
        if not content:
            return None
        if self._question_mark_pattern.search(content):
            return None

        if re.fullmatch(r"(?:忘记酒店|清除酒店|删除酒店)\s*", content):
            return PlannerOutput(
                action=PlannerAction.CALL_TOOL,
                intent=IntentType.MEMORY,
                tool_name=PlannerToolName.MEMORY_UPDATE,
                normalized_query=content,
                confidence=0.92,
                memory={"operation": "clear_hotel"},
            )

        if re.fullmatch(r"(?:重置记忆|清空记忆|忘记我的偏好|忘记我偏好)\s*", content):
            return PlannerOutput(
                action=PlannerAction.CALL_TOOL,
                intent=IntentType.MEMORY,
                tool_name=PlannerToolName.MEMORY_UPDATE,
                normalized_query=content,
                confidence=0.92,
                memory={"operation": "reset_profile"},
            )

        hotel_direct = re.search(r"把(?P<place>[^，。；;！？?]+?)记为酒店", content)
        if hotel_direct:
            place = self._normalize_memory_place(hotel_direct.group("place"))
            city = self._extract_city_for_memory(content)
            return PlannerOutput(
                action=PlannerAction.CALL_TOOL if place else PlannerAction.CLARIFY,
                intent=IntentType.MEMORY,
                tool_name=PlannerToolName.MEMORY_UPDATE,
                normalized_query=content,
                confidence=0.9 if place else 0.55,
                memory={"operation": "set_hotel", "hotel_location": place, "travel_city": city},
                missing_slots=[] if place else ["hotel_location"],
                clarification_question=(
                    None if place else "酒店位置我还没听清，可以再说得具体一点吗？"
                ),
            )

        hotel_match = re.search(r"(?:我的酒店在|我住在)\s*(?P<place>[^，。；;！？?]+)", content)
        if hotel_match:
            place = self._normalize_memory_place(hotel_match.group("place"))
            city = self._extract_city_for_memory(content)
            return PlannerOutput(
                action=PlannerAction.CALL_TOOL if place else PlannerAction.CLARIFY,
                intent=IntentType.MEMORY,
                tool_name=PlannerToolName.MEMORY_UPDATE,
                normalized_query=content,
                confidence=0.88 if place else 0.55,
                memory={"operation": "set_hotel", "hotel_location": place, "travel_city": city},
                missing_slots=[] if place else ["hotel_location"],
                clarification_question=(
                    None if place else "酒店位置我还没听清，可以再说得具体一点吗？"
                ),
            )

        city_match = re.search(
            r"(?:^|[，,。；;！？!])\s*我(?:现在)?在(?P<city>[^\s，,。；;！？?]{2,12}?)(?:旅行|旅游|出差|游玩|玩)(?:$|[，,。；;！？!])",
            content,
        )
        if city_match:
            city = self._normalize_memory_city(city_match.group("city"))
            return PlannerOutput(
                action=PlannerAction.CALL_TOOL if city else PlannerAction.CLARIFY,
                intent=IntentType.MEMORY,
                tool_name=PlannerToolName.MEMORY_UPDATE,
                normalized_query=content,
                confidence=0.9 if city else 0.55,
                memory={"operation": "set_city", "travel_city": city},
                missing_slots=[] if city else ["travel_city"],
                clarification_question=(None if city else "你当前所在城市我还没听清，可以说成“我在温州旅行”。"),
            )
        return None

    def _extract_city_for_memory(self, content: str) -> str | None:
        city_match = re.search(
            r"(?:^|[，,。；;！？!])\s*我(?:现在)?在(?P<city>[^\s，,。；;！？?]{2,12}?)(?:旅行|旅游|出差|游玩|玩)(?:$|[，,。；;！？!])",
            content,
        )
        if not city_match:
            return None
        return self._normalize_memory_city(city_match.group("city"))

    def _extract_route_slots(self, text: str) -> tuple[str, str] | None:
        if not text:
            return None

        candidate = text.strip()
        route_like = any(cue in candidate for cue in self._route_cues) or any(token in candidate for token in ("到", "道", "至"))
        if not route_like:
            return None

        for pattern in self._route_patterns:
            match = pattern.search(candidate)
            if not match:
                continue
            origin = self._clean_place_text(match.group("origin"))
            destination = self._clean_place_text(match.group("destination"))
            if origin and destination and origin != destination:
                return origin, destination

        return None

    @staticmethod
    def _clean_place_text(value: str) -> str | None:
        cleaned = re.sub(r"[，,。！？?；;]", " ", value).strip()
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = re.sub(r"^(请问|请|帮我|我想|想问|咨询)", "", cleaned)
        cleaned = re.sub(r"(怎么走|怎么去|怎么坐|怎么坐车|如何|最快|最短|多久|多长时间|多少时间|路程|路线|通勤|要多久|需要多久)$", "", cleaned)
        cleaned = cleaned.strip()
        return cleaned or None

    def _extract_weather_location(self, text: str) -> str | None:
        match = self._weather_location_pattern.search(text)
        if not match:
            return None
        return self._normalize_weather_location(match.group("location"))

    def _normalize_weather_location(self, location: str) -> str | None:
        normalized = location.strip()
        normalized = self._time_prefix_pattern.sub("", normalized).strip()
        normalized = self._time_inline_pattern.sub("", normalized).strip()
        normalized = self._date_full_pattern.sub("", normalized).strip()
        normalized = self._date_md_pattern.sub("", normalized).strip()
        normalized = self._trailing_particle_pattern.sub("", normalized).strip()
        normalized = re.sub(r"\s+", "", normalized)
        return normalized or None

    def _infer_weather_when_and_date(self, text: str) -> tuple[str, str | None]:
        today = date.today()
        if any(token in text for token in self._weather_tomorrow_cues):
            target = today + timedelta(days=1)
            return "tomorrow", target.isoformat()
        if any(token in text for token in self._weather_after_tomorrow_cues):
            target = today + timedelta(days=2)
            return "date", target.isoformat()

        full = self._date_full_pattern.search(text)
        if full:
            try:
                target = date(int(full.group("year")), int(full.group("month")), int(full.group("day")))
                return "date", target.isoformat()
            except ValueError:
                pass

        md = self._date_md_pattern.search(text)
        if md:
            month = int(md.group("month"))
            day = int(md.group("day"))
            target = self._resolve_month_day(today, month, day)
            if target is not None:
                return "date", target.isoformat()

        if any(token in text for token in self._weather_week_cues):
            return "week", None
        if any(token in text for token in self._weather_realtime_cues):
            return "realtime", None
        return "realtime", None

    @staticmethod
    def _resolve_month_day(today: date, month: int, day: int) -> date | None:
        try:
            candidate = date(today.year, month, day)
        except ValueError:
            return None

        if candidate < today:
            try:
                next_year = date(today.year + 1, month, day)
                return next_year
            except ValueError:
                return candidate
        return candidate

    @staticmethod
    def _infer_route_mode(text: str) -> str | None:
        lowered = text.lower()
        if any(token in lowered for token in ("驾车", "开车", "打车", "自驾", "car")):
            return "driving"
        if any(token in lowered for token in ("步行", "走路", "walk", "徒步")):
            return "walking"
        if any(token in lowered for token in ("公交", "地铁", "bus", "subway", "公共交通", "坐车")):
            return "transit"
        return None

    @staticmethod
    def _infer_route_goal(text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("最快", "最短", "多久", "多长时间", "多少时间", "几分钟", "要多久", "需要多久")):
            return "fastest"
        if any(token in lowered for token in ("最便宜", "省钱", "便宜")):
            return "cheapest"
        if any(token in lowered for token in ("少走路", "步行少", "少步行")):
            return "least_walking"
        return "balanced"

    def _looks_like_weather_query(self, text: str) -> bool:
        return any(keyword in text for keyword in self._weather_keywords)

    def _looks_like_search_query(self, text: str) -> bool:
        return any(keyword in text for keyword in self._search_keywords)

    def _looks_like_nearby_query(self, text: str) -> bool:
        if not text:
            return False
        if any(cue in text for cue in self._nearby_cues):
            return True
        return False

    def _extract_nearby_slots(self, text: str) -> tuple[str | None, str | None, int]:
        content = text.strip()
        radius_m = self._infer_nearby_radius(content)
        cleaned = self._nearby_radius_km_pattern.sub("", content)
        cleaned = self._nearby_radius_m_pattern.sub("", cleaned)
        cleaned = re.sub(r"[？?。！!]+$", "", cleaned).strip()

        match = self._nearby_prefix.match(cleaned)
        if match:
            location = self._clean_place_text(match.group("location") or "")
            keyword = self._normalize_nearby_keyword(match.group("keyword") or "")
            return location, keyword, radius_m

        suffix = self._nearby_suffix.match(cleaned)
        if suffix:
            keyword = self._normalize_nearby_keyword(suffix.group("keyword") or "")
            return None, keyword, radius_m

        if "附近" in cleaned:
            left, _, right = cleaned.partition("附近")
            location = self._clean_place_text(left)
            keyword = self._normalize_nearby_keyword(right)
            return location, keyword, radius_m
        if "周边" in cleaned:
            left, _, right = cleaned.partition("周边")
            location = self._clean_place_text(left)
            keyword = self._normalize_nearby_keyword(right)
            return location, keyword, radius_m
        if "周围" in cleaned:
            left, _, right = cleaned.partition("周围")
            location = self._clean_place_text(left)
            keyword = self._normalize_nearby_keyword(right)
            return location, keyword, radius_m
        return None, self._normalize_nearby_keyword(cleaned), radius_m

    @staticmethod
    def _normalize_nearby_keyword(raw: str) -> str | None:
        keyword = (raw or "").strip()
        keyword = re.sub(r"^(有|有没有|有无|有什么|是否有|能不能找到)", "", keyword)
        keyword = re.sub(r"^(?:内|里|范围内)", "", keyword)
        keyword = re.sub(r"(吗|么|嘛|呢|啊|呀)$", "", keyword)
        keyword = re.sub(r"[，,。！？?；;]+$", "", keyword)
        keyword = keyword.strip()
        if keyword in {"", "什么", "啥", "哪些", "哪里", "店", "地方"}:
            return None
        return keyword

    def _infer_nearby_radius(self, text: str) -> int:
        km_match = self._nearby_radius_km_pattern.search(text)
        if km_match:
            km = float(km_match.group("km"))
            return max(100, min(int(km * 1000), 5000))
        m_match = self._nearby_radius_m_pattern.search(text)
        if m_match:
            meters = int(m_match.group("m"))
            return max(100, min(meters, 5000))
        return 1000

    def _looks_like_nearby_followup(self, text: str) -> bool:
        content = (text or "").strip()
        if not content:
            return False
        if any(cue in content for cue in self._nearby_expand_cues):
            return True
        if any(cue in content for cue in self._nearby_shrink_cues):
            return True
        if any(cue in content for cue in self._nearby_followup_cues) and any(
            range_cue in content for range_cue in self._nearby_range_cues
        ):
            return True
        has_radius = bool(self._nearby_radius_km_pattern.search(content) or self._nearby_radius_m_pattern.search(content))
        if has_radius and any(cue in content for cue in (*self._nearby_range_cues, *self._nearby_expand_cues, *self._nearby_shrink_cues)):
            return True
        return False

    def _derive_nearby_followup_radius(self, *, text: str, known_slots: dict[str, Any]) -> int:
        current_radius = 1000
        raw_radius = known_slots.get("radius_m")
        if isinstance(raw_radius, str) and raw_radius.isdigit():
            current_radius = int(raw_radius)
        elif isinstance(raw_radius, (int, float)):
            current_radius = int(raw_radius)
        current_radius = max(100, min(current_radius, 5000))

        has_explicit_radius = bool(self._nearby_radius_km_pattern.search(text) or self._nearby_radius_m_pattern.search(text))
        if has_explicit_radius:
            explicit_radius = self._infer_nearby_radius(text)
            if any(cue in text for cue in self._nearby_expand_cues):
                return max(current_radius, explicit_radius)
            return explicit_radius

        if any(cue in text for cue in self._nearby_shrink_cues):
            return max(100, current_radius // 2)

        if any(cue in text for cue in (*self._nearby_expand_cues, *self._nearby_range_cues, *self._nearby_followup_cues)):
            return min(current_radius * 2, 5000)

        return current_radius

    @staticmethod
    def _pending_intent(pending: dict[str, Any]) -> IntentType | None:
        raw = pending.get("intent")
        if not isinstance(raw, str):
            return None
        try:
            return IntentType(raw)
        except ValueError:
            return None

    @staticmethod
    def _default_tool_for_intent(intent: IntentType) -> PlannerToolName | None:
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
    def _infer_missing_slots(intent: IntentType, plan: PlannerOutput) -> list[str]:
        if intent == IntentType.ROUTE:
            missing: list[str] = []
            if not (plan.route.origin or "").strip():
                missing.append("origin")
            if not (plan.route.destination or "").strip():
                missing.append("destination")
            return missing
        if intent == IntentType.WEATHER:
            return ["location"] if not (plan.weather.location or "").strip() else []
        if intent == IntentType.SEARCH:
            return ["query"] if not (plan.search.query or "").strip() else []
        if intent == IntentType.NEARBY:
            missing: list[str] = []
            if not (plan.nearby.location or "").strip():
                missing.append("location")
            if not (plan.nearby.keyword or "").strip():
                missing.append("keyword")
            return missing
        if intent == IntentType.MEMORY:
            operation = (plan.memory.operation or "").strip()
            if operation == "set_city":
                return ["travel_city"] if not (plan.memory.travel_city or "").strip() else []
            if operation == "set_hotel":
                return ["hotel_location"] if not (plan.memory.hotel_location or "").strip() else []
            return []
        return []

    @staticmethod
    def _default_clarify_question(intent: IntentType, missing_slots: list[str]) -> str:
        if intent == IntentType.ROUTE:
            if set(missing_slots) == {"origin", "destination"}:
                return "请告诉我起点和终点，例如“从上海虹桥站到同济大学嘉定校区怎么走”。"
            if "origin" in missing_slots:
                return "请补充起点位置。"
            if "destination" in missing_slots:
                return "请补充终点位置。"
        if intent == IntentType.WEATHER:
            return "请告诉我要查询天气的城市，例如“上海天气”。"
        if intent == IntentType.SEARCH:
            return "请补充你想搜索的主题，例如“杭州三日游攻略”。"
        if intent == IntentType.NEARBY:
            if set(missing_slots) == {"location", "keyword"}:
                return "请告诉我中心地点和要找的类型，例如“温州站附近有咖啡店吗”。"
            if "location" in missing_slots:
                return "请告诉我要在哪个地点附近查，比如“温州站附近”。"
            if "keyword" in missing_slots:
                return "请补充你要找的类型，例如“咖啡店”“便利店”。"
        if intent == IntentType.MEMORY:
            if "travel_city" in missing_slots:
                return "你当前所在城市我还没听清，可以说成“我在温州旅行”。"
            if "hotel_location" in missing_slots:
                return "酒店位置我还没听清，可以再说得具体一点吗？"
        return "请再具体一点，我好继续为你处理。"

    @staticmethod
    def _normalize_memory_place(value: str) -> str | None:
        normalized = re.sub(r"[，。！？,.!?；;：:\s]+", "", (value or "").strip())
        normalized = re.sub(r"(这边|这里|那边|那里)$", "", normalized)
        normalized = normalized.strip()
        if not normalized:
            return None
        if len(normalized) <= 1:
            return None
        if normalized in {"哪", "哪里", "在哪"}:
            return None
        if re.fullmatch(r"\d{3,}", normalized):
            return None
        return normalized

    @staticmethod
    def _normalize_memory_city(value: str) -> str | None:
        normalized = QueryPlanner._normalize_memory_place(value)
        if not normalized:
            return None
        if len(normalized) > 12:
            return None
        if re.search(r"(酒店|宾馆|旅馆|车站|机场|地铁|公交|大道|路|街|景区|公园|校区)", normalized):
            return None
        return normalized

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        payload = text.strip()
        if payload.startswith("```"):
            payload = re.sub(r"^```[a-zA-Z]*\s*", "", payload)
            payload = re.sub(r"\s*```$", "", payload)

        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass

        start = payload.find("{")
        end = payload.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("planner output missing json object")

        candidate = payload[start : end + 1]
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("planner output is not a json object")
        return parsed
