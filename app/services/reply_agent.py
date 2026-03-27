from __future__ import annotations

import json
import re
from typing import Any

from app.schemas import PlannerOutput


class ReplyAgent:
    # Keep persona as a short, stable service-side card to reduce token usage.
    SYSTEM_PROMPT = (
        "你是“真由理”，一个可爱元气但非常靠谱的旅行助手。"
        "默认用简洁自然中文，先给结论，再给必要细节。"
        "语气轻快温暖不过度卖萌；口头禅“嘟嘟噜”自然出现一次。"
        "用户着急时自动更简洁。"
        "优先基于 tool_result 作答，禁止编造。"
        "时间、距离、票价、温度等数值只能来自 tool_result；没有就明确说没有。"
        "当 scenario=no_tool_direct_chat 时，可直接基于 user_text 与 memory_hints 对话回答。"
        "当 tool_name=memory.update 时，用确认语气回复变更结果，不要扩写无关内容。"
        "当 scenario=weather_over_range 时，必须提醒“超过7天，天气数据来自于网络，不一定准确”。"
        "若 weather_over_range 且 focus=temperature：temperature_estimate 为空则明确无法给出可靠具体温度；"
        "temperature_estimate 不为空则给出区间并标注“仅供参考”。"
        "输出自然对话文本，不用代码块，不暴露内部字段名。"
    )

    def __init__(self, llm_provider) -> None:
        self._llm_provider = llm_provider

    async def compose(
        self,
        *,
        user_text: str,
        plan: PlannerOutput,
        tool_result: dict[str, Any],
        fallback_text: str,
    ) -> str:
        payload = {
            "user_text": user_text,
            "plan": plan.model_dump(mode="json"),
            "tool_result": tool_result,
        }
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        try:
            raw = await self._llm_provider.chat(messages)
        except Exception:
            return fallback_text

        normalized = self._normalize_text(raw)
        return normalized or fallback_text

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()
