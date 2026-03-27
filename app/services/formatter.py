from __future__ import annotations

from app.schemas import NearbyPlace, RoutePlan, SearchResult, WeatherForecastResult, WeatherResult


def format_weather_reply(weather: WeatherResult) -> str:
    suggestion = _weather_suggestion(weather.temperature_c, weather.condition)
    feels_like = f"{weather.feels_like_c:.0f}°C" if weather.feels_like_c is not None else "未知"
    humidity = f"{weather.humidity}%" if weather.humidity is not None else "未知"

    return (
        f"{weather.location}现在{weather.condition}，气温 {weather.temperature_c:.0f}°C，体感 {feels_like}，"
        f"湿度 {humidity}，风向 {weather.wind_direction or '未知'}。{suggestion}"
    )


def format_weather_forecast_reply(forecast: WeatherForecastResult) -> str:
    temp_min = f"{forecast.temp_min_c:.0f}°C" if forecast.temp_min_c is not None else "未知"
    temp_max = f"{forecast.temp_max_c:.0f}°C" if forecast.temp_max_c is not None else "未知"
    humidity = f"{forecast.humidity}%" if forecast.humidity is not None else "未知"
    night = f"，夜间{forecast.condition_night}" if forecast.condition_night else ""
    wind = forecast.wind_direction or "未知"
    wind_scale = forecast.wind_scale or "未知"
    date_label = forecast.forecast_date.isoformat()
    suggestion = _forecast_suggestion(forecast.temp_max_c, forecast.condition_day)

    return (
        f"{forecast.location}{date_label}白天{forecast.condition_day}{night}，预计气温 {temp_min}~{temp_max}，"
        f"湿度 {humidity}，风向 {wind}，风力 {wind_scale}。{suggestion}"
    )


def format_route_reply(
    origin: str,
    destination: str,
    plans: list[RoutePlan],
    goal: str | None = None,
) -> str:
    primary = _pick_primary_plan(plans, goal)
    if goal == "fastest":
        opening = f"从{origin}到{destination}，最快大约 {primary.duration_minutes} 分钟（{primary.mode}）。"
    elif goal == "least_walking":
        opening = f"从{origin}到{destination}，少步行优先建议 {primary.mode}，大约 {primary.duration_minutes} 分钟。"
    elif goal == "cheapest":
        opening = f"从{origin}到{destination}，更省钱的建议是 {primary.mode}，大约 {primary.duration_minutes} 分钟。"
    else:
        opening = f"从{origin}到{destination}，我先推荐 {primary.mode}，大约 {primary.duration_minutes} 分钟。"

    lines: list[str] = [opening, "我帮你整理了几个可选路线："]
    for index, plan in enumerate(plans, start=1):
        distance = f"，步行约 {plan.distance_km:.1f} km" if plan.distance_km is not None else ""
        lines.append(f"方案{index}：{plan.mode}，预计 {plan.duration_minutes} 分钟{distance}。{plan.summary}")

    lines.append("如果你告诉我出发时间和是否赶时间，我可以再帮你细化成更稳妥的一条。")
    return "\n".join(lines)


def format_search_reply(query: str, results: list[SearchResult]) -> tuple[str, list[str]]:
    references = [item.link for item in results if item.link]
    top_titles = [item.title.strip() for item in results if item.title.strip()][:3]
    title_hint = "；".join(top_titles) if top_titles else "暂无稳定来源标题"
    sample = _short_snippet(results[0].snippet if results else None)

    return (
        f"我已经帮你完成“{query}”的联网检索，并先做了一轮信息筛选。"
        f"这次综合了 {len(results)} 条结果，重点参考包括：{title_hint}。{sample}"
        "如果你愿意，我可以继续按你的预算、天数和偏好整理成可执行方案。"
    ), references


def format_nearby_reply(
    *,
    location: str,
    keyword: str,
    radius_m: int,
    places: list[NearbyPlace],
) -> str:
    if not places:
        return (
            f"在{location}附近 {radius_m} 米内，我暂时没找到明确的“{keyword}”结果。"
            "你可以试试换个关键词（例如“棋牌室”“休闲娱乐”）或把范围扩大到 2 公里。"
        )

    lines = [f"在{location}附近 {radius_m} 米内，我找到了 {len(places)} 个和“{keyword}”相关的地点："]
    for index, place in enumerate(places[:5], start=1):
        distance = f"{place.distance_m} 米" if place.distance_m is not None else "距离未知"
        address = place.address or "地址未提供"
        lines.append(f"{index}. {place.name}（{distance}，{address}）")
    lines.append("如果你愿意，我可以继续按“最近/步行可达/更安静”帮你再筛一轮。")
    return "\n".join(lines)


def format_weather_search_fallback_reply(
    *,
    location: str,
    target_date_text: str,
    available_start: str,
    available_end: str,
    reliable_results: list[SearchResult],
    focus: str,
    temperature_estimate: str | None,
) -> tuple[str, list[str]]:
    references = [item.link for item in reliable_results if item.link]
    top_titles = [item.title.strip() for item in reliable_results if item.title.strip()][:3]
    title_hint = "；".join(top_titles)

    if focus == "temperature":
        if temperature_estimate:
            return (
                f"按联网长周期预报看，{location}{target_date_text}温度大致在 {temperature_estimate}（仅供参考）。"
                f"和风天气当前可预报范围是 {available_start} 到 {available_end}，超过7天，天气数据来自于网络，不一定准确。"
                f"{f' 参考信息：{title_hint}。' if title_hint else ''}"
                "建议你在出行前 1-3 天再查一次，我可以再帮你做最终决策。"
            ), references

        return (
            f"目前我没法给出{location}{target_date_text}的可靠具体温度。"
            f"和风天气当前可预报范围是 {available_start} 到 {available_end}，超过7天，天气数据来自于网络，不一定准确。"
            "这次检索里也没有该日期的明确温度数值。"
            "建议在出行前 1-3 天再查一次，我会给你更可靠的判断。"
        ), references

    if reliable_results:
        return (
            f"我检索到了一些与{location}{target_date_text}相关的网络天气信息。"
            f"和风天气当前可预报范围是 {available_start} 到 {available_end}，超过7天，天气数据来自于网络，不一定准确。"
            f"{f' 参考信息：{title_hint}。' if title_hint else ''}"
            "建议临近出行前 1-3 天再确认一次，我可以再帮你把方案定下来。"
        ), references

    return (
        f"目前还没检索到与{location}{target_date_text}强相关、可直接采用的网络天气预报。"
        f"和风天气当前可预报范围是 {available_start} 到 {available_end}，超过7天，天气数据来自于网络，不一定准确。"
        "这次结果和目标城市/日期的匹配度不够。"
        "建议在出行前 1-3 天再查一次，我会给你更可靠的建议。"
    ), []


def format_unavailable_reply(reason: str) -> str:
    return f"我这边暂时没法稳定完成这次查询（{reason}）。你可以稍后再试，或者把问题说得更具体一点，我继续帮你处理。"


def _weather_suggestion(temp_c: float, condition: str) -> str:
    if "雨" in condition:
        return "建议随身带伞，优先选择地铁或网约车，避免步行过久。"
    if temp_c <= 10:
        return "建议保暖出行，早晚温差大可加外套。"
    if temp_c >= 30:
        return "建议避开中午暴晒时段，并注意补水。"
    return "天气较舒适，常规出行即可。"


def _forecast_suggestion(max_temp_c: float | None, condition: str) -> str:
    if "雨" in condition:
        return "建议带伞并预留通勤缓冲时间。"
    if max_temp_c is None:
        return "建议出门前再次确认临近天气变化。"
    if max_temp_c <= 10:
        return "建议做好保暖，早晚温差通常更明显。"
    if max_temp_c >= 30:
        return "建议注意防晒和补水，避开午后高温时段。"
    return "天气整体适合常规出行。"


def _pick_primary_plan(plans: list[RoutePlan], goal: str | None) -> RoutePlan:
    if not plans:
        raise ValueError("plans should not be empty")

    if goal == "fastest":
        return min(plans, key=lambda plan: plan.duration_minutes)

    if goal == "least_walking":
        ranked = sorted(
            plans,
            key=lambda plan: (
                plan.distance_km if plan.distance_km is not None else float("inf"),
                plan.duration_minutes,
            ),
        )
        return ranked[0]

    if goal == "cheapest":
        transit_plans = [plan for plan in plans if plan.mode == "transit"]
        if transit_plans:
            return min(transit_plans, key=lambda plan: plan.duration_minutes)

    priority = {"transit": 0, "driving": 1, "walking": 2}
    return sorted(plans, key=lambda plan: (priority.get(plan.mode, 99), plan.duration_minutes))[0]


def _short_snippet(snippet: str | None) -> str:
    if not snippet:
        return ""
    compact = " ".join(snippet.split())
    if len(compact) > 80:
        compact = f"{compact[:80]}..."
    return f" 检索摘要：{compact}"
