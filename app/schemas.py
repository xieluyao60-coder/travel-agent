from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    WEATHER = "weather"
    ROUTE = "route"
    SEARCH = "search"
    NEARBY = "nearby"
    MEMORY = "memory"
    CHAT = "chat"


class IntentDecision(BaseModel):
    intent: IntentType
    location: str | None = None
    origin: str | None = None
    destination: str | None = None
    query: str | None = None


class WeatherResult(BaseModel):
    location: str
    condition: str
    temperature_c: float
    feels_like_c: float | None = None
    humidity: int | None = None
    wind_direction: str | None = None
    wind_scale: str | None = None
    observed_at: str | None = None


class WeatherForecastResult(BaseModel):
    location: str
    forecast_date: date
    condition_day: str
    condition_night: str | None = None
    temp_min_c: float | None = None
    temp_max_c: float | None = None
    humidity: int | None = None
    wind_direction: str | None = None
    wind_scale: str | None = None
    sunrise: str | None = None
    sunset: str | None = None


class RoutePlan(BaseModel):
    mode: Literal["transit", "driving", "walking"]
    duration_minutes: int
    distance_km: float | None = None
    summary: str


class NearbyPlace(BaseModel):
    name: str
    address: str | None = None
    distance_m: int | None = None
    category: str | None = None


class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str | None = None


class AssistantReply(BaseModel):
    intent: IntentType
    text: str
    references: list[str] = Field(default_factory=list)


class PlannerAction(str, Enum):
    CALL_TOOL = "call_tool"
    CLARIFY = "clarify"
    CHAT = "chat"


class PlannerToolName(str, Enum):
    WEATHER_NOW = "weather.now"
    ROUTE_PLAN = "route.plan"
    SEARCH_WEB = "search.search"
    NEARBY_SEARCH = "nearby.search"
    MEMORY_UPDATE = "memory.update"


class PlannerRouteParams(BaseModel):
    origin: str | None = None
    destination: str | None = None
    mode: Literal["transit", "driving", "walking"] | None = None
    goal: Literal["fastest", "cheapest", "least_walking", "balanced"] | None = None


class PlannerWeatherParams(BaseModel):
    location: str | None = None
    when: Literal["realtime", "date", "tomorrow", "week"] | None = None
    target_date: str | None = None


class PlannerSearchParams(BaseModel):
    query: str | None = None
    top_k: int = 5


class PlannerNearbyParams(BaseModel):
    location: str | None = None
    keyword: str | None = None
    radius_m: int | None = None


class PlannerMemoryParams(BaseModel):
    operation: Literal["set_city", "set_hotel", "clear_hotel", "reset_profile"] | None = None
    travel_city: str | None = None
    hotel_location: str | None = None


class PlannerOutput(BaseModel):
    action: PlannerAction
    intent: IntentType = IntentType.CHAT
    tool_name: PlannerToolName | None = None
    normalized_query: str
    confidence: float = 0.0
    weather: PlannerWeatherParams = Field(default_factory=PlannerWeatherParams)
    route: PlannerRouteParams = Field(default_factory=PlannerRouteParams)
    search: PlannerSearchParams = Field(default_factory=PlannerSearchParams)
    nearby: PlannerNearbyParams = Field(default_factory=PlannerNearbyParams)
    memory: PlannerMemoryParams = Field(default_factory=PlannerMemoryParams)
    missing_slots: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
