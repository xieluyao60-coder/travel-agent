from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import Settings
from app.providers.llm import OpenAICompatibleLLM
from app.providers.route import AmapRouteProvider
from app.providers.search import SerpApiProvider
from app.providers.weather import QWeatherProvider
from app.services.orchestrator import ChatOrchestrator
from app.services.planner import QueryPlanner
from app.services.reply_agent import ReplyAgent
from app.services.session_store import SessionStore


@dataclass
class AppContainer:
    settings: Settings
    orchestrator: ChatOrchestrator
    session_store: SessionStore
    http_client: httpx.AsyncClient

    async def aclose(self) -> None:
        await self.session_store.close()
        await self.http_client.aclose()


async def build_container(settings: Settings) -> AppContainer:
    http_client = httpx.AsyncClient(timeout=settings.http_timeout_seconds, trust_env=False)

    session_store = SessionStore(
        redis_url=settings.redis_url,
        ttl_seconds=settings.session_ttl_seconds,
        history_limit=settings.session_history_limit,
        profile_ttl_seconds=settings.profile_ttl_seconds,
    )
    await session_store.connect()

    weather_provider = QWeatherProvider(
        api_key=settings.qweather_api_key,
        api_host=settings.qweather_api_host,
        client=http_client,
    )
    route_provider = AmapRouteProvider(
        api_key=settings.amap_api_key,
        default_city=settings.amap_default_city,
        client=http_client,
    )
    search_provider = SerpApiProvider(api_key=settings.serpapi_api_key, client=http_client)
    llm_provider = OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        client=http_client,
    )
    planner = QueryPlanner(llm_provider=llm_provider)
    reply_agent = ReplyAgent(llm_provider=llm_provider)

    orchestrator = ChatOrchestrator(
        planner=planner,
        reply_agent=reply_agent,
        session_store=session_store,
        weather_provider=weather_provider,
        route_provider=route_provider,
        search_provider=search_provider,
        llm_provider=llm_provider,
        history_limit=settings.session_history_limit,
        memory_enabled=settings.memory_enabled,
    )

    return AppContainer(
        settings=settings,
        orchestrator=orchestrator,
        session_store=session_store,
        http_client=http_client,
    )
