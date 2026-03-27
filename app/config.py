from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "travel-planner-assistant"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    redis_url: str = "redis://redis:6379/0"
    session_ttl_seconds: int = 24 * 60 * 60
    profile_ttl_seconds: int = 30 * 24 * 60 * 60
    session_history_limit: int = 12
    memory_enabled: bool = True
    http_timeout_seconds: float = 8.0

    wecom_token: str = ""
    wecom_agent_id: str = ""
    wecom_corp_id: str = ""
    wecom_encoding_aes_key: str = ""
    wecom_connection_mode: str = "webhook"
    wecom_bot_id: str = ""
    wecom_bot_secret: str = ""
    wecom_ws_url: str = "wss://openws.work.weixin.qq.com"
    wecom_ws_heartbeat_interval_seconds: int = 30
    wecom_ws_max_missed_heartbeat: int = 2
    wecom_ws_reconnect_base_delay_seconds: float = 1.0
    wecom_ws_reconnect_max_delay_seconds: float = 30.0
    wecom_ws_max_auth_failure_attempts: int = 5

    qq_enabled: bool = False
    qq_bot_app_id: str = ""
    qq_bot_client_secret: str = ""
    qq_api_base_url: str = "https://api.sgroup.qq.com"
    qq_auth_base_url: str = "https://bots.qq.com"
    qq_ws_intents: int = 1107296256
    qq_ws_max_missed_heartbeat: int = 2
    qq_ws_reconnect_base_delay_seconds: float = 1.0
    qq_ws_reconnect_max_delay_seconds: float = 30.0
    qq_ws_max_auth_failure_attempts: int = 5
    qq_event_dedup_ttl_seconds: int = 120
    qq_event_dedup_max_size: int = 5000

    qweather_api_key: str = ""
    qweather_api_host: str = ""
    amap_api_key: str = ""
    amap_default_city: str = "上海"
    serpapi_api_key: str = ""

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-5.4-mini"
    llm_timeout_seconds: float = 8.0
    llm_temperature: float = 0.2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)
