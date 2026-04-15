from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
ENV_FILE = BACKEND_DIR / ".env"

# Load backend/.env once at import time. Existing process env vars keep priority.
load_dotenv(dotenv_path=ENV_FILE, override=False)


def _read_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    if value is None:
        return default

    cleaned = value.strip()
    # Treat empty env values as missing so defaults still apply.
    return cleaned if cleaned else default


@dataclass(frozen=True)
class Settings:
    ollama_model: str
    ollama_timeout: float
    ollama_api_key: str
    # Redis connection URL used by shared cache and shared rate limiter.
    redis_url: str
    # Max requests allowed per window for one client key.
    rate_limit_requests: int
    # Window length in seconds for fixed-window rate limiting.
    rate_limit_window_seconds: int
    # TTL (seconds) for cached LLM health payload in Redis.
    llm_health_cache_ttl_seconds: int
    # Prefix namespace to avoid key collisions in shared Redis.
    redis_key_prefix: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate runtime settings from environment variables once."""
    model = _read_env("OLLAMA_MODEL")
    if not model:
        model = "gpt-oss:120b"

    api_key = _read_env("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("Missing OLLAMA_API_KEY. Cloud mode requires an API key.")

    timeout_raw = _read_env("OLLAMA_TIMEOUT", "120")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 120.0

    redis_url = _read_env("REDIS_URL", "redis://localhost:6379/0")

    rate_limit_requests_raw = _read_env("RATE_LIMIT_REQUESTS", "10")
    try:
        rate_limit_requests = max(1, int(rate_limit_requests_raw))
    except ValueError:
        rate_limit_requests = 10

    rate_limit_window_raw = _read_env("RATE_LIMIT_WINDOW_SECONDS", "60")
    try:
        rate_limit_window_seconds = max(1, int(rate_limit_window_raw))
    except ValueError:
        rate_limit_window_seconds = 60

    cache_ttl_raw = _read_env("LLM_HEALTH_CACHE_TTL_SECONDS", "20")
    try:
        llm_health_cache_ttl_seconds = max(1, int(cache_ttl_raw))
    except ValueError:
        llm_health_cache_ttl_seconds = 20

    redis_key_prefix = _read_env("REDIS_KEY_PREFIX", "house_price_api")

    return Settings(
        ollama_model=model,
        ollama_timeout=timeout,
        ollama_api_key=api_key,
        redis_url=redis_url,
        rate_limit_requests=rate_limit_requests,
        rate_limit_window_seconds=rate_limit_window_seconds,
        llm_health_cache_ttl_seconds=llm_health_cache_ttl_seconds,
        redis_key_prefix=redis_key_prefix,
    )