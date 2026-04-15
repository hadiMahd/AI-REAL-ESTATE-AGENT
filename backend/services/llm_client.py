from __future__ import annotations

import json
import time
from functools import lru_cache

import orjson
from ollama import Client

CLOUD_OLLAMA_HOST = "https://ollama.com"


try:
    from backend.config import get_settings
    from backend.services.redis_client import get_redis_client
except ImportError:  # pragma: no cover - fallback for alternative run paths
    from config import get_settings
    from services.redis_client import get_redis_client


@lru_cache(maxsize=1)
def get_ollama_client() -> Client:
    """Create and memoize one Ollama client per process."""
    settings = get_settings()
    headers = None
    if settings.ollama_api_key:
        headers = {"Authorization": f"Bearer {settings.ollama_api_key}"}

    return Client(
        host=CLOUD_OLLAMA_HOST,
        timeout=settings.ollama_timeout,
        headers=headers,
    )


def get_default_model() -> str:
    """Return the default model configured for LLM calls."""
    return get_settings().ollama_model


def _health_cache_key(model: str) -> str:
    """Build the Redis key that stores the health-check payload for one model."""
    settings = get_settings()
    return f"{settings.redis_key_prefix}:llm_health:{model}"


def _get_cached_health(model: str) -> dict[str, object] | None:
    """Read health-check payload from Redis cache if still available."""
    redis = get_redis_client()
    # GET returns None when key does not exist or has expired.
    raw_payload = redis.get(_health_cache_key(model))
    if not raw_payload:
        return None

    # Payload is stored as JSON text in Redis.
    parsed = json.loads(raw_payload)
    return parsed if isinstance(parsed, dict) else None


def _set_cached_health(model: str, payload: dict[str, object]) -> None:
    """Write health-check payload to Redis with TTL-based expiry."""
    settings = get_settings()
    redis = get_redis_client()
    # EX sets TTL in seconds so Redis evicts stale health entries automatically.
    redis.set(
        _health_cache_key(model),
        orjson.dumps(payload).decode("utf-8"),
        ex=settings.llm_health_cache_ttl_seconds,
    )


def test_llm_response(model: str | None = None) -> dict[str, object]:
    """Return a lightweight LLM responsiveness result, using Redis cache first."""
    client = get_ollama_client()
    settings = get_settings()
    target_model = model or settings.ollama_model

    try:
        # Cache hit path: skip external model call for lower latency/cost.
        cached = _get_cached_health(target_model)
        if cached is not None:
            return cached
    except Exception:
        # Cache failures should not block the core health check.
        pass

    start_time = time.perf_counter()
    response = client.generate(
        model=target_model,
        prompt='Reply with a single word: pongshay.',
        stream=False,
    )
    elapsed_seconds = round(time.perf_counter() - start_time, 3)

    message = response.response.strip() if response.response else ''
    result = {
        'ok': bool(message),
        'model': target_model,
        'response': message,
        'elapsed_seconds': elapsed_seconds,
    }

    try:
        # Best-effort cache write: endpoint still succeeds even if Redis cache write fails.
        _set_cached_health(target_model, result)
    except Exception:
        # Cache failures should not block the core health check.
        pass

    return result