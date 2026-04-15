from __future__ import annotations

from functools import lru_cache

from redis import Redis

try:
    from backend.config import get_settings
except ImportError:  # pragma: no cover - fallback for alternative run paths
    from config import get_settings


@lru_cache(maxsize=1)
def get_redis_client() -> Redis:
    """Create and memoize one Redis client per process.

    Reusing one client avoids reconnect overhead on each request.
    """
    settings = get_settings()
    # decode_responses=True returns str values (not bytes), which simplifies JSON parsing.
    return Redis.from_url(settings.redis_url, decode_responses=True)