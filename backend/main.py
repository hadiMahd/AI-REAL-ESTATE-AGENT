from __future__ import annotations

import hashlib
import time

from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.services.redis_client import get_redis_client
from backend.services.llm_client import get_default_model, test_llm_response


class LLMHealthResponse(BaseModel):
    ok: bool
    model: str
    response: str
    elapsed_seconds: float = Field(ge=0)


app = FastAPI(title='House Price Assistant Backend')


def _get_client_key(request: Request) -> str:
    """Return a stable client identifier used as the rate-limit subject."""
    # Respect proxy forwarding when present, otherwise fall back to direct client IP.
    forwarded_for = request.headers.get('x-forwarded-for', '')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    return request.client.host if request.client else 'unknown'


def _rate_limit_redis_key(client_key: str) -> str:
    """Build the Redis key for the current client's fixed-window bucket."""
    settings = get_settings()
    # Hash the client identifier before storing it in Redis keys to avoid raw IP leakage.
    key_hash = hashlib.sha256(client_key.encode('utf-8')).hexdigest()
    now = time.time()
    window = settings.rate_limit_window_seconds
    # Fixed-window bucket index. Same bucket => same key => shared counter.
    bucket = int(now // window)
    return f"{settings.redis_key_prefix}:rl:{bucket}:{key_hash}"


def _enforce_rate_limit(request: Request) -> tuple[bool, int]:
    """Apply Redis-backed fixed-window rate limiting.

    Returns:
        (allowed, retry_after_seconds)
    """
    settings = get_settings()
    # Shared Redis client means all app workers/instances use the same counters.
    redis = get_redis_client()
    # Generate the current bucket key for this client.
    key = _rate_limit_redis_key(_get_client_key(request))

    # Use Redis INCR + EXPIRE for atomic fixed-window limiting shared by all instances.
    # INCR atomically increments and returns the new count for this key.
    current_count = redis.incr(key)
    if current_count == 1:
        # First hit in this bucket: set TTL so Redis auto-cleans the bucket key.
        redis.expire(key, settings.rate_limit_window_seconds)

    if current_count > settings.rate_limit_requests:
        # TTL tells how many seconds remain before this window expires.
        retry_after = redis.ttl(key)
        # Guard against edge cases where TTL is not available.
        return False, max(1, retry_after if retry_after > 0 else settings.rate_limit_window_seconds)

    return True, 0


@app.get('/health/llm', response_model=LLMHealthResponse)
def health_llm(request: Request) -> LLMHealthResponse | JSONResponse:
    """Health endpoint that is protected by Redis rate limiting and Redis cache."""
    try:
        allowed, retry_after = _enforce_rate_limit(request)
    except Exception as exc:
        # If limiter storage is down, fail closed to avoid unbounded traffic spikes.
        raise HTTPException(status_code=503, detail=f'Rate limiter unavailable: {exc}') from exc

    if not allowed:
        # Explicit 429 response instead of raising to preserve Retry-After header.
        return JSONResponse(
            status_code=429,
            content={'detail': 'Rate limit exceeded. Try again shortly.'},
            headers={'Retry-After': str(retry_after)},
        )

    try:
        result = test_llm_response(get_default_model())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return LLMHealthResponse(**result)


def main() -> None:
    print('FastAPI app is ready. Import `app` from backend.main to run it.')


if __name__ == '__main__':
    main()