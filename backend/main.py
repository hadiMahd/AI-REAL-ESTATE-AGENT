from __future__ import annotations

import json
import hashlib
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.schemas.stage_1 import Stage1ExtractedFeatures, Stage1ExtractionCandidates, Stage1Input, Stage1Output
from backend.schemas.stage_2 import (
    PipelineInterpretationResponse,
    PipelinePredictionResponse,
    PredictionInterpretationRequest,
    PredictionInterpretationResponse,
    PredictionResponse,
)
from backend.services.feature_extraction import extract_feature_candidates
from backend.services.llm_client import get_default_model, test_llm_response
from backend.services.prediction_interpretation import interpret_prediction
from backend.services.price_prediction import predict_price
from backend.services.redis_client import get_redis_client


class LLMHealthResponse(BaseModel):
    ok: bool
    model: str
    response: str
    elapsed_seconds: float = Field(ge=0)


class APIHealthResponse(BaseModel):
    status: str


app = FastAPI(title='House Price Assistant Backend')
LOGS_DIR = Path(__file__).resolve().parent / 'logs'
STAGE1_LOG_FILE = LOGS_DIR / 'stage1_extraction.jsonl'


class Stage1LogsResponse(BaseModel):
    count: int
    entries: list[dict[str, Any]] = Field(default_factory=list)


def _select_best_candidate(candidates: Stage1ExtractionCandidates) -> tuple[str, Stage1Output]:
    """Pick the best valid Stage 1 candidate for sequential pipeline execution."""
    valid_candidates = [
        candidate
        for candidate in candidates.candidates
        if candidate.output is not None and candidate.output.ready_for_prediction
    ]
    if not valid_candidates:
        raise ValueError('No Stage 1 candidate is complete enough for prediction. Use /ml/predict for manual ML testing or fill the missing fields first.')

    best_candidate = max(valid_candidates, key=lambda candidate: candidate.output.completeness)
    return best_candidate.prompt_version, best_candidate.output


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


@app.get('/health', response_model=APIHealthResponse)
def health() -> APIHealthResponse:
    """Stage flow: infrastructure check only (no Stage 1/ML/interpretation)."""
    return APIHealthResponse(status='ok')


@app.get('/health/llm', response_model=LLMHealthResponse)
def health_llm(request: Request) -> LLMHealthResponse | JSONResponse:
    """Stage flow: LLM connectivity check only (pre-Stage 1)."""
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


@app.post('/stage1/extract', response_model=Stage1ExtractionCandidates)
def stage1_extract(payload: Stage1Input) -> Stage1ExtractionCandidates:
    """Stage flow: Stage 1 only (free text -> structured extraction candidates)."""
    try:
        # Run the extraction service and return both prompt candidates.
        return extract_feature_candidates(user_query=payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'Stage1 extraction failed: {exc}') from exc


@app.post('/ml/predict', response_model=PredictionResponse)
def ml_predict(
    payload: Stage1ExtractedFeatures = Body(
        example={
            'OverallQual': 7,
            'GrLivArea': 1800,
            'GarageCars': 2,
            'FullBath': 2,
            'YearBuilt': 1995,
            'YearRemodAdd': 2000,
            'MasVnrArea': 0.0,
            'Fireplaces': 1,
            'BsmtFinSF1': 400,
            'LotFrontage': 70.0,
            '1stFlrSF': 1000,
            'OpenPorchSF': 50,
        }
    ),
) -> PredictionResponse:
    """Stage flow: ML stage only (structured features -> predicted price)."""
    try:
        # Call the trained model with validated structured features.
        return predict_price(features=payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'Price prediction failed: {exc}') from exc


@app.post('/interpret/predict', response_model=PredictionInterpretationResponse)
def interpret_only(
    payload: PredictionInterpretationRequest = Body(
        example={
            'features': {
                'OverallQual': 7,
                'GrLivArea': 1800,
                'GarageCars': 2,
                'FullBath': 2,
                'YearBuilt': 1995,
                'YearRemodAdd': 2000,
                'MasVnrArea': 0.0,
                'Fireplaces': 1,
                'BsmtFinSF1': 400,
                'LotFrontage': 70.0,
                '1stFlrSF': 1000,
                'OpenPorchSF': 50,
            },
            'prediction_value': 214189.49,
        }
    ),
) -> PredictionInterpretationResponse:
    """Stage flow: interpretation stage only (features + prediction -> explanation)."""
    try:
        prediction_value = payload.prediction_value
        if prediction_value is None:
            if payload.prediction is None:
                raise ValueError('Provide either prediction_value or prediction in the request body.')
            prediction_value = payload.prediction.predicted_price

        # Interpret an already-known prediction without running extraction or ML.
        return interpret_prediction(
            features=payload.features,
            prediction_value=prediction_value,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'Prediction interpretation failed: {exc}') from exc


@app.post('/pipeline/predict', response_model=PipelinePredictionResponse)
def pipeline_predict(payload: Stage1Input) -> PipelinePredictionResponse:
    """Stage flow: Stage 1 -> ML prediction."""
    try:
        # 1) Stage 1 service function
        candidates = extract_feature_candidates(user_query=payload.query)

        # 2) Pick best candidate output from Stage 1
        selected_prompt_version, selected_output = _select_best_candidate(candidates)

        # 3) Stage 2 service function
        prediction = predict_price(features=selected_output.features)

        return PipelinePredictionResponse(
            query=payload.query,
            selected_prompt_version=selected_prompt_version,
            stage1_output=selected_output,
            prediction=prediction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'Pipeline prediction failed: {exc}') from exc


@app.post('/pipeline/interpret', response_model=PipelineInterpretationResponse)
def pipeline_interpret(payload: Stage1Input) -> PipelineInterpretationResponse:
    """Stage flow: Stage 1 -> ML prediction -> interpretation."""
    try:
        # Sequential orchestration across all three service stages.
        candidates = extract_feature_candidates(user_query=payload.query)
        selected_prompt_version, selected_output = _select_best_candidate(candidates)
        prediction = predict_price(features=selected_output.features)
        interpretation = interpret_prediction(
            features=selected_output.features,
            prediction_value=prediction.predicted_price,
        )
        return PipelineInterpretationResponse(
            query=payload.query,
            selected_prompt_version=selected_prompt_version,
            stage1_output=selected_output,
            prediction=prediction,
            interpretation=interpretation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'Pipeline interpretation failed: {exc}') from exc


@app.get('/logs/stage1', response_model=Stage1LogsResponse)
def get_stage1_logs(limit: int = Query(default=20, ge=1, le=200)) -> Stage1LogsResponse:
    """Stage flow: diagnostics only for Stage 1 logs (no stage execution)."""
    if not STAGE1_LOG_FILE.exists():
        return Stage1LogsResponse(count=0, entries=[])

    # Keep only the latest N lines to avoid loading very large logs into memory.
    last_lines: deque[str] = deque(maxlen=limit)
    with STAGE1_LOG_FILE.open('r', encoding='utf-8') as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                last_lines.append(stripped)

    entries: list[dict[str, Any]] = []
    for line in last_lines:
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                entries.append(parsed)
        except json.JSONDecodeError:
            continue

    return Stage1LogsResponse(count=len(entries), entries=entries)


def main() -> None:
    print('FastAPI app is ready. Import `app` from backend.main to run it.')


if __name__ == '__main__':
    main()