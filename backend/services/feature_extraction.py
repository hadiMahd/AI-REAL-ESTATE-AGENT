from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import Lock

from pydantic import ValidationError

try:
    from backend.schemas.stage_1 import Stage1ExtractionCandidates, Stage1Output, Stage1PromptCandidate
    from backend.services.llm_client import get_default_model, get_ollama_client
except ImportError:  # pragma: no cover - fallback for alternative run paths
    from schemas.stage_1 import Stage1ExtractionCandidates, Stage1Output, Stage1PromptCandidate
    from services.llm_client import get_default_model, get_ollama_client


SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent
PROMPTS_DIR = BACKEND_DIR / "prompts"
REGISTRY_PATH = PROMPTS_DIR / "registry.json"
LOGS_DIR = BACKEND_DIR / "logs"
STAGE1_LOG_FILE = LOGS_DIR / "stage1_extraction.jsonl"


# Prompt runs are parallel, so file appends are protected with a lock.
_log_write_lock = Lock()


@lru_cache(maxsize=1)
def _load_prompt_registry() -> dict[str, object]:
    """Load the prompt registry once so we do not read the file on every request."""
    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _get_stage1_prompt_config() -> dict[str, object]:
    """Return the stage1 prompt block from the registry and validate its shape."""
    registry = _load_prompt_registry()
    stage1_config = registry.get("stage1")
    if not isinstance(stage1_config, dict):
        raise ValueError("Invalid prompt registry: missing stage1 config")
    return stage1_config


def _get_stage1_prompt_candidates() -> list[str]:
    """Return the list of Stage 1 prompt files to run."""
    stage1_config = _get_stage1_prompt_config()
    candidates = stage1_config.get("candidates", [])
    if not isinstance(candidates, list) or not all(isinstance(item, str) and item for item in candidates):
        raise ValueError("Invalid prompt registry: stage1 candidates must be a list of filenames")

    # Put the default prompt first so the UI sees the preferred version before the alternatives.
    default_prompt = stage1_config.get("default")
    if isinstance(default_prompt, str) and default_prompt in candidates:
        ordered = [default_prompt]
        ordered.extend(candidate for candidate in candidates if candidate != default_prompt)
        return ordered

    return candidates


def _resolve_stage1_prompt_filename(prompt_version: str | None = None) -> str:
    """Resolve one Stage 1 prompt filename, either explicit or default."""
    stage1_config = _get_stage1_prompt_config()

    if prompt_version:
        candidates = stage1_config.get("candidates", [])
        if prompt_version not in candidates:
            raise ValueError(f"Unknown stage1 prompt version: {prompt_version}")
        return prompt_version

    default_prompt = stage1_config.get("default")
    if not isinstance(default_prompt, str) or not default_prompt:
        raise ValueError("Invalid prompt registry: missing stage1 default prompt")
    return default_prompt


def load_stage1_prompt_template(prompt_version: str | None = None) -> str:
    """Read one prompt template file from disk."""
    filename = _resolve_stage1_prompt_filename(prompt_version)
    prompt_path = PROMPTS_DIR / filename
    with prompt_path.open("r", encoding="utf-8") as file:
        return file.read()


def build_stage1_prompt(user_query: str, prompt_version: str | None = None) -> str:
    """Fill the prompt template with the user's free-text query."""
    if not user_query or not user_query.strip():
        raise ValueError("user_query must not be empty")

    template = load_stage1_prompt_template(prompt_version)
    return template.replace("{{user_query}}", user_query.strip())


def _parse_llm_json_response(raw_text: str) -> dict[str, object]:
    """Parse the model response into a JSON object, stripping code fences if needed."""
    text = raw_text.strip()
    if text.startswith("```"):
        # Some models still wrap JSON in markdown fences even when asked not to.
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Stage1 response must be a JSON object")
    return parsed


def _log_stage1_run(
    prompt_version: str,
    user_query: str,
    output: object,
    validation_result: str,
) -> None:
    """Append one Stage 1 prompt run log entry as JSONL in backend/logs."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "promptversion": prompt_version,
        "input": user_query,
        "output": output,
        "validation_result": validation_result,
    }

    # JSONL keeps writes simple and easy to inspect/parse later.
    with _log_write_lock:
        with STAGE1_LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _run_stage1_prompt(
    user_query: str,
    prompt_version: str,
    model: str,
) -> Stage1PromptCandidate:
    """Execute one prompt version and return either a validated result or an error."""
    try:
        prompt = build_stage1_prompt(user_query=user_query, prompt_version=prompt_version)
        client = get_ollama_client()
        response = client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            format="json",
        )
        raw_output = response.response or ""
        parsed = _parse_llm_json_response(raw_output)
        output = Stage1Output.model_validate(parsed)
        _log_stage1_run(
            prompt_version=prompt_version,
            user_query=user_query,
            output=parsed,
            validation_result="valid",
        )
        return Stage1PromptCandidate(prompt_version=prompt_version, output=output)
    except ValidationError as exc:
        _log_stage1_run(
            prompt_version=prompt_version,
            user_query=user_query,
            output={"error": str(exc)},
            validation_result="invalid",
        )
        return Stage1PromptCandidate(prompt_version=prompt_version, error=f"Stage1 validation failed: {exc}")
    except Exception as exc:
        _log_stage1_run(
            prompt_version=prompt_version,
            user_query=user_query,
            output={"error": str(exc)},
            validation_result="invalid",
        )
        return Stage1PromptCandidate(prompt_version=prompt_version, error=str(exc))


def extract_feature_candidates(
    user_query: str,
    model: str | None = None,
) -> Stage1ExtractionCandidates:
    """Run every Stage 1 prompt version in parallel and return both candidate results."""
    if not user_query or not user_query.strip():
        raise ValueError("user_query must not be empty")

    target_model = model or get_default_model()
    prompt_versions = _get_stage1_prompt_candidates()

    # Run both prompt versions at the same time so the UI can compare them side by side.
    with ThreadPoolExecutor(max_workers=len(prompt_versions)) as executor:
        futures = [
            executor.submit(_run_stage1_prompt, user_query.strip(), prompt_version, target_model)
            for prompt_version in prompt_versions
        ]

        # Preserve the original prompt ordering so the default prompt stays first.
        candidates = [future.result() for future in futures]

    return Stage1ExtractionCandidates(
        query=user_query.strip(),
        model=target_model,
        candidates=candidates,
    )


def extract_features(
    user_query: str,
    prompt_version: str | None = None,
    model: str | None = None,
) -> Stage1Output:
    """Run one specific Stage 1 prompt version and return its validated output."""
    prompt = build_stage1_prompt(user_query=user_query, prompt_version=prompt_version)
    client = get_ollama_client()
    target_model = model or get_default_model()

    response = client.generate(
        model=target_model,
        prompt=prompt,
        stream=False,
        format="json",
    )

    raw_output = response.response or ""
    parsed = _parse_llm_json_response(raw_output)

    try:
        return Stage1Output.model_validate(parsed)
    except ValidationError as exc:
        raise ValueError(f"Stage1 validation failed: {exc}") from exc


def extract_features_dict(
    user_query: str,
    prompt_version: str | None = None,
    model: str | None = None,
) -> dict[str, object]:
    """Return one validated Stage 1 output as a plain dictionary for API responses."""
    result = extract_features(
        user_query=user_query,
        prompt_version=prompt_version,
        model=model,
    )
    return result.model_dump(by_alias=True)
