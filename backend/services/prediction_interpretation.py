"""Stage 3 prediction interpretation service using training stats and feature signals."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

try:
	from backend.schemas.stage_1 import Stage1ExtractedFeatures
	from backend.schemas.stage_2 import PredictionInterpretationResponse
	from backend.services.llm_client import get_default_model, get_ollama_client
except ImportError:  # pragma: no cover - fallback for alternative run paths
	from schemas.stage_1 import Stage1ExtractedFeatures
	from schemas.stage_2 import PredictionInterpretationResponse
	from services.llm_client import get_default_model, get_ollama_client


SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent
PROMPTS_DIR = BACKEND_DIR / 'prompts'
REGISTRY_PATH = PROMPTS_DIR / 'registry.json'
PRICE_SUMMARY_PATH = BACKEND_DIR / 'artifacts' / 'price_summary_stats.json'
FEATURE_METRICS_PATH = BACKEND_DIR / 'artifacts' / 'ames_input_metrics.json'

FEATURE_LABELS = {
	'OverallQual': 'Overall quality',
	'GrLivArea': 'Above-ground living area',
	'GarageCars': 'Garage capacity',
	'FullBath': 'Full bathrooms',
	'YearBuilt': 'Year built',
	'YearRemodAdd': 'Year remodeled',
	'MasVnrArea': 'Masonry veneer area',
	'Fireplaces': 'Fireplaces',
	'BsmtFinSF1': 'Finished basement area',
	'LotFrontage': 'Lot frontage',
	'1stFlrSF': 'First floor area',
	'OpenPorchSF': 'Open porch area',
}


def _is_missing_number(value: object) -> bool:
	"""Return True for missing, zero, or non-numeric placeholders."""
	return value is None or value == 0 or value == 0.0


def _sanitize_price_summary(summary_payload: dict[str, object]) -> dict[str, object]:
	"""Replace placeholder summary values with explicit nulls so the prompt does not treat them as real stats."""
	sanitized = dict(summary_payload)
	median_price = sanitized.get('median_price')
	if _is_missing_number(median_price):
		sanitized['median_price'] = None

	typical_range = sanitized.get('typical_range')
	if isinstance(typical_range, dict):
		range_copy = dict(typical_range)
		if _is_missing_number(range_copy.get('low')):
			range_copy['low'] = None
		if _is_missing_number(range_copy.get('high')):
			range_copy['high'] = None
		sanitized['typical_range'] = range_copy

	return sanitized


def _load_price_summary_payload(price_summary_json: dict[str, object] | None = None) -> dict[str, object]:
	"""Load and sanitize the sale price summary payload used by the prompt."""
	if price_summary_json is not None:
		return _sanitize_price_summary(price_summary_json)

	return _sanitize_price_summary(_load_json_file(PRICE_SUMMARY_PATH))


def _load_feature_metrics_payload(feature_metrics_json: dict[str, object] | None = None) -> dict[str, object]:
	"""Load the feature metrics payload used by the prompt."""
	if feature_metrics_json is not None:
		return feature_metrics_json

	return _load_json_file(FEATURE_METRICS_PATH)


def _load_json_file(path: Path) -> dict[str, object]:
	"""Load a JSON artifact from disk and ensure it is an object."""
	if not path.exists():
		raise FileNotFoundError(f'JSON artifact not found at {path}')

	with path.open('r', encoding='utf-8') as file:
		parsed = json.load(file)

	if not isinstance(parsed, dict):
		raise ValueError(f'JSON artifact at {path} must contain an object')
	return parsed


@lru_cache(maxsize=1)
def _load_prompt_registry() -> dict[str, object]:
	"""Load the prompt registry once so we do not read the file on every request."""
	return _load_json_file(REGISTRY_PATH)


def _get_stage2_prompt_config() -> dict[str, object]:
	"""Return the stage2 prompt block from the registry and validate its shape."""
	registry = _load_prompt_registry()
	stage2_config = registry.get('stage2')
	if not isinstance(stage2_config, dict):
		raise ValueError('Invalid prompt registry: missing stage2 config')
	return stage2_config


def _resolve_stage2_prompt_filename(prompt_version: str | None = None) -> str:
	"""Resolve the stage2 prompt filename, either explicit or default."""
	stage2_config = _get_stage2_prompt_config()
	if prompt_version:
		default_prompt = stage2_config.get('default')
		if prompt_version != default_prompt:
			raise ValueError(f'Unknown stage2 prompt version: {prompt_version}')
		return prompt_version

	default_prompt = stage2_config.get('default')
	if not isinstance(default_prompt, str) or not default_prompt:
		raise ValueError('Invalid prompt registry: missing stage2 default prompt')
	return default_prompt


def load_stage2_prompt_template(prompt_version: str | None = None) -> str:
	"""Read the stage2 interpretation prompt template from disk."""
	filename = _resolve_stage2_prompt_filename(prompt_version)
	prompt_path = PROMPTS_DIR / filename
	with prompt_path.open('r', encoding='utf-8') as file:
		return file.read()


def build_prediction_interpretation_prompt(
	features: Stage1ExtractedFeatures,
	prediction_value: float,
	price_summary_json: dict[str, object] | None = None,
	feature_metrics_json: dict[str, object] | None = None,
	prompt_version: str | None = None,
) -> str:
	"""Fill the stage2 interpretation prompt with feature, prediction, and summary data."""
	template = load_stage2_prompt_template(prompt_version)
	features_json = features.model_dump(by_alias=True)
	summary_payload = _load_price_summary_payload(price_summary_json)
	metrics_payload = _load_feature_metrics_payload(feature_metrics_json)
	prediction_context = (
		'prediction_value is unusually small for a raw house-price prediction. Mention that it may be scaled or manually entered if that seems likely.'
		if prediction_value < 1000
		else 'prediction_value is a raw USD house-price prediction.'
	)

	return (
		template
		.replace('{{features_json}}', json.dumps(features_json, ensure_ascii=True))
		.replace('{{prediction_value}}', json.dumps(prediction_value))
		.replace('{{price_summary_json}}', json.dumps(summary_payload, ensure_ascii=True))
		.replace('{{feature_metrics_json}}', json.dumps(metrics_payload, ensure_ascii=True))
		.replace('{{prediction_context}}', prediction_context)
	)


def _parse_llm_json_response(raw_text: str) -> dict[str, object]:
	"""Parse the model response into a JSON object, stripping code fences if needed."""
	text = raw_text.strip()
	if text.startswith('```'):
		lines = text.splitlines()
		if len(lines) >= 3 and lines[-1].strip() == '```':
			text = '\n'.join(lines[1:-1]).strip()

	parsed = json.loads(text)
	if not isinstance(parsed, dict):
		raise ValueError('Prediction interpretation response must be a JSON object')
	return parsed


def _with_readable_feature_names(text: str) -> str:
	"""Replace schema keys with user-friendly feature names in free-form text."""
	updated = text
	for raw_name, label in FEATURE_LABELS.items():
		updated = updated.replace(raw_name, label)
	return updated


def _normalize_key_drivers(raw_items: object) -> list[str]:
	"""Return key drivers with readable feature names and plain user language."""
	if not isinstance(raw_items, list):
		return []

	normalized: list[str] = []
	for item in raw_items:
		if not isinstance(item, str):
			continue
		text = _with_readable_feature_names(item).strip()
		if text:
			normalized.append(text)
	return normalized


def _normalize_caveats(raw_items: object) -> list[str]:
	"""Keep caveats customer-friendly and avoid internal training-data language."""
	default_caveat = (
		'This price is an estimate. Real sale price can change based on neighborhood, location, nearby amenities, market timing, and property condition details.'
	)

	if not isinstance(raw_items, list):
		return [default_caveat]

	blocked_terms = (
		'training',
		'dataset',
		'model error',
		'missing values in the training',
	)

	filtered: list[str] = []
	for item in raw_items:
		if not isinstance(item, str):
			continue
		text = item.strip()
		if not text:
			continue
		lower_text = text.lower()
		if any(term in lower_text for term in blocked_terms):
			continue
		filtered.append(text)

	if not filtered:
		return [default_caveat]

	if not any('estimate' in item.lower() for item in filtered):
		filtered.append(default_caveat)

	return filtered


def _normalize_interpretation_payload(parsed: dict[str, object]) -> dict[str, object]:
	"""Apply final response shaping rules for user-facing interpretation fields."""
	normalized = dict(parsed)
	if isinstance(normalized.get('summary'), str):
		normalized['summary'] = _with_readable_feature_names(str(normalized['summary']))
	normalized['key_drivers'] = _normalize_key_drivers(normalized.get('key_drivers'))
	normalized['caveats'] = _normalize_caveats(normalized.get('caveats'))
	return normalized


def interpret_prediction(
	features: Stage1ExtractedFeatures,
	prediction_value: float,
	price_summary_json: dict[str, object] | None = None,
	feature_metrics_json: dict[str, object] | None = None,
	model: str | None = None,
) -> PredictionInterpretationResponse:
	"""Interpret a predicted sale price using extracted features and training statistics."""
	target_model = model or get_default_model()
	prompt = build_prediction_interpretation_prompt(
		features=features,
		prediction_value=prediction_value,
		price_summary_json=price_summary_json,
		feature_metrics_json=feature_metrics_json,
	)

	client = get_ollama_client()
	response = client.generate(
		model=target_model,
		prompt=prompt,
		stream=False,
		format='json',
	)

	raw_output = response.response or ''
	parsed = _parse_llm_json_response(raw_output)
	parsed = _normalize_interpretation_payload(parsed)

	try:
		return PredictionInterpretationResponse.model_validate(parsed)
	except ValidationError as exc:
		raise ValueError(f'Prediction interpretation validation failed: {exc}') from exc


def interpret_prediction_dict(
	features: Stage1ExtractedFeatures,
	prediction_value: float,
	price_summary_json: dict[str, object] | None = None,
	feature_metrics_json: dict[str, object] | None = None,
	model: str | None = None,
) -> dict[str, object]:
	"""Return one interpretation as a plain dictionary for API responses."""
	result = interpret_prediction(
		features=features,
		prediction_value=prediction_value,
		price_summary_json=price_summary_json,
		feature_metrics_json=feature_metrics_json,
		model=model,
	)
	return result.model_dump()
