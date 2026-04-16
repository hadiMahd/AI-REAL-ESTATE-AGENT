from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
STAGE1_EXTRACT_ENDPOINT = f"{BACKEND_URL}/stage1/extract"
ML_PREDICT_ENDPOINT = f"{BACKEND_URL}/ml/predict"
INTERPRET_ENDPOINT = f"{BACKEND_URL}/interpret/predict"

FEATURE_ORDER = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "Fireplaces",
    "BsmtFinSF1",
    "LotFrontage",
    "1stFlrSF",
    "OpenPorchSF",
]

FLOAT_FEATURES = {"MasVnrArea", "LotFrontage"}

FEATURE_LABELS = {
    "OverallQual": "Overall quality",
    "GrLivArea": "Above-ground living area (sq ft)",
    "GarageCars": "Garage capacity (cars)",
    "FullBath": "Full bathrooms",
    "YearBuilt": "Year built",
    "YearRemodAdd": "Year remodeled",
    "MasVnrArea": "Masonry veneer area (sq ft)",
    "Fireplaces": "Fireplaces",
    "BsmtFinSF1": "Finished basement area (sq ft)",
    "LotFrontage": "Lot frontage (ft)",
    "1stFlrSF": "First floor area (sq ft)",
    "OpenPorchSF": "Open porch area (sq ft)",
}


def _normalize_feature_key(key: str) -> str:
    if key == "first_flr_sf":
        return "1stFlrSF"
    return key


def _canonicalize_features(raw_features: dict[str, Any] | None) -> dict[str, Any]:
    canonical = {name: None for name in FEATURE_ORDER}
    if not raw_features:
        return canonical

    for raw_key, value in raw_features.items():
        key = _normalize_feature_key(raw_key)
        if key in canonical:
            canonical[key] = value
    return canonical


def _friendly_feature_items(features: dict[str, Any]) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for feature_name in FEATURE_ORDER:
        items.append((FEATURE_LABELS.get(feature_name, feature_name), features.get(feature_name)))
    return items


def _format_feature_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _pick_default_candidate_index(candidates: list[dict[str, Any]]) -> int:
    best_index = 0
    best_score = -1.0
    for idx, candidate in enumerate(candidates):
        output = candidate.get("output")
        if not isinstance(output, dict):
            continue
        completeness = float(output.get("completeness", 0.0) or 0.0)
        if completeness > best_score:
            best_score = completeness
            best_index = idx
    return best_index


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise RuntimeError(f"{url} returned {response.status_code}: {detail}")
    return response.json()


def _run_ml_and_interpret(feature_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction = _post_json(ML_PREDICT_ENDPOINT, feature_payload)
    interpretation = _post_json(
        INTERPRET_ENDPOINT,
        {
            "features": feature_payload,
            "prediction": prediction,
        },
    )
    return prediction, interpretation


def _render_final_result(prediction: dict[str, Any], interpretation: dict[str, Any]) -> None:
    st.subheader("Final Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Price", f"${prediction.get('predicted_price', 0):,.2f}")
        st.write(f"Model: {prediction.get('model_version', 'N/A')}")
    with col2:
        position = str(interpretation.get("position_vs_market", "unknown"))
        st.write("Position vs Market")
        st.success(position.replace("_", " ").title())

    st.subheader("Interpretation")
    interpretation_text = interpretation.get("summary")
    if not isinstance(interpretation_text, str) or not interpretation_text.strip():
        interpretation_text = str(interpretation)
    st.text(interpretation_text)

    key_drivers = interpretation.get("key_drivers")
    if isinstance(key_drivers, list) and key_drivers:
        st.subheader("Key Drivers")
        for item in key_drivers:
            st.text(str(item))

    caveats = interpretation.get("caveats")
    if isinstance(caveats, list) and caveats:
        st.subheader("Caveats")
        for item in caveats:
            st.text(str(item))

st.set_page_config(page_title="House Price Assistant", page_icon="🏠", layout="wide")
st.title("AI Real Estate Agent")
st.markdown(
    """
1. Enter a house description.
2. Compare the extracted feature sets.
3. Fill any missing values.
4. Get the predicted price and explanation.
"""
)

with st.form("query_form"):
    query = st.text_area(
        "House description",
        placeholder=(
            "Example: Beautiful 2-story house built in 1995 remodeled in 2000 with overall quality 7, "
            "1800 sqft living area, 2-car garage, 2 full baths, masonry veneer area 0, 1 fireplace, "
            "400 sqft finished basement, 70 ft lot frontage, 1000 sqft first floor, and 50 sqft open porch."
        ),
        height=180,
    )
    submitted = st.form_submit_button("Run Full Analysis")

if submitted:
    if not query.strip():
        st.warning("Please enter a house description.")
    else:
        with st.spinner("Running Stage 1 extraction..."):
            try:
                stage1_data = _post_json(STAGE1_EXTRACT_ENDPOINT, {"query": query.strip()})
                st.session_state["stage1_data"] = stage1_data
                candidates = stage1_data.get("candidates", []) if isinstance(stage1_data, dict) else []
                if isinstance(candidates, list) and candidates:
                    st.session_state["chosen_candidate_index"] = _pick_default_candidate_index(candidates)
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"Stage 1 failed: {exc}")

stage1_data = st.session_state.get("stage1_data")
if isinstance(stage1_data, dict):
    candidates = stage1_data.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        st.error("No Stage 1 candidates were returned.")
    else:
        st.subheader("Which set of features suits your property the most?")
        default_idx = _pick_default_candidate_index(candidates)
        chosen_index = int(st.session_state.get("chosen_candidate_index", default_idx))
        if chosen_index < 0 or chosen_index >= len(candidates):
            chosen_index = default_idx
            st.session_state["chosen_candidate_index"] = chosen_index

        selected_candidate = candidates[chosen_index]

        selected_output = selected_candidate.get("output")
        selected_error = selected_candidate.get("error")

        if selected_error:
            st.warning(f"Selected candidate error: {selected_error}")

        base_features = _canonicalize_features(
            selected_output.get("features") if isinstance(selected_output, dict) else None
        )

        missing_fields = set()
        if isinstance(selected_output, dict):
            for field_name in selected_output.get("missing_fields", []):
                missing_fields.add(_normalize_feature_key(str(field_name)))
        for feature_name, value in base_features.items():
            if value is None:
                missing_fields.add(feature_name)

        st.markdown("### Feature set details")
        comparison_rows: list[dict[str, str]] = []
        candidate_features_list: list[dict[str, Any]] = []
        candidate_completeness: list[float | None] = []

        for candidate in candidates:
            output = candidate.get("output")
            if isinstance(output, dict):
                candidate_features_list.append(_canonicalize_features(output.get("features")))
                candidate_completeness.append(float(output.get("completeness", 0.0) or 0.0))
            else:
                candidate_features_list.append(_canonicalize_features(None))
                candidate_completeness.append(None)

        for feature_name in FEATURE_ORDER:
            row = {"Feature": FEATURE_LABELS.get(feature_name, feature_name)}
            for idx, candidate_features in enumerate(candidate_features_list):
                row[f"Feature Set {idx + 1}"] = _format_feature_value(candidate_features.get(feature_name))
            comparison_rows.append(row)

        st.table(comparison_rows)

        # Keep buttons aligned under each feature-set column by reserving a spacer for the Feature column.
        selection_columns = st.columns([2] + [1] * len(candidates))
        for idx, candidate in enumerate(candidates):
            with selection_columns[idx + 1]:
                completeness = candidate_completeness[idx]
                if completeness is None:
                    st.caption(f"Feature Set {idx + 1}: unavailable")
                else:
                    st.caption(f"Feature Set {idx + 1} completeness: {completeness:.0%}")

                is_selected = idx == chosen_index
                button_text = "Selected" if is_selected else f"Select Feature Set {idx + 1}"
                if st.button(button_text, key=f"select_candidate_{idx}", disabled=is_selected):
                    st.session_state["chosen_candidate_index"] = idx
                    st.rerun()

        if missing_fields:
            st.info("Some features are missing. Please enter them manually before running prediction.")

        manual_values: dict[str, Any] = {}
        with st.form("predict_interpret_form"):
            for feature_name in FEATURE_ORDER:
                if feature_name not in missing_fields:
                    continue

                existing = base_features.get(feature_name)
                feature_label = FEATURE_LABELS.get(feature_name, feature_name)
                if feature_name in FLOAT_FEATURES:
                    manual_values[feature_name] = st.number_input(
                        feature_label,
                        value=float(existing) if existing is not None else 0.0,
                        step=1.0,
                        format="%.2f",
                        key=f"manual_{feature_name}",
                    )
                else:
                    manual_values[feature_name] = st.number_input(
                        feature_label,
                        value=int(existing) if existing is not None else 0,
                        step=1,
                        key=f"manual_{feature_name}",
                    )

            run_prediction = st.form_submit_button("Get Final Price + Explanation")

        if run_prediction:
            # Fill missing fields with user-entered values before calling ML.
            feature_payload = dict(base_features)
            for feature_name in missing_fields:
                feature_payload[feature_name] = manual_values.get(feature_name, feature_payload.get(feature_name))

            unresolved = [name for name in FEATURE_ORDER if feature_payload.get(name) is None]
            if unresolved:
                st.error(f"Missing required features: {', '.join(unresolved)}")
            else:
                if missing_fields:
                    # Keep submission pending until user confirms manually-entered values.
                    st.session_state["pending_feature_payload"] = feature_payload
                    st.session_state["pending_missing_fields"] = sorted(missing_fields)
                else:
                    with st.spinner("Running ML prediction and interpretation..."):
                        try:
                            prediction, interpretation = _run_ml_and_interpret(feature_payload)
                        except (requests.RequestException, RuntimeError) as exc:
                            st.error(f"Pipeline failed: {exc}")
                        else:
                            _render_final_result(prediction, interpretation)

        pending_payload = st.session_state.get("pending_feature_payload")
        if isinstance(pending_payload, dict):
            pending_missing = st.session_state.get("pending_missing_fields", [])
            st.warning("Please confirm your manually entered values before submitting to ML.")
            if pending_missing:
                st.write("You manually provided:")
                for feature_name in pending_missing:
                    label = FEATURE_LABELS.get(feature_name, feature_name)
                    value = pending_payload.get(feature_name)
                    st.write(f"- {label}: {_format_feature_value(value)}")

            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Confirm and Submit", key="confirm_manual_submission"):
                    with st.spinner("Running ML prediction and interpretation..."):
                        try:
                            prediction, interpretation = _run_ml_and_interpret(pending_payload)
                        except (requests.RequestException, RuntimeError) as exc:
                            st.error(f"Pipeline failed: {exc}")
                        else:
                            _render_final_result(prediction, interpretation)
                    st.session_state.pop("pending_feature_payload", None)
                    st.session_state.pop("pending_missing_fields", None)

            with cancel_col:
                if st.button("Cancel", key="cancel_manual_submission"):
                    st.session_state.pop("pending_feature_payload", None)
                    st.session_state.pop("pending_missing_fields", None)
                    st.info("Submission canceled. You can edit values and submit again.")
