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

st.set_page_config(page_title="House Price Assistant", page_icon="🏠", layout="wide")
st.title("House Price Assistant")
st.caption("Enter a house description, review extracted features, fill any missing values, then run prediction and interpretation.")

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
    submitted = st.form_submit_button("Run Stage 1 Extraction")

if submitted:
    if not query.strip():
        st.warning("Please enter a house description.")
    else:
        with st.spinner("Running Stage 1 extraction..."):
            try:
                stage1_data = _post_json(STAGE1_EXTRACT_ENDPOINT, {"query": query.strip()})
                st.session_state["stage1_data"] = stage1_data
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"Stage 1 failed: {exc}")

stage1_data = st.session_state.get("stage1_data")
if isinstance(stage1_data, dict):
    candidates = stage1_data.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        st.error("No Stage 1 candidates were returned.")
    else:
        st.subheader("Stage 1 Candidates")
        labels: list[str] = []
        for candidate in candidates:
            prompt_version = candidate.get("prompt_version", "unknown")
            output = candidate.get("output")
            if isinstance(output, dict):
                completeness = float(output.get("completeness", 0.0) or 0.0)
                labels.append(f"{prompt_version} ({completeness:.0%} complete)")
            else:
                labels.append(f"{prompt_version} (failed)")

        default_idx = _pick_default_candidate_index(candidates)
        chosen_label = st.radio(
            "Choose candidate",
            options=labels,
            index=default_idx,
        )
        chosen_index = labels.index(chosen_label)
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

        st.write("Extracted features")
        st.json(base_features)

        if missing_fields:
            st.info("Some features are missing. Please enter them manually before running prediction.")

        manual_values: dict[str, Any] = {}
        with st.form("predict_interpret_form"):
            for feature_name in FEATURE_ORDER:
                if feature_name not in missing_fields:
                    continue

                existing = base_features.get(feature_name)
                if feature_name in FLOAT_FEATURES:
                    manual_values[feature_name] = st.number_input(
                        f"{feature_name}",
                        value=float(existing) if existing is not None else 0.0,
                        step=1.0,
                        format="%.2f",
                        key=f"manual_{feature_name}",
                    )
                else:
                    manual_values[feature_name] = st.number_input(
                        f"{feature_name}",
                        value=int(existing) if existing is not None else 0,
                        step=1,
                        key=f"manual_{feature_name}",
                    )

            run_prediction = st.form_submit_button("Run ML + Interpretation")

        if run_prediction:
            # Fill missing fields with user-entered values before calling ML.
            feature_payload = dict(base_features)
            for feature_name in missing_fields:
                feature_payload[feature_name] = manual_values.get(feature_name, feature_payload.get(feature_name))

            unresolved = [name for name in FEATURE_ORDER if feature_payload.get(name) is None]
            if unresolved:
                st.error(f"Missing required features: {', '.join(unresolved)}")
            else:
                with st.spinner("Running ML prediction and interpretation..."):
                    try:
                        prediction = _post_json(ML_PREDICT_ENDPOINT, feature_payload)
                        interpretation = _post_json(
                            INTERPRET_ENDPOINT,
                            {
                                "features": feature_payload,
                                "prediction": prediction,
                            },
                        )
                    except (requests.RequestException, RuntimeError) as exc:
                        st.error(f"Pipeline failed: {exc}")
                    else:
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
                        st.write(interpretation.get("summary", "No summary available."))

                        key_drivers = interpretation.get("key_drivers", [])
                        if key_drivers:
                            st.write("Key Drivers")
                            for item in key_drivers:
                                st.write(f"- {item}")

                        caveats = interpretation.get("caveats", [])
                        if caveats:
                            st.write("Caveats")
                            for item in caveats:
                                st.write(f"- {item}")

with st.expander("Backend endpoints"):
    st.code(
        "\n".join(
            [
                STAGE1_EXTRACT_ENDPOINT,
                ML_PREDICT_ENDPOINT,
                INTERPRET_ENDPOINT,
            ]
        )
    )
