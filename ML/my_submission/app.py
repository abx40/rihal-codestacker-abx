#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from solution import DocFusionSolution


def draw_boxes(image: Image.Image, boxes: dict[str, dict[str, int]]) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    colors = {
        "vendor": "#1D4ED8",
        "date": "#9333EA",
        "total": "#DC2626",
    }
    for field, box in boxes.items():
        color = colors.get(field, "#0F766E")
        left = int(box.get("left", 0))
        top = int(box.get("top", 0))
        right = int(box.get("right", 0))
        bottom = int(box.get("bottom", 0))
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left, max(0, top - 14)), field.upper(), fill=color)
    return rendered


def _default_model_dir() -> str:
    trained = Path("./models/latest/model").resolve()
    if trained.exists():
        return str(trained)
    return str(Path("./models/latest").resolve())


def main() -> None:
    st.set_page_config(page_title="DocFusion Dashboard", layout="wide")
    st.title("DocFusion: Receipt Extraction + Anomaly Dashboard")
    st.caption("Upload a receipt image, extract fields, and flag suspicious records.")

    solution = DocFusionSolution()

    # Always use the latest on-disk model to avoid stale session paths.
    st.session_state["model_dir"] = _default_model_dir()
    if "llm_enabled" not in st.session_state:
        st.session_state["llm_enabled"] = False
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = "gpt-4o-mini"

    with st.sidebar:
        st.subheader("Model")
        active_model_dir = str(st.session_state["model_dir"])
        st.caption(f"Using latest model path: `{active_model_dir}`")
        model_json = Path(active_model_dir) / "model.json"
        if not model_json.exists():
            st.warning("No model.json found in latest model path. Run training from terminal first.")

        st.subheader("LLM Explanation")
        st.checkbox(
            "Enable LLM anomaly summary",
            key="llm_enabled",
            help="Requires OPENAI_API_KEY (or DOCFUSION_LLM_API_KEY).",
        )
        st.text_input(
            "LLM Model",
            key="llm_model",
            disabled=not bool(st.session_state.get("llm_enabled", False)),
            help="Any OpenAI-compatible chat model.",
        )

    uploaded = st.file_uploader(
        "Upload receipt image",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
    )
    if uploaded is None:
        st.info("Upload an image to start analysis.")
        return

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(uploaded.read())
        temp_image_path = Path(tmp.name)

    try:
        analysis = solution.analyze_image(
            model_dir=str(st.session_state["model_dir"]),
            image_path=str(temp_image_path),
            include_llm_summary=bool(st.session_state.get("llm_enabled", False)),
            llm_model=str(st.session_state.get("llm_model", "gpt-4o-mini")),
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    debug = analysis.get("_debug", {}) if isinstance(analysis, dict) else {}
    boxes = debug.get("boxes", {}) if isinstance(debug, dict) else {}

    try:
        image = Image.open(temp_image_path)
    except Exception:
        st.error("Could not open uploaded image.")
        return

    col_left, col_right = st.columns([1.2, 1.0])
    with col_left:
        st.subheader("Receipt")
        st.image(draw_boxes(image, boxes), use_container_width=True)

    with col_right:
        st.subheader("Extraction")
        st.write(
            {
                "vendor": analysis.get("vendor"),
                "date": analysis.get("date"),
                "total": analysis.get("total"),
            }
        )

        is_forged = int(analysis.get("is_forged", 0))
        probability = float(debug.get("probability", 0.0)) if isinstance(debug, dict) else 0.0
        status = "SUSPICIOUS" if is_forged == 1 else "LIKELY GENUINE"
        if is_forged == 1:
            st.error(f"Status: {status} ({probability:.2%})")
        else:
            st.success(f"Status: {status} ({probability:.2%})")

        reasons = debug.get("reasons", []) if isinstance(debug, dict) else []
        if reasons:
            st.subheader("Why flagged")
            for reason in reasons:
                st.write(f"- {reason}")

        summary = debug.get("anomaly_summary") if isinstance(debug, dict) else None
        if isinstance(summary, str) and summary.strip():
            source = str(debug.get("summary_source", "heuristic")).upper() if isinstance(debug, dict) else "HEURISTIC"
            st.subheader("Anomaly Summary")
            st.info(summary)
            st.caption(f"Summary source: {source}")

        llm_error = debug.get("llm_error") if isinstance(debug, dict) else None
        if bool(st.session_state.get("llm_enabled", False)) and isinstance(llm_error, str) and llm_error.strip():
            st.warning(f"LLM summary unavailable: {llm_error}")

        with st.expander("Raw debug"):
            st.json(analysis)


if __name__ == "__main__":
    main()
