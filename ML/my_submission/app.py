#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import tempfile
from contextlib import suppress
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from solution import CURRENT_ANOMALY_PIPELINE, DocFusionSolution

FIELD_ORDER = ("vendor", "date", "total")
FIELD_COLORS = {
    "vendor": "#1D4ED8",
    "date": "#7C3AED",
    "total": "#C2410C",
}
FIELD_LABELS = {
    "vendor": "Vendor",
    "date": "Date",
    "total": "Total",
}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #132238;
            --muted: #5b6678;
            --paper: #f7f4ec;
            --panel: rgba(255, 252, 245, 0.92);
            --line: rgba(19, 34, 56, 0.10);
            --blue: #2051d1;
            --amber: #cb5b19;
            --green: #1e7a46;
            --red: #b42318;
            --mono: "SFMono-Regular", "IBM Plex Mono", Menlo, monospace;
            --sans: "Avenir Next", "Trebuchet MS", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 210, 156, 0.35), transparent 30%),
                radial-gradient(circle at top right, rgba(109, 161, 255, 0.18), transparent 28%),
                linear-gradient(180deg, #f4efe3 0%, #ece4d3 100%);
            color: var(--ink);
            font-family: var(--sans);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }

        h1, h2, h3, h4 {
            color: var(--ink);
            font-family: var(--sans);
            letter-spacing: -0.02em;
        }

        section[data-testid="stSidebar"] {
            background: rgba(17, 24, 39, 0.94);
            color: #f3f4f6;
        }

        section[data-testid="stSidebar"] * {
            color: #f3f4f6;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255, 252, 245, 0.98), rgba(245, 236, 219, 0.92));
            border: 1px solid rgba(19, 34, 56, 0.08);
            border-radius: 26px;
            padding: 1.6rem 1.7rem;
            box-shadow: 0 18px 50px rgba(77, 59, 30, 0.10);
            margin-bottom: 1.2rem;
        }

        .eyebrow {
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--amber);
            margin-bottom: 0.45rem;
            font-weight: 700;
        }

        .hero-title {
            font-size: 2.2rem;
            line-height: 1.02;
            margin: 0;
            font-weight: 700;
        }

        .hero-copy {
            color: var(--muted);
            margin-top: 0.6rem;
            max-width: 48rem;
            font-size: 1rem;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 12px 32px rgba(77, 59, 30, 0.08);
        }

        .status-card {
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            color: #fffdf7;
            box-shadow: 0 18px 42px rgba(19, 34, 56, 0.16);
            margin-bottom: 1rem;
        }

        .status-card.safe {
            background: linear-gradient(135deg, #146c43, #2a8f63);
        }

        .status-card.watch {
            background: linear-gradient(135deg, #8a4b0b, #d97706);
        }

        .status-card.alert {
            background: linear-gradient(135deg, #7a1521, #c2410c);
        }

        .status-kicker {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.76rem;
            opacity: 0.88;
            font-weight: 700;
        }

        .status-title {
            font-size: 1.7rem;
            line-height: 1.05;
            margin-top: 0.35rem;
            font-weight: 700;
        }

        .status-subcopy {
            margin-top: 0.5rem;
            font-size: 0.96rem;
            color: rgba(255, 250, 242, 0.92);
        }

        .meter-shell {
            margin-top: 0.95rem;
            background: rgba(255, 248, 236, 0.22);
            height: 14px;
            border-radius: 999px;
            overflow: hidden;
        }

        .meter-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(255,255,255,0.85), rgba(255, 213, 160, 0.95));
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.8rem 0 1.1rem;
        }

        .metric-card {
            background: rgba(255, 252, 245, 0.92);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 0.95rem;
        }

        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            margin-bottom: 0.45rem;
            font-weight: 700;
        }

        .metric-value {
            font-size: 1.42rem;
            line-height: 1;
            color: var(--ink);
            font-weight: 700;
        }

        .metric-note {
            margin-top: 0.45rem;
            font-size: 0.9rem;
            color: var(--muted);
        }

        .field-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .field-card {
            border-radius: 20px;
            padding: 1rem 1rem 0.95rem;
            border: 1px solid var(--line);
            background: rgba(255, 252, 245, 0.96);
            min-height: 148px;
        }

        .field-card.flagged {
            border-color: rgba(203, 91, 25, 0.45);
            background: linear-gradient(180deg, rgba(255, 240, 230, 0.96), rgba(255, 251, 245, 0.96));
            box-shadow: inset 0 0 0 1px rgba(203, 91, 25, 0.10);
        }

        .field-card.clean {
            border-color: rgba(32, 81, 209, 0.22);
            background: linear-gradient(180deg, rgba(241, 246, 255, 0.96), rgba(255, 252, 245, 0.96));
        }

        .field-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            font-weight: 700;
        }

        .field-value {
            font-size: 1.1rem;
            line-height: 1.25;
            margin-top: 0.55rem;
            color: var(--ink);
            font-weight: 700;
            word-break: break-word;
        }

        .field-value.missing {
            color: #8b5e3c;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.9rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.24rem 0.62rem;
            font-size: 0.78rem;
            line-height: 1;
            font-weight: 700;
            border: 1px solid transparent;
        }

        .pill.clean {
            color: var(--blue);
            background: rgba(32, 81, 209, 0.10);
            border-color: rgba(32, 81, 209, 0.14);
        }

        .pill.flagged {
            color: var(--amber);
            background: rgba(203, 91, 25, 0.10);
            border-color: rgba(203, 91, 25, 0.16);
        }

        .signal-list {
            display: grid;
            gap: 0.65rem;
        }

        .signal-item {
            border-radius: 16px;
            border: 1px solid rgba(19, 34, 56, 0.08);
            background: rgba(255, 252, 245, 0.96);
            padding: 0.78rem 0.9rem;
        }

        .signal-item strong {
            color: var(--ink);
        }

        .legend {
            margin-top: 0.75rem;
            color: var(--muted);
            font-size: 0.9rem;
        }

        .legend-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            margin-right: 0.4rem;
            transform: translateY(1px);
        }

        .json-shell {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(19, 34, 56, 0.08);
        }

        .code-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.6rem;
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            margin-bottom: 0.8rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.25rem 0.8rem;
            background: rgba(255, 252, 245, 0.55);
            border: 1px solid rgba(19, 34, 56, 0.08);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(32, 81, 209, 0.10);
            color: var(--blue);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _default_model_dir() -> str:
    base_dir = Path(__file__).resolve().parent
    trained = (base_dir / "models/latest/model").resolve()
    if trained.exists():
        return str(trained)
    return str((base_dir / "models/latest").resolve())


def _validate_latest_model_dir(model_dir: str) -> dict[str, object]:
    model_path = Path(model_dir) / "model.json"
    if not model_path.exists():
        raise RuntimeError(f"Missing model.json in {model_dir}. Train the latest model first.")

    with model_path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)

    pipeline = model.get("anomaly_pipeline")
    if pipeline != CURRENT_ANOMALY_PIPELINE:
        raise RuntimeError(
            f"Latest model is incompatible: expected {CURRENT_ANOMALY_PIPELINE}, got {pipeline!r}."
        )
    if not model.get("ml_enabled", False):
        raise RuntimeError("Latest model does not have the feature-based forgery classifier enabled.")
    if not (Path(model_dir) / "anomaly_model.pkl").exists():
        raise RuntimeError(f"Missing anomaly_model.pkl in {model_dir}.")
    if not (Path(model_dir) / "outlier_model.pkl").exists():
        raise RuntimeError(f"Missing outlier_model.pkl in {model_dir}.")
    return model


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _pill(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{_escape(text)}</span>'


def _metric_card(label: str, value: str, note: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{_escape(label)}</div>'
        f'<div class="metric-value">{_escape(value)}</div>'
        f'<div class="metric-note">{_escape(note)}</div>'
        '</div>'
    )


def _field_card(field: str, meta: dict[str, object]) -> str:
    title = FIELD_LABELS[field]
    value = meta.get("value") or "Not extracted"
    flagged = bool(meta.get("flagged"))
    located = bool(meta.get("located"))
    notes = meta.get("notes", [])
    note_html = "".join(_pill(str(note), "flagged" if flagged else "clean") for note in notes if str(note).strip())
    if not note_html:
        note_html = _pill("No field-specific concerns", "clean")
    state_class = "flagged" if flagged else "clean"
    value_class = "field-value missing" if value == "Not extracted" else "field-value"
    location_text = "Located on receipt" if located else "No box found"
    location_kind = "clean" if located else "flagged"
    return (
        f'<div class="field-card {state_class}">'
        f'<div class="field-label">{_escape(title)}</div>'
        f'<div class="{value_class}">{_escape(value)}</div>'
        f'<div class="pill-row">{note_html}{_pill(location_text, location_kind)}</div>'
        '</div>'
    )


def _status_payload(score: float, threshold: float, is_forged: int) -> dict[str, str | float]:
    if is_forged == 1 and score >= max(threshold + 0.20, 0.70):
        return {
            "css": "alert",
            "kicker": "Escalation",
            "title": "Likely forged receipt",
            "copy": "Multiple signals crossed the review threshold. Inspect the highlighted fields first.",
        }
    if is_forged == 1:
        return {
            "css": "watch",
            "kicker": "Review needed",
            "title": "Suspicious receipt",
            "copy": "The detector is above threshold, but the evidence is moderate rather than overwhelming.",
        }
    if score >= max(threshold * 0.7, threshold - 0.08):
        return {
            "css": "watch",
            "kicker": "Borderline",
            "title": "Likely genuine, but worth a quick check",
            "copy": "The receipt stayed below the decision threshold, but it still carries some suspicious signals.",
        }
    return {
        "css": "safe",
        "kicker": "Clear",
        "title": "Likely genuine receipt",
        "copy": "No strong anomaly pattern crossed the decision threshold for this document.",
    }


def _field_review(analysis: dict[str, object], debug: dict[str, object]) -> dict[str, dict[str, object]]:
    features = debug.get("features", {}) if isinstance(debug, dict) else {}
    features = features if isinstance(features, dict) else {}
    boxes = debug.get("boxes", {}) if isinstance(debug, dict) else {}
    boxes = boxes if isinstance(boxes, dict) else {}

    review: dict[str, dict[str, object]] = {}
    for field in FIELD_ORDER:
        value = analysis.get(field)
        notes: list[str] = []
        flagged = False

        if field == "vendor":
            if float(features.get("missing_vendor", 0.0)) >= 0.5:
                flagged = True
                notes.append("Missing vendor")
            elif value:
                notes.append("Extracted cleanly")
        elif field == "date":
            if float(features.get("missing_date", 0.0)) >= 0.5:
                flagged = True
                notes.append("Missing date")
            elif float(features.get("invalid_date", 0.0)) >= 0.5:
                flagged = True
                notes.append("Date format looks invalid")
            elif value:
                notes.append("Date parsed")
        elif field == "total":
            if float(features.get("missing_total", 0.0)) >= 0.5:
                flagged = True
                notes.append("Missing total")
            if float(features.get("math_available", 0.0)) >= 0.5:
                if float(features.get("math_consistent", 1.0)) < 0.5:
                    flagged = True
                    notes.append("Math check failed")
                else:
                    notes.append("Math check passed")
            if float(features.get("iso_outlier", 0.0)) >= 0.5:
                flagged = True
                notes.append("Total looks like a numeric outlier")
            if value and not notes:
                notes.append("Extracted cleanly")

        if not value and not notes:
            notes.append("Not extracted")

        review[field] = {
            "value": value,
            "flagged": flagged,
            "notes": notes,
            "located": isinstance(boxes.get(field), dict),
        }
    return review


def draw_boxes(
    image: Image.Image,
    boxes: dict[str, dict[str, int]],
    field_review: dict[str, dict[str, object]],
) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)

    for field in FIELD_ORDER:
        box = boxes.get(field)
        if not isinstance(box, dict):
            continue
        flagged = bool(field_review.get(field, {}).get("flagged", False))
        base_color = FIELD_COLORS.get(field, "#0F766E")
        color = "#D97706" if flagged else base_color
        left = int(box.get("left", 0))
        top = int(box.get("top", 0))
        right = int(box.get("right", 0))
        bottom = int(box.get("bottom", 0))
        draw.rectangle([left, top, right, bottom], outline=color, width=4)
        tag_text = f"{FIELD_LABELS[field].upper()} {'REVIEW' if flagged else 'OK'}"
        tag_y = max(0, top - 20)
        draw.rectangle([left, tag_y, min(rendered.width, left + 170), top], fill=color)
        draw.text((left + 6, tag_y + 3), tag_text, fill="#FFF9ED")
    return rendered


def _quality_snapshot(debug: dict[str, object], field_review: dict[str, dict[str, object]]) -> dict[str, str]:
    features = debug.get("features", {}) if isinstance(debug, dict) else {}
    features = features if isinstance(features, dict) else {}
    extracted_count = sum(1 for field in FIELD_ORDER if field_review[field].get("value"))
    flagged_count = sum(1 for field in FIELD_ORDER if bool(field_review[field].get("flagged")))

    if float(features.get("math_available", 0.0)) < 0.5:
        math_note = "Math check unavailable"
    elif float(features.get("math_consistent", 1.0)) < 0.5:
        math_note = "Totals disagree with common receipt math"
    else:
        math_note = "Totals are internally consistent"

    if float(features.get("iso_outlier", 0.0)) >= 0.5:
        numeric_note = "Numeric pattern looks unusual"
    else:
        numeric_note = "Numeric profile stays near the training baseline"

    if float(features.get("ela_high_ratio", 0.0)) >= 0.08 or float(features.get("ela_max", 0.0)) >= 0.25:
        image_note = "Recompression signal looks elevated"
    else:
        image_note = "Image forensics look ordinary"

    return {
        "extracted": f"{extracted_count}/3",
        "flagged": str(flagged_count),
        "math": math_note,
        "numeric": numeric_note,
        "image": image_note,
    }


def _render_signals(reasons: list[str]) -> None:
    if not reasons:
        st.markdown(
            '<div class="signal-item"><strong>No explicit signals fired.</strong><br/>The document stayed below the rule-based reason thresholds.</div>',
            unsafe_allow_html=True,
        )
        return

    items = []
    for idx, reason in enumerate(reasons, start=1):
        items.append(
            '<div class="signal-item">'
            f'<strong>Signal {idx}</strong><br/>{_escape(reason)}'
            '</div>'
        )
    st.markdown(f'<div class="signal-list">{"".join(items)}</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="DocFusion Review Desk", layout="wide")
    _inject_css()

    st.session_state["model_dir"] = _default_model_dir()
    active_model_dir = str(st.session_state["model_dir"])

    with st.sidebar:
        st.subheader("Review Session")
        try:
            model_meta = _validate_latest_model_dir(active_model_dir)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        st.caption(f"Model path: `{active_model_dir}`")
        st.caption(
            f"Pipeline: `{model_meta.get('anomaly_pipeline')}`\n"
            f"Threshold: `{float(model_meta.get('ml_threshold', 0.0)):.2%}`\n"
            f"Variant: `{model_meta.get('anomaly_variant', 'unknown')}`"
        )
        st.markdown("---")
        st.markdown(
            "**How to read this screen**\n"
            "- Orange boxes mean the detector sees risk around that extracted field.\n"
            "- Blue and purple boxes mark extracted fields that look cleaner.\n"
            "- The score is the model's suspicious probability before thresholding."
        )

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Level 3B / Human Review Desk</div>
          <div class="hero-title">Inspect one receipt, one decision, one evidence trail.</div>
          <div class="hero-copy">
            Upload a receipt image to review extraction, anomaly status, and the exact fields the current model wants a human to check.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploader_panel, note_panel = st.columns([1.25, 0.75])
    with uploader_panel:
        uploaded = st.file_uploader(
            "Receipt image",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            label_visibility="collapsed",
        )
    with note_panel:
        st.markdown(
            '<div class="panel"><div class="code-label">Scope</div>'
            'This UI is intentionally simple: upload a receipt, inspect extraction, review anomaly evidence, and compare the score against the live model threshold.'
            '</div>',
            unsafe_allow_html=True,
        )

    if uploaded is None:
        st.markdown(
            '<div class="panel"><div class="code-label">Waiting for input</div>'
            'Drop in a receipt image to render the review layout. The app will use the latest on-disk model only, not any cached legacy artifact.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    image_bytes = uploaded.getvalue()
    suffix = Path(uploaded.name or "receipt.png").suffix or ".png"
    temp_image_path: Path | None = None
    solution = DocFusionSolution()

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            temp_image_path = Path(tmp.name)

        analysis = solution.analyze_image(
            model_dir=active_model_dir,
            image_path=str(temp_image_path),
        )
        image = Image.open(temp_image_path).convert("RGB")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return
    finally:
        if temp_image_path is not None:
            with suppress(OSError):
                temp_image_path.unlink()

    debug = analysis.get("_debug", {}) if isinstance(analysis, dict) else {}
    debug = debug if isinstance(debug, dict) else {}
    boxes = debug.get("boxes", {}) if isinstance(debug.get("boxes"), dict) else {}
    reasons = debug.get("reasons", []) if isinstance(debug.get("reasons"), list) else []
    score = float(debug.get("suspicious_score", 0.0))
    threshold = float(debug.get("threshold", 0.5))
    status = _status_payload(score, threshold, int(analysis.get("is_forged", 0)))
    field_review = _field_review(analysis, debug)
    snapshot = _quality_snapshot(debug, field_review)
    rendered = draw_boxes(image, boxes, field_review)

    left, right = st.columns([1.15, 0.95], gap="large")

    with left:
        st.subheader("Receipt review")
        st.image(rendered, use_container_width=True)
        st.markdown(
            '<div class="legend">'
            '<span class="legend-dot" style="background:#D97706"></span>Flagged field '
            '<span class="legend-dot" style="background:#1D4ED8"></span>Vendor '
            '<span class="legend-dot" style="background:#7C3AED"></span>Date '
            '<span class="legend-dot" style="background:#C2410C"></span>Total'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f'''<div class="status-card {status["css"]}">
                <div class="status-kicker">{_escape(status["kicker"])}</div>
                <div class="status-title">{_escape(status["title"])}</div>
                <div class="status-subcopy">{_escape(status["copy"])}<br/>Suspicious score: {score:.2%} | Threshold: {threshold:.2%}</div>
                <div class="meter-shell"><div class="meter-fill" style="width:{max(2.0, min(100.0, score * 100.0)):.1f}%"></div></div>
            </div>''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-grid">'
            + _metric_card("Fields extracted", snapshot["extracted"], "Mandatory fields found on this document")
            + _metric_card("Flagged fields", snapshot["flagged"], "Field-level areas the UI is pushing to review")
            + _metric_card("Decision", "FORGED" if int(analysis.get("is_forged", 0)) == 1 else "GENUINE", "Binary output returned by the live model")
            + '</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Field review")
        st.markdown(
            '<div class="field-grid">'
            + ''.join(_field_card(field, field_review[field]) for field in FIELD_ORDER)
            + '</div>',
            unsafe_allow_html=True,
        )

    tabs = st.tabs(["Evidence", "Extraction", "Debug"])

    with tabs[0]:
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown('<div class="code-label">Primary signals</div>', unsafe_allow_html=True)
            _render_signals([str(reason) for reason in reasons])
        with col_b:
            st.markdown('<div class="code-label">Evidence snapshot</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="signal-list">'
                f'<div class="signal-item"><strong>Math integrity</strong><br/>{_escape(snapshot["math"])}</div>'
                f'<div class="signal-item"><strong>Numeric profile</strong><br/>{_escape(snapshot["numeric"])}</div>'
                f'<div class="signal-item"><strong>Image forensics</strong><br/>{_escape(snapshot["image"])}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    with tabs[1]:
        st.markdown('<div class="code-label">Extracted payload</div>', unsafe_allow_html=True)
        st.json(
            {
                "vendor": analysis.get("vendor"),
                "date": analysis.get("date"),
                "total": analysis.get("total"),
                "is_forged": analysis.get("is_forged"),
            }
        )

    with tabs[2]:
        st.markdown('<div class="code-label">Raw model output</div>', unsafe_allow_html=True)
        st.json(analysis)


if __name__ == "__main__":
    main()
