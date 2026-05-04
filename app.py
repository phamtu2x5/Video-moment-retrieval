from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from momentlens_mvp import (
    IMPROVEMENT_CKPT_PATH,
    SF50_CKPT_PATH,
    QUERY_BACKBONE_NAME,
    load_phase3_runtime,
    load_slowfast_extractor,
    predict_moment,
    render_timeline,
    write_predicted_clip,
)


st.set_page_config(page_title="MomentLens MVP", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    video {
        width: 100% !important;
        max-width: 720px !important;
        max-height: 360px !important;
        height: auto !important;
        object-fit: contain !important;
        display: block !important;
        margin: 0 auto !important;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("MomentLens MVP")
st.caption("Upload a video, type a query, and get the predicted moment span.")
st.info("Máy cùi nên inference hơi lâu, bạn chờ xíu nha <3")

with st.spinner("Loading models..."):
    _ = load_phase3_runtime()
    _ = load_slowfast_extractor()

with st.sidebar:
    st.header("Runtime")
    st.write("Model status: loaded")
    st.write(f"Improvement model: `{IMPROVEMENT_CKPT_PATH.relative_to(Path(__file__).resolve().parent)}`")
    st.write(f"Query backbone: `{QUERY_BACKBONE_NAME}`")
    st.write(f"SlowFast-50 checkpoint: `{SF50_CKPT_PATH.relative_to(Path(__file__).resolve().parent)}`")

st.subheader("1) Input")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"])
query = st.text_input("Query", placeholder="e.g. a person opens the fridge")
status = st.empty()

run_button = st.button("Predict moment", type="primary", use_container_width=True)

if uploaded_video is not None:
    st.video(uploaded_video)

if run_button:
    if uploaded_video is None:
        status.error("Please upload a video first.")
        st.stop()
    if not query.strip():
        status.error("Please enter a query.")
        st.stop()
    status.empty()

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix or ".mp4") as tmp:
        tmp.write(uploaded_video.getbuffer())
        video_path = tmp.name

    try:
        with st.spinner("Running inference..."):
            result = predict_moment(video_path, query.strip())
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as clip_tmp:
            predicted_clip_path = clip_tmp.name
        write_predicted_clip(video_path, result["start_sec"], result["end_sec"], predicted_clip_path)
        predicted_clip_bytes = Path(predicted_clip_path).read_bytes()
        try:
            Path(predicted_clip_path).unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as exc:
        status.error(f"Prediction failed: {exc}")
        st.stop()
    finally:
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted start", f"{result['start_sec']:.2f}s")
    c2.metric("Predicted end", f"{result['end_sec']:.2f}s")
    c3.metric("Duration", f"{result['duration']:.2f}s")
    c4.metric("Clip windows", str(result["num_windows"]))

    st.pyplot(render_timeline(result["duration"], result["start_sec"], result["end_sec"]))

    st.subheader("2) Prediction")
    st.write(
        f"Predicted moment: **[{result['start_sec']:.2f}s, {result['end_sec']:.2f}s]** "
        f"(window indices: {result['start_idx']} -> {result['end_idx']})"
    )
    st.caption(
        "The model is the phase 3 predictor from `Model/Improvement_Ckp/best.pt`. "
        "Feature extraction uses SlowFast-50 on 1-second windows."
    )

    with st.expander("Show raw prediction payload"):
        st.json({
            "query": query.strip(),
            "video_file": uploaded_video.name,
            "prediction": {
                "start_sec": result["start_sec"],
                "end_sec": result["end_sec"],
                "start_idx": result["start_idx"],
                "end_idx": result["end_idx"],
            },
            "feature_shape": result["features_shape"],
            "duration": result["duration"],
            "num_windows": result["num_windows"],
        })

    st.subheader("3) Predicted clip")
    st.image(predicted_clip_bytes)
    st.caption("This clip is cut from the uploaded video and burns in the predicted moment on each frame.")
