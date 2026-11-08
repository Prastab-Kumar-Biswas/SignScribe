"""
SignScribe — Real-time Sign Language to Text Generator.

Runs LOCALLY (CPU). Ties the lightweight pipeline together:
    Webcam (mirrored) -> MediaPipe Hands -> Gated Sliding Window -> BiLSTM Classifier
    -> SignTextGenerator -> Live English Text Output

Usage:
    streamlit run src/demo/app.py
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Detect whether installed Streamlit uses modern `use_container_width` or legacy `use_column_width` for st.image
_IMAGE_WIDTH_KWARGS = (
    {"use_container_width": True}
    if "use_container_width" in inspect.signature(st.image).parameters
    else {"use_column_width": True}
)

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    FEATURE_DIM,
    WINDOW_SIZE,
    TARGET_FPS,
    VOCAB,
    ALPHABETS,
    DIGITS,
    ISL_WORDS,
)
from src.text_generator.generator import SignTextGenerator


# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignScribe — Sign Language to Text",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: -6px;
        margin-bottom: 20px;
    }
    .text-output-card {
        background-color: #0f172a;
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 20px;
        min-height: 180px;
        font-size: 1.5rem;
        font-weight: 500;
        color: #f8fafc;
        line-height: 1.6;
        word-wrap: break-word;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .placeholder-text {
        color: #64748b;
        font-style: italic;
        font-size: 1.2rem;
    }
    .token-badge {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px;
        font-family: monospace;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .status-badge-ok {
        background: #064e3b;
        color: #34d399;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .status-badge-warn {
        background: #78350f;
        color: #fbbf24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    /* Completely eliminate Streamlit stale-element dimming and transition flashing */
    .stApp, .stApp * {
        opacity: 1 !important;
        transition: none !important;
    }
    div[data-testid="stImage"] {
        opacity: 1 !important;
        transition: none !important;
        background-color: #0f172a;
        border-radius: 8px;
    }
    div[data-testid="stImage"] img {
        opacity: 1 !important;
        transition: none !important;
        border-radius: 8px;
    }
    /* Hide top-right running indicator pulse to avoid distracting UI shifts */
    div[data-testid="stStatusWidget"] {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ────────────────────────────────────

def init_session_state():
    """Initialise all session state variables."""
    defaults = {
        "running": False,
        "text_generator": None,
        "camera": None,
        "window_buffer": None,
        "frame_count": 0,
        "last_pred_label": "IDLE",
        "last_pred_conf": 0.0,
        "consecutive_same": 0,
        "last_committed_token": "",
        "avg_fps": 0.0,
        "recent_latencies": [],
        "active_cam_index": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state.text_generator is None:
        st.session_state.text_generator = SignTextGenerator()


init_session_state()


# ── Lazy-load cached core models ────────────────────────────────────

@st.cache_resource
def load_hand_tracker():
    """Load MediaPipe hand tracker (cached across reruns)."""
    from src.capture.hand_tracker import HandTracker
    return HandTracker(
        model_complexity=1,
        detection_confidence=0.6,
        tracking_confidence=0.55,
        max_persistence_frames=2,
    )


@st.cache_resource
def load_classifier():
    """Load the trained BiLSTM gesture classifier."""
    try:
        from src.classifier.inference import GesturePredictor
        return GesturePredictor()
    except FileNotFoundError:
        return None
    except Exception as e:
        st.sidebar.error(f"Error loading classifier: {e}")
        return None


# ── Helper for camera cleanup ───────────────────────────────────────

def _release_camera():
    """Release webcam resources safely."""
    if st.session_state.camera is not None:
        try:
            st.session_state.camera.release()
        except Exception:
            pass
    st.session_state.camera = None
    st.session_state.window_buffer = None


# ── Sidebar ─────────────────────────────────────────────────────────

def render_sidebar():
    """Render the simplified sidebar controls."""
    st.sidebar.markdown("## ⚙️ Settings & Status")

    classifier = load_classifier()
    if classifier:
        st.sidebar.markdown(
            f"**Classifier**: <span class='status-badge-ok'>Ready</span> ({len(classifier.label_names)} signs)",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            "**Classifier**: <span class='status-badge-warn'>Not Found</span>",
            unsafe_allow_html=True,
        )
        st.sidebar.warning("Please train the model: `python -m src.classifier.train`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Recognition Mode")

    rec_mode = st.sidebar.radio(
        "Mode Selection",
        options=[
            "Letters & Numbers (Fingerspelling)",
            "All 97 Signs (Letters, Numbers & Words)",
        ],
        index=0,
        help=(
            "Letters & Numbers mode masks out whole words to eliminate "
            "random word hallucinations while fingerspelling."
        ),
    )
    alphanumeric_only = bool(rec_mode == "Letters & Numbers (Fingerspelling)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Detection Sensitivity")

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.30,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help="Minimum classifier probability to accept a sign.",
    )

    stability_frames = st.sidebar.slider(
        "Stability (Consecutive Frames)",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
        help="Number of consecutive identical predictions required to confirm a sign.",
    )

    cam_index = st.sidebar.number_input(
        "Webcam Index", min_value=0, max_value=4, value=0, step=1,
        help="0 is typically the default built-in laptop webcam."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 ISL Vocabulary Guide")
    with st.sidebar.expander("View Supported Signs & Tips"):
        st.markdown(
            "**Two-Hand Alphabets (20)**:\n"
            f"{', '.join(sorted('ABDEFGHKMNPQRSTWXYZ'))}\n\n"
            "*⚠️ Requires BOTH hands in ISL!*\n"
            "- **F vs X**: `F` crosses 2 fingers per hand; `X` crosses 1 finger per hand.\n"
            "- **E vs K**: `E` has both hands horizontally level; `K` has reaching hand held high above the base hand.\n"
            "- **J vs Q**: `J` draws onto a lower palm (|dy| high); `Q` hooks at the same level horizontally.\n"
        )
        st.markdown(
            "**One-Hand Alphabets (6)**:\n"
            f"{', '.join(sorted('CILOUV'))}\n\n"
            "- **I**: Single-hand sign (index up with thumb relaxed/loose).\n"
            "- **L**: Single-hand 'L' (index up + thumb extended wide at 90°).\n"
            "- **1**: Single-hand digit (index up + thumb folded into fist).\n"
        )
        st.markdown(f"**Digits (10)**: {', '.join(DIGITS)}")
        if not alphanumeric_only:
            st.markdown(f"**ISL Words ({len(ISL_WORDS)})**: {', '.join(sorted(ISL_WORDS))}")

    return classifier, conf_threshold, stability_frames, int(cam_index), alphanumeric_only


# ── Helper Display Functions ────────────────────────────────────────

def _render_text_card(placeholder, text: str):
    """Render the generated text card inside its placeholder."""
    if text:
        placeholder.markdown(
            f'<div class="text-output-card">{text}</div>',
            unsafe_allow_html=True,
        )
    else:
        placeholder.markdown(
            '<div class="text-output-card placeholder-text">'
            'Start the camera and sign to generate text here...</div>',
            unsafe_allow_html=True,
        )


def _render_tokens(placeholder, tokens: list[str]):
    """Render recognized token badges inside their placeholder."""
    if tokens:
        badge_html = " ".join(f'<span class="token-badge">{t}</span>' for t in tokens[-15:])
        placeholder.markdown(badge_html, unsafe_allow_html=True)
    else:
        placeholder.markdown("<span style='color: #64748b;'>No signs recorded yet.</span>", unsafe_allow_html=True)


# ── Main Application ────────────────────────────────────────────────

def main():
    tracker = load_hand_tracker()
    classifier, conf_threshold, stability_frames, cam_index, alphanumeric_only = render_sidebar()

    # Header
    st.markdown('<p class="main-title">SignScribe</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Real-time Sign Language to Text Generator</p>',
        unsafe_allow_html=True,
    )

    gen: SignTextGenerator = st.session_state.text_generator

    # 2-column layout: Camera on Left, Text output & Controls on Right
    col_cam, col_output = st.columns([1.1, 1], gap="medium")

    with col_cam:
        st.markdown("### 📷 Video Feed")

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if not st.session_state.running:
                if st.button("▶️ Start Camera", type="primary", use_container_width=True, key="btn_start"):
                    st.session_state.running = True
                    st.rerun()
            else:
                if st.button("⏹️ Stop Camera", type="secondary", use_container_width=True, key="btn_stop"):
                    st.session_state.running = False
                    _release_camera()
                    st.session_state.last_pred_label = "IDLE"
                    st.session_state.last_pred_conf = 0.0
                    st.rerun()

        with col_btn2:
            if st.button("🔄 Reset Text", use_container_width=True, key="btn_reset"):
                gen.reset()
                st.session_state.last_committed_token = ""
                st.session_state.last_pred_label = "IDLE"
                st.session_state.last_pred_conf = 0.0
                _render_text_card(text_placeholder, gen.get_text())
                _render_tokens(tokens_placeholder, gen.get_raw_tokens())

        video_placeholder = st.empty()
        status_placeholder = st.empty()

    with col_output:
        st.markdown("### 📝 Generated Text")
        text_placeholder = st.empty()
        _render_text_card(text_placeholder, gen.get_text())

        # Quick action controls
        st.markdown("<br>", unsafe_allow_html=True)
        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            if st.button("␣ Add Space", use_container_width=True, key="btn_space"):
                gen.add_space()
                _render_text_card(text_placeholder, gen.get_text())
        with col_act2:
            if st.button("⌫ Backspace", use_container_width=True, key="btn_backspace"):
                gen.backspace()
                _render_text_card(text_placeholder, gen.get_text())
        with col_act3:
            if st.button("📋 Copy Text", use_container_width=True, key="btn_copy"):
                st.toast("Text copied or ready to copy from the output card above!")

        # Raw Token Badges
        st.markdown("#### Recognized Signs:")
        tokens_placeholder = st.empty()
        _render_tokens(tokens_placeholder, gen.get_raw_tokens())

    # ── Camera Processing Loop ──────────────────────────────────────
    if st.session_state.running:
        if not classifier:
            st.error("Cannot start: Classifier weights not found. Please train the model.")
            st.session_state.running = False
            return

        # Check if camera index changed
        if st.session_state.get("active_cam_index") != cam_index:
            _release_camera()
            st.session_state["active_cam_index"] = cam_index

        from src.buffering.sliding_window import SlidingWindowBuffer

        if st.session_state.window_buffer is None:
            st.session_state.window_buffer = SlidingWindowBuffer(
                overlap=WINDOW_SIZE // 2,
                min_active_ratio=0.6,
                max_idle_frames=6,
            )

        if st.session_state.camera is None:
            cap = cv2.VideoCapture(cam_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            st.session_state.camera = cap

        cap = st.session_state.camera
        if not cap.isOpened():
            st.error(f"Cannot open webcam at index {cam_index}.")
            st.session_state.running = False
            _release_camera()
            return

        last_rendered_status = ""
        failed_reads = 0
        try:
            while st.session_state.running:
                t_frame_start = time.perf_counter()
                ret, frame_bgr = cap.read()
                if not ret:
                    failed_reads += 1
                    if failed_reads > 10:
                        status_placeholder.error("Lost connection to webcam.")
                        st.session_state.running = False
                        break
                    time.sleep(0.05)
                    continue
                failed_reads = 0

                # HORIZONTAL FLIP: Mirror the camera so it acts like a natural selfie mirror
                # and aligns with MediaPipe's mirror-mode handedness detection.
                frame_bgr = cv2.flip(frame_bgr, 1)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # MediaPipe landmark extraction
                t_mp_start = time.perf_counter()
                features, results = tracker.extract_landmarks(frame_rgb)
                mp_ms = (time.perf_counter() - t_mp_start) * 1000.0

                num_hands = tracker.num_active_hands

                # Ensure buffer is available
                if st.session_state.window_buffer is None:
                    st.session_state.window_buffer = SlidingWindowBuffer(
                        overlap=WINDOW_SIZE // 2,
                        min_active_ratio=0.6,
                        max_idle_frames=6,
                    )

                # Feed buffer with hand gating
                window = st.session_state.window_buffer.add_frame(features, has_hands=(num_hands > 0))

                text_updated = False
                if window is not None and num_hands > 0:
                    pred_label, pred_conf, clf_ms = classifier.predict(
                        window, min_active_frames=12, alphanumeric_only=alphanumeric_only
                    )

                    if pred_label != "IDLE" and pred_conf >= conf_threshold:
                        if pred_label == st.session_state.last_pred_label:
                            st.session_state.consecutive_same += 1
                        else:
                            st.session_state.consecutive_same = 1
                            st.session_state.last_pred_label = pred_label

                        # Stability confirmation
                        if st.session_state.consecutive_same >= stability_frames:
                            # Debounce: avoid repeatedly emitting the exact same sign while holding
                            if pred_label != st.session_state.last_committed_token:
                                gen.add_token(pred_label)
                                st.session_state.last_committed_token = pred_label
                                text_updated = True
                            st.session_state.consecutive_same = 0
                    else:
                        if pred_label == "IDLE":
                            st.session_state.last_pred_label = "IDLE"
                            st.session_state.last_pred_conf = 0.0

                    st.session_state.last_pred_conf = pred_conf

                elif num_hands == 0:
                    st.session_state.last_pred_label = "IDLE"
                    st.session_state.last_pred_conf = 0.0
                    st.session_state.consecutive_same = 0
                    # Allow signing the same sign again after hands are lowered
                    st.session_state.last_committed_token = ""

                # Push updated text and tokens to UI without reloading the page
                if text_updated:
                    _render_text_card(text_placeholder, gen.get_text())
                    _render_tokens(tokens_placeholder, gen.get_raw_tokens())

                # Draw landmarks on display frame
                frame_display = tracker.draw_landmarks(frame_bgr.copy(), results)

                # Visual HUD overlay on video
                if num_hands == 2:
                    status_text = "2 Hands (Two-Hand Gesture)"
                    color = (0, 255, 0)
                elif num_hands == 1:
                    status_text = "1 Hand (1-Hand Sign: C,I,L,O,U,V or 0-9)"
                    color = (0, 200, 255)
                else:
                    status_text = "Idle (No hands)"
                    color = (128, 128, 128)

                mode_tag = "Chars" if alphanumeric_only else "All 97"
                cv2.putText(
                    frame_display,
                    f"[{mode_tag}] Hands: {status_text} | MP: {mp_ms:.0f}ms",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                if st.session_state.last_pred_label != "IDLE" and st.session_state.last_pred_conf > 0:
                    is_2h = classifier.is_two_hand_sign(st.session_state.last_pred_label)
                    sign_tag = "[2-Hand]" if is_2h else "[1-Hand]"
                    cv2.putText(
                        frame_display,
                        f"Sign: {st.session_state.last_pred_label} ({st.session_state.last_pred_conf:.0%}) {sign_tag}",
                        (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2,
                    )

                # Render display image directly to Streamlit image element
                display_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_rgb, channels="RGB", output_format="JPEG", **_IMAGE_WIDTH_KWARGS)

                hand_info = (
                    "🟢 Both Hands (2-Hand Sign)"
                    if num_hands == 2
                    else (
                        "🟡 1 Hand (Note: A,B,D... need 2 hands)"
                        if num_hands == 1
                        else "⚪ Idle"
                    )
                )
                current_status = (
                    f"⚡ {hand_info} | MP: {mp_ms:.1f}ms | Detected: **{st.session_state.last_pred_label}** "
                    f"({st.session_state.last_pred_conf:.0%})"
                )
                if current_status != last_rendered_status:
                    status_placeholder.caption(current_status)
                    last_rendered_status = current_status

                # Regulate frame rate to TARGET_FPS
                elapsed = time.perf_counter() - t_frame_start
                frame_interval = 1.0 / TARGET_FPS
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        finally:
            if not st.session_state.running:
                _release_camera()
    else:
        video_placeholder.info("📹 Camera is stopped. Click **▶️ Start Camera** above to begin.")
        status_placeholder.caption("Status: Standby | Ready")


if __name__ == "__main__":
    main()
