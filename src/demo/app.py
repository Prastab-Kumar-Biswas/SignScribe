"""
M7 -- Streamlit demo application for SignScribe.

Runs LOCALLY.  Ties the full pipeline together:
    Webcam -> MediaPipe -> Sliding Window -> BiLSTM Classifier
    -> Path A (Rule-based) + Path B (LLM) side by side

Usage:
    streamlit run src/demo/app.py

Requirements:
    - Trained classifier at models/classifier.pt
    - (Optional) Fine-tuned LLM at models/llm_finetuned/
    - Webcam connected
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    FEATURE_DIM,
    WINDOW_SIZE,
    TARGET_FPS,
    BUDGET_MEDIAPIPE_MS,
    BUDGET_CLASSIFIER_MS,
    BUDGET_TOTAL_MS,
)


# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignScribe",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .path-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
    }
    .path-a-header {
        color: #4ecdc4;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .path-b-header {
        color: #ff6b6b;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .metric-label {
        color: #aaa;
        font-size: 0.85rem;
    }
    .gloss-token {
        display: inline-block;
        background: #2d2d44;
        color: #e0e0ff;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 2px;
        font-family: monospace;
        font-size: 0.95rem;
    }
    .sentence-output {
        font-size: 1.3rem;
        font-weight: 500;
        padding: 10px;
        border-radius: 8px;
        min-height: 50px;
    }
    .sentence-a {
        background: rgba(78, 205, 196, 0.1);
        border-left: 4px solid #4ecdc4;
        color: #4ecdc4;
    }
    .sentence-b {
        background: rgba(255, 107, 107, 0.1);
        border-left: 4px solid #ff6b6b;
        color: #ff6b6b;
    }
    .status-ok { color: #4ecdc4; }
    .status-warn { color: #ffd93d; }
    .status-err { color: #ff6b6b; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ────────────────────────────────────

def init_session_state():
    """Initialise all session state variables."""
    defaults = {
        "running": False,
        "gloss_history": [],
        "path_a_output": "",
        "path_b_output": "",
        "classifier_loaded": False,
        "llm_loaded": False,
        "frame_count": 0,
        "last_prediction": "",
        "last_confidence": 0.0,
        "avg_mediapipe_ms": 0.0,
        "avg_classifier_ms": 0.0,
        "latency_buffer": [],
        "classifier_latency_buffer": [],
        "consecutive_same": 0,
        "stability_threshold": 3,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Lazy-load heavy components ──────────────────────────────────────

@st.cache_resource
def load_hand_tracker():
    """Load MediaPipe hand tracker (cached across reruns)."""
    from src.capture.hand_tracker import HandTracker
    return HandTracker()


@st.cache_resource
def load_classifier():
    """Load the trained gesture classifier."""
    try:
        from src.classifier.inference import GesturePredictor
        predictor = GesturePredictor()
        return predictor
    except FileNotFoundError as e:
        st.sidebar.error(f"Classifier not found: {e}")
        return None


@st.cache_resource
def load_rule_mapper():
    """Load the rule-based grammar mapper."""
    from src.grammar_mapper.rule_mapper import RuleBasedMapper
    return RuleBasedMapper()


@st.cache_resource
def load_llm_mapper():
    """Load the LLM-based mapper (may take a moment on first load)."""
    try:
        from src.grammar_mapper.llm_mapper import LLMMapper
        return LLMMapper()
    except Exception as e:
        st.sidebar.warning(f"LLM mapper not available: {e}")
        return None


# ── Sidebar ─────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with controls and status."""
    st.sidebar.markdown("## SignScribe Controls")

    # Pipeline status
    st.sidebar.markdown("### Pipeline Status")

    tracker = load_hand_tracker()
    st.sidebar.markdown("- MediaPipe: **Ready**")

    classifier = load_classifier()
    if classifier:
        st.sidebar.markdown(
            f"- Classifier: **Loaded** ({len(classifier.label_names)} classes)"
        )
        st.session_state.classifier_loaded = True
    else:
        st.sidebar.markdown("- Classifier: **Not trained**")
        st.sidebar.info(
            "Train the classifier first:\n"
            "```\npython -m src.classifier.train\n```"
        )

    rule_mapper = load_rule_mapper()
    st.sidebar.markdown("- Path A (Rules): **Ready**")

    llm_mapper = load_llm_mapper()
    if llm_mapper and llm_mapper.is_finetuned:
        st.sidebar.markdown("- Path B (LLM): **Fine-tuned**")
        st.session_state.llm_loaded = True
    elif llm_mapper:
        st.sidebar.markdown("- Path B (LLM): **Base model** (not fine-tuned)")
        st.session_state.llm_loaded = True
    else:
        st.sidebar.markdown("- Path B (LLM): **Not available**")

    # Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")

    confidence_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.1, max_value=0.99, value=0.5, step=0.05,
        help="Minimum classifier confidence to accept a prediction"
    )

    stability_threshold = st.sidebar.slider(
        "Stability (frames)",
        min_value=1, max_value=10, value=3, step=1,
        help="How many consecutive identical predictions needed to accept"
    )
    st.session_state.stability_threshold = stability_threshold

    # Latency stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Latency")
    if st.session_state.avg_mediapipe_ms > 0:
        mp_color = "status-ok" if st.session_state.avg_mediapipe_ms <= BUDGET_MEDIAPIPE_MS else "status-warn"
        st.sidebar.markdown(
            f'<span class="{mp_color}">MediaPipe: '
            f'{st.session_state.avg_mediapipe_ms:.1f} ms</span>',
            unsafe_allow_html=True,
        )
    if st.session_state.avg_classifier_ms > 0:
        clf_color = "status-ok" if st.session_state.avg_classifier_ms <= BUDGET_CLASSIFIER_MS else "status-warn"
        st.sidebar.markdown(
            f'<span class="{clf_color}">Classifier: '
            f'{st.session_state.avg_classifier_ms:.1f} ms</span>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "*SignScribe - Final Year Project (PROJ-IT781)*"
    )

    return tracker, classifier, rule_mapper, llm_mapper, confidence_threshold


# ── Main content ────────────────────────────────────────────────────

def render_header():
    """Render the main header."""
    st.markdown(
        '<p class="main-title">SignScribe</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">'
        'Real-time Sign Language Recognition &mdash; '
        'Rule-based vs LLM Translation Comparison'
        '</p>',
        unsafe_allow_html=True,
    )


def render_gloss_tokens(tokens: list[str]):
    """Display the accumulated gloss tokens as styled badges."""
    if not tokens:
        st.markdown("*Waiting for gestures...*")
        return

    html = " ".join(
        f'<span class="gloss-token">{t}</span>' for t in tokens
    )
    st.markdown(html, unsafe_allow_html=True)


def render_outputs(path_a: str, path_b: str):
    """Render Path A and Path B outputs side by side."""
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            '<p class="path-a-header">Path A: Rule-Based</p>',
            unsafe_allow_html=True,
        )
        display_a = path_a if path_a else "..."
        st.markdown(
            f'<div class="sentence-output sentence-a">{display_a}</div>',
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            '<p class="path-b-header">Path B: LLM (T5-small)</p>',
            unsafe_allow_html=True,
        )
        display_b = path_b if path_b else "..."
        st.markdown(
            f'<div class="sentence-output sentence-b">{display_b}</div>',
            unsafe_allow_html=True,
        )


def render_manual_input(rule_mapper, llm_mapper):
    """Manual text input mode for testing without webcam."""
    st.markdown("---")
    st.markdown("### Manual Gloss Input")
    st.markdown(
        "Type ISL gloss tokens (space-separated) to test the "
        "translation paths without the webcam."
    )

    gloss_input = st.text_input(
        "Enter glosses:",
        placeholder="e.g., I SCHOOL GO",
        key="manual_gloss",
    )

    if gloss_input:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                '<p class="path-a-header">Path A: Rule-Based</p>',
                unsafe_allow_html=True,
            )
            path_a = rule_mapper.map(gloss_input)
            st.markdown(
                f'<div class="sentence-output sentence-a">{path_a}</div>',
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(
                '<p class="path-b-header">Path B: LLM (T5-small)</p>',
                unsafe_allow_html=True,
            )
            if llm_mapper:
                path_b = llm_mapper.map(gloss_input)
            else:
                path_b = "(LLM not loaded)"
            st.markdown(
                f'<div class="sentence-output sentence-b">{path_b}</div>',
                unsafe_allow_html=True,
            )


def render_webcam_mode(tracker, classifier, rule_mapper, llm_mapper, confidence_threshold):
    """Live webcam mode with real-time recognition."""
    from src.buffering.sliding_window import SlidingWindowBuffer

    st.markdown("---")
    st.markdown("### Live Webcam Recognition")

    # Controls
    col_start, col_clear, col_info = st.columns([1, 1, 3])
    with col_start:
        start_btn = st.button(
            "Start Camera" if not st.session_state.running else "Stop Camera",
            type="primary",
        )
        if start_btn:
            st.session_state.running = not st.session_state.running
            if not st.session_state.running:
                st.session_state.gloss_history = []

    with col_clear:
        if st.button("Clear History"):
            st.session_state.gloss_history = []
            st.session_state.path_a_output = ""
            st.session_state.path_b_output = ""

    # Gloss token display
    st.markdown("**Recognised Glosses:**")
    render_gloss_tokens(st.session_state.gloss_history)

    # Output display
    render_outputs(
        st.session_state.path_a_output,
        st.session_state.path_b_output,
    )

    # Webcam feed
    if st.session_state.running:
        if not classifier:
            st.error(
                "Classifier not loaded. Train it first: "
                "`python -m src.classifier.train`"
            )
            return

        video_placeholder = st.empty()
        status_placeholder = st.empty()

        buffer = SlidingWindowBuffer(overlap=WINDOW_SIZE)  # no overlap for recording-style

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam. Is it connected?")
            st.session_state.running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_interval = 1.0 / TARGET_FPS

        try:
            while st.session_state.running:
                t_frame_start = time.perf_counter()

                ret, frame_bgr = cap.read()
                if not ret:
                    st.warning("Failed to read from webcam.")
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # MediaPipe
                t_mp_start = time.perf_counter()
                features, results = tracker.extract_landmarks(frame_rgb)
                t_mp_end = time.perf_counter()
                mp_ms = (t_mp_end - t_mp_start) * 1000

                # Update latency tracking
                st.session_state.latency_buffer.append(mp_ms)
                if len(st.session_state.latency_buffer) > 30:
                    st.session_state.latency_buffer.pop(0)
                st.session_state.avg_mediapipe_ms = np.mean(
                    st.session_state.latency_buffer
                )

                # Draw landmarks
                frame_bgr = tracker.draw_landmarks(frame_bgr, results)

                num_hands = (
                    len(results.multi_hand_landmarks)
                    if results.multi_hand_landmarks
                    else 0
                )

                # Add to sliding window buffer
                window = buffer.add_frame(features)

                if window is not None and num_hands > 0:
                    # Classify
                    t_clf_start = time.perf_counter()
                    label, confidence, clf_ms = classifier.predict(window)
                    t_clf_end = time.perf_counter()

                    st.session_state.classifier_latency_buffer.append(clf_ms)
                    if len(st.session_state.classifier_latency_buffer) > 30:
                        st.session_state.classifier_latency_buffer.pop(0)
                    st.session_state.avg_classifier_ms = np.mean(
                        st.session_state.classifier_latency_buffer
                    )

                    if confidence >= confidence_threshold:
                        # Stability check
                        if label == st.session_state.last_prediction:
                            st.session_state.consecutive_same += 1
                        else:
                            st.session_state.consecutive_same = 1
                            st.session_state.last_prediction = label

                        if st.session_state.consecutive_same >= st.session_state.stability_threshold:
                            # Accept this prediction
                            if (not st.session_state.gloss_history or
                                    st.session_state.gloss_history[-1] != label):
                                st.session_state.gloss_history.append(label)

                                # Update translations
                                st.session_state.path_a_output = rule_mapper.map_sequence(
                                    st.session_state.gloss_history
                                )
                                if llm_mapper:
                                    st.session_state.path_b_output = llm_mapper.map_sequence(
                                        st.session_state.gloss_history
                                    )

                            st.session_state.consecutive_same = 0

                    st.session_state.last_confidence = confidence

                # HUD overlay on frame
                cv2.putText(
                    frame_bgr,
                    f"MP: {mp_ms:.0f}ms | Hands: {num_hands}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if mp_ms <= BUDGET_MEDIAPIPE_MS else (0, 0, 255),
                    2,
                )
                if st.session_state.last_prediction:
                    cv2.putText(
                        frame_bgr,
                        f"Pred: {st.session_state.last_prediction} "
                        f"({st.session_state.last_confidence:.0%})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2,
                    )

                # Display frame
                video_placeholder.image(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_column_width=True,
                )

                # Throttle
                elapsed = time.perf_counter() - t_frame_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

                st.session_state.frame_count += 1

        finally:
            cap.release()


# ── Evaluation panel ────────────────────────────────────────────────

def render_evaluation_panel(rule_mapper, llm_mapper):
    """Show evaluation metrics comparison panel."""
    st.markdown("---")
    st.markdown("### Evaluation Comparison")
    st.markdown(
        "Run the evaluation module with your test data to see metrics here. "
        "For now, here's a comparison using the manual inputs above."
    )

    # Quick demo with a few examples
    test_glosses = [
        "I SCHOOL GO",
        "HE FOOD EAT",
        "SHE BOOK READ",
        "I WATER DRINK FUTURE",
        "HELLO",
        "WE PARK GO TOMORROW",
        "YOU HELP WANT",
        "I FRIEND HELP",
    ]
    references = [
        "I go to school.",
        "He eats food.",
        "She reads a book.",
        "I will drink water.",
        "Hello.",
        "We will go to the park tomorrow.",
        "Do you want help?",
        "I help my friend.",
    ]

    rule_preds = [rule_mapper.map(g) for g in test_glosses]
    llm_preds = (
        [llm_mapper.map(g) for g in test_glosses]
        if llm_mapper
        else ["(not available)"] * len(test_glosses)
    )

    # Display comparison table
    comparison_data = []
    for g, ref, rp, lp in zip(test_glosses, references, rule_preds, llm_preds):
        comparison_data.append({
            "Gloss": g,
            "Reference": ref,
            "Path A (Rules)": rp,
            "Path B (LLM)": lp,
        })

    st.dataframe(comparison_data, use_container_width=True, hide_index=True)

    # Compute metrics if LLM is available
    if llm_mapper:
        from src.evaluation.metrics import compute_bleu, compute_hallucination_rate

        col_bleu, col_hall = st.columns(2)

        with col_bleu:
            st.markdown("#### BLEU-4 Scores")
            rule_bleu = compute_bleu(rule_preds, references)
            llm_bleu = compute_bleu(llm_preds, references)
            st.metric("Path A (Rule-based)", f"{rule_bleu['corpus_bleu4']:.1f}")
            st.metric("Path B (LLM)", f"{llm_bleu['corpus_bleu4']:.1f}")

        with col_hall:
            st.markdown("#### Hallucination Rate")
            rule_hall = compute_hallucination_rate(rule_preds, test_glosses)
            llm_hall = compute_hallucination_rate(llm_preds, test_glosses)
            st.metric(
                "Path A (Rule-based)",
                f"{rule_hall['hallucination_rate']:.1%}",
            )
            st.metric(
                "Path B (LLM)",
                f"{llm_hall['hallucination_rate']:.1%}",
            )


# ── Main app ────────────────────────────────────────────────────────

def main():
    tracker, classifier, rule_mapper, llm_mapper, confidence_threshold = (
        render_sidebar()
    )
    render_header()

    # Tab layout
    tab_live, tab_manual, tab_eval = st.tabs([
        "Live Webcam",
        "Manual Input",
        "Evaluation",
    ])

    with tab_live:
        render_webcam_mode(
            tracker, classifier, rule_mapper, llm_mapper,
            confidence_threshold,
        )

    with tab_manual:
        render_manual_input(rule_mapper, llm_mapper)

    with tab_eval:
        render_evaluation_panel(rule_mapper, llm_mapper)


if __name__ == "__main__":
    main()
