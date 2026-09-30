"""
SignScribe — Central configuration.

All magic numbers from the synopsis live here so they're easy to find,
change, and reference from any module.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_WEIGHTS = MODELS_DIR / "classifier.pt"
LLM_FINETUNED_DIR = MODELS_DIR / "llm_finetuned"

# ── MediaPipe / Feature extraction ────────────────────────────────────
NUM_LANDMARKS = 21          # per hand
NUM_COORDS = 3              # x, y, z
NUM_HANDS = 2               # left + right
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS * NUM_HANDS  # 126

# ── Temporal windowing ────────────────────────────────────────────────
WINDOW_SIZE = 30            # frames per gesture window
TARGET_FPS = 25             # processing fps (downsample webcam to this)
WINDOW_DURATION_SEC = WINDOW_SIZE / TARGET_FPS  # 1.2 s

# ── Classifier architecture ──────────────────────────────────────────
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_BIDIRECTIONAL = True
LSTM_DROPOUT = 0.3

# ── Vocabulary ────────────────────────────────────────────────────────
# 26 letters + 10 digits + 60 ISL words = 96 total classes
# ⚠️  The ISL_WORDS list is a PLACEHOLDER — replace with your actual
#     curated list before recording data.  The 26 letters and 10 digits
#     are auto-generated below.
ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = [str(d) for d in range(10)]
ISL_WORDS: list[str] = [
    # --- TODO: fill in your 60 common ISL words ---
    # Example entries (remove and replace with real list):
    # "HELLO", "THANK_YOU", "PLEASE", "SORRY", "YES", "NO", ...
]
VOCAB = ALPHABETS + DIGITS + ISL_WORDS
NUM_CLASSES = len(VOCAB)  # should be 96 when ISL_WORDS has 60 entries

# ── Dataset sizing (synopsis targets) ────────────────────────────────
TARGET_SIGNERS = 10         # aspirational; realistically 1-3
TARGET_REPS_PER_SIGNER = 15
AUGMENTATION_FACTOR = 4     # mirror + scale + temporal jitter + brightness
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ── Latency budget (ms) — REVISED per Phase 1 benchmarks ────────────
# MediaPipe on Intel i7 (integrated graphics, no CUDA) measures ~21 ms,
# higher than the synopsis's 15 ms estimate.  However, the BiLSTM
# classifier is much smaller than typical and will use ~3 ms (not 12).
# Net result: total still fits comfortably under 40 ms → 25 fps.
BUDGET_MEDIAPIPE_MS = 22    # measured: ~21 ms mean on target hardware
BUDGET_BUFFERING_MS = 1
BUDGET_CLASSIFIER_MS = 5    # revised: 680K-param BiLSTM, ~3 ms expected
BUDGET_GRAMMAR_MS = 2
BUDGET_RENDER_MS = 5
BUDGET_TOTAL_MS = 40        # = 1000 / TARGET_FPS

# ── LLM ──────────────────────────────────────────────────────────────
LLM_MODEL_NAME = "t5-small"   # ~60 M params, seq2seq
LLM_MAX_INPUT_LEN = 64        # gloss tokens are short
LLM_MAX_OUTPUT_LEN = 128      # generated sentence length cap
