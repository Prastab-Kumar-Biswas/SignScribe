# SignScribe

**Real-time Indian Sign Language (ISL) to Text Generator**.

Lightweight, high-performance sign language recognition pipeline designed to run smoothly on standard CPU laptops with zero external GPU requirements.

---

## 🌟 Key Features

- **Real-Time Camera Pipeline**: Captures mirrored webcam frames and extracts 21 3D hand landmarks per hand at up to 60 FPS using MediaPipe Hands Lite.
- **BiLSTM Gesture Classifier**: Evaluates 30-frame temporal windows across 97 sign classes with ~15 ms CPU inference latency.
- **Physical & Geometric Disambiguation**:
  - **`F` vs `X`**: Disambiguated by extended finger count (cross 2 fingers per hand for `F` vs 1 finger per hand for `X`).
  - **`K` vs `E`**: Disambiguated by vertical wrist displacement ($|dy| \ge 0.35$ for reaching hand in `K`).
  - **`J` vs `Q`**: Disambiguated by wrist elevation separation ($|dy| \ge 0.28$ for drawing down into palm in `J`).
  - **`L` vs `1`**: Disambiguated by physical thumb spread ($d \ge 0.18$ for wide $90^\circ$ thumb in `L`).
  - **`I`**: Cleanly evaluated as a single-hand gesture with relaxed thumb spread ($d \ge 0.16$).
- **Fingerspelling Mode (Masked Logits)**: Automatically eliminates spurious whole-word hallucinations during letter typing by masking logits to the 36 alphanumeric classes (`0–9`, `A–Z`).
- **Deterministic Text Generator**: Assembles fingerspelled letters into cohesive English words and sentences with automated space handling and punctuation—no heavy LLMs or rule mappers required.
- **Dual Interfaces**: Includes both an interactive Streamlit Web UI and a zero-latency native OpenCV desktop application.

---

## 🏗️ Architecture Overview

```
Webcam (Mirrored) ──► MediaPipe Hands (21 Landmarks × 2 hands)
                            │
                            ▼
              Gated Sliding Window Buffer (30 Frames)
              [Active Hand Presence Gating & Idle Reset]
                            │
                            ▼
             BiLSTM Classifier (Spatial & Geometric Arbitration)
              [Fingerspelling Masking & Stability Thresholds]
                            │
                            ▼
                   SignTextGenerator
              [Fingerspelling Assembly & Sentence Formatting]
                            │
                            ▼
                   English Text Output
```

### Supported Vocabulary (97 Signs)
- **Two-Hand Alphabets (20)**: `A`, `B`, `D`, `E`, `F`, `G`, `H`, `J`, `K`, `M`, `N`, `P`, `Q`, `R`, `S`, `T`, `W`, `X`, `Y`, `Z`
- **One-Hand Alphabets (6)**: `C`, `I`, `L`, `O`, `U`, `V`
- **Digits (10)**: `0` to `9`
- **ISL Words (61)**: Greetings, food items, verbs, family relations, animals, and common expressions (e.g., `HELLO`, `GOOD_MORNING`, `THANK_YOU`, `WHAT_IS_YOUR_NAME`, `WATER`, `DRINK`, etc.)

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Prastab-Kumar-Biswas/signscribe.git
cd signscribe

# Create and activate virtual environment (Python 3.10 - 3.12)
python -m venv venv
.\venv\Scripts\Activate.ps1       # Windows PowerShell
# source venv/bin/activate       # macOS / Linux

# Install CPU PyTorch
pip install torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Readiness

Run preflight checks to confirm environment and model weights:
```bash
python -m src.preflight
```

---

## 💻 Running the Application

### Option A: Ultra-Fast Native Desktop Window (Recommended)
Runs directly via native OpenCV with 30–60 FPS responsiveness:
```bash
python run_live.py
```
- **HUD**: Real-time FPS, hands detected, predicted sign badge, and live generated text.
- **Keyboard Controls**:
  - `[Space]` — Complete word / add space
  - `[Backspace]` — Delete last letter or word
  - `[c]` — Clear all generated text
  - `[m]` — Toggle mode (`Letters & Numbers Only` vs `All 97 Signs`)
  - `[q]` — Quit

### Option B: Streamlit Web Application
Runs the browser interface with side-by-side video feed, live text cards, and sensitivity sliders:
```bash
streamlit run src/demo/app.py
```
- **Mode Selection**: Switch between **Letters & Numbers (Fingerspelling)** and **All 97 Signs** directly from the sidebar.
- **Controls**: Real-time buttons for Space, Backspace, Copy, and Reset.
- **ISL Guide**: Interactive cheat-sheet with signing tips.

---

## 🤟 ISL Signing Guide & Tips

- **`I` (One Hand)**: Hold up **one hand** with the index finger pointing up and thumb relaxed. (Do not touch with the other hand—in this dataset, `I` is a single-handed sign).
- **`L` vs `1` (One Hand)**:
  - **`L`**: Point index finger up and extend thumb wide horizontally at $90^\circ$.
  - **`1`**: Point index finger up and fold thumb tight across the fist.
- **`F` vs `X` (Two Hands)**:
  - **`F`**: Cross with **2 extended fingers** (index + middle) on both hands.
  - **`X`**: Cross with **only 1 extended finger** (index finger) on both hands.
- **`K` vs `E` (Two Hands)**:
  - **`K`**: Keep the reaching hand held **high above** the base hand.
  - **`E`**: Keep both hands at the **same horizontal level**.
- **`J` vs `Q` (Two Hands)**:
  - **`J`**: Hold the base palm lower down and draw the 'J' from above.
  - **`Q`**: Keep both hands at the same horizontal level.

---

## 📂 Project Structure

```
signscribe/
├── run_live.py           # Native high-speed OpenCV desktop runner (30-60 FPS)
├── requirements.txt      # Clean, minimal CPU-only dependencies
├── models/
│   └── classifier.pt     # Trained BiLSTM classifier checkpoint (97 classes)
├── data/
│   └── raw/              # Recorded gesture landmark samples (.npy)
├── src/
│   ├── config.py         # Global constants and vocabulary definitions
│   ├── preflight.py      # Preflight sanity checks
│   ├── capture/          # Webcam capture & MediaPipe hand tracker
│   ├── buffering/        # Hand-gated sliding window buffer
│   ├── classifier/       # BiLSTM model architecture, dataset, inference
│   ├── text_generator/   # Fast fingerspelling assembly & text formatter
│   └── demo/             # Streamlit web demo application
└── scripts/
    ├── convert_csv_landmarks.py   # Landmark data converter
    └── convert_video_dataset.py   # Video to landmarks converter
```

---

## 📜 License

MIT License. Designed and developed for assistive technology and accessible real-time communication.
