# SignScribe

**Real-time Indian Sign Language → Fluent English Text**, comparing rule-based grammar mapping (Path A) against a fine-tuned LLM (Path B).

> Final-year B.Tech IT project (PROJ-IT781).

---

## Quick Summary

| Stage | What | Where | Module |
|-------|------|-------|--------|
| M1 | Webcam video capture | **Local** | `src/capture/` |
| M2 | MediaPipe hand landmark extraction | **Local** | `src/capture/` |
| M3 | Sliding-window buffer + dataset recording tool | **Local** | `src/buffering/` |
| M4 | BiLSTM gesture classifier (train + infer) | **Local** (CPU) | `src/classifier/` |
| M5a | Rule-based grammar mapper (Path A) | **Local** | `src/grammar_mapper/` |
| M5b | T5-small LoRA fine-tuning (Path B) | **Google Colab** (T4 GPU) | `colab/llm_finetune.ipynb` |
| M5b | LLM inference (using fine-tuned weights) | **Local** (CPU) | `src/grammar_mapper/` or `src/demo/` |
| M6 | Evaluation (Accuracy, BLEU-4, Hallucination Rate) | **Local** | `src/evaluation/` |
| M7 | Streamlit demo app | **Local** | `src/demo/` |

---

## Hardware Requirements

| Machine | Specs | Role |
|---------|-------|------|
| Local laptop | Intel Core i7, 16 GB RAM, **no discrete GPU** | Everything except LLM fine-tuning |
| Google Colab (free tier) | T4 GPU, ~16 GB VRAM | LLM fine-tuning only |

> **Important**: The local setup uses **CPU-only PyTorch**. No CUDA required.

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/signscribe.git
cd signscribe

# 2. Create a virtual environment (Python 3.12)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install CPU-only PyTorch first (special index URL)
pip install torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import torch; print(torch.__version__); import mediapipe; print('MediaPipe OK'); import streamlit; print('Streamlit OK')"
```

### Check project readiness

Before training or presenting the demo, run the read-only preflight check:

```bash
python -m src.preflight
```

Use `--strict` to return a non-zero exit code when the recorded dataset does
not cover the configured vocabulary, a checkpoint is missing or incompatible,
or a sample has an invalid shape.

---

## Google Colab Setup (LLM Fine-Tuning Only)

1. Open `colab/llm_finetune.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Follow the notebook instructions to upload your gloss→sentence dataset.
4. After fine-tuning, download the LoRA adapter weights.
5. Place them in `models/llm_finetuned/` on your local machine.

---

## Project Structure

```
signscribe/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # Recorded gesture sequences
│   └── processed/        # Windowed & augmented tensors
├── src/
│   ├── config.py         # Central configuration (all constants)
│   ├── capture/          # M1, M2 — Webcam + MediaPipe
│   ├── buffering/        # M3 — Sliding window + recording tool
│   ├── classifier/       # M4 — BiLSTM model, training, inference
│   ├── grammar_mapper/   # M5a — Rule-based path
│   ├── evaluation/       # M6 — Metrics
│   └── demo/             # M7 — Streamlit app
├── colab/
│   └── llm_finetune.ipynb  # M5b — T5-small LoRA fine-tuning
└── models/
    ├── classifier.pt       # Trained classifier weights
    └── llm_finetuned/      # LoRA adapter weights from Colab
```

---

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Feature vector | 21 landmarks × 3 coords × 2 hands = **126 dims** |
| Temporal window | **30 frames** @ 25 fps = 1.2 s |
| Classifier | 2-layer bidirectional LSTM, hidden=128, **~680K params** |
| Classes | 26 letters + 10 digits + 60 ISL words = **96** |
| Latency target | **≤ 40 ms/frame** (25 fps) |
| LLM | **T5-small** (~60M params) fine-tuned with **LoRA/PEFT** |

### Evaluation Metrics

- **Accuracy, Precision, Recall, F1** — via scikit-learn
- **BLEU-4** — via sacrebleu (LLM fluency vs. reference sentences)
- **Hallucination Rate** — custom metric (unsupported words / total words)

---

## Build Phases

| Phase | Modules | Status |
|-------|---------|--------|
| 0 — Scaffold | Repo structure, configs | Done |
| 1 — Webcam + MediaPipe | M1, M2 | Done |
| 2 — Buffer + Recorder | M3 | Done |
| 3 — Classifier | M4 | Done |
| 4 — Rule-based Mapper | M5a | Done |
| 5 — LLM Fine-tuning | M5b (Colab) | Done |
| 6 — Evaluation | M6 | Done |
| 7 — Demo App | M7 | Done |

---

## Running the Demo

```bash
# Make sure venv is activated, classifier is trained, and you're in the project root
streamlit run src/demo/app.py
```

The demo has three tabs:
- **Live Webcam** — real-time gesture recognition with both translation paths
- **Manual Input** — type gloss tokens to test translations without webcam
- **Evaluation** — side-by-side comparison table with BLEU-4 and hallucination rate

---

## Running the Data Recorder

```bash
python -m src.buffering.data_recorder
```

## Training the Classifier (locally)

```bash
python -m src.classifier.train --epochs 50 --batch-size 32
```

Or use the Colab notebook (`colab/llm_finetune.ipynb`) for faster GPU training.

---

## License

This project is developed for academic purposes as part of B.Tech IT final-year coursework.
