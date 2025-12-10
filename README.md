# ECHO_TWIN_CORE — TTS Research Pipeline & Assistant

A comprehensive audio processing, training, and synthesis framework that combines educational implementations with modern production-grade tools.

---

## 🎯 Overview

Echo Twin Core is a dual-purpose project designed to demonstrate:
1.  **Core Audio Concepts:** A "from-scratch" implementation of an audio autoencoder using pure NumPy, showing how raw audio is preprocessed, encoded into mel spectrograms, and reconstructed.
2.  **Modern Voice Assistant Pipeline:** A fully functional voice assistant integration using industry-standard tools (Hugging Face Transformers, Gradio, Vosk/Whisper) for Speech-to-Text (ASR) and Text-to-Speech (TTS).

Whether you want to learn about audio deep learning fundamentals or run a local voice assistant, Echo Twin Core provides the modular blocks to do so.

## ✨ Key Features

*   **Training Pipeline:**
    *   Raw audio preprocessing (WAV to Mel Spectrograms).
    *   **NumPy-based MLP Autoencoder:** Train a neural network from scratch without PyTorch/TensorFlow for educational purposes.
    *   Griffin-Lim algorithm for audio reconstruction.
    *   Watermarking for model authenticity verification.
*   **Voice Assistant:**
    *   **Speech-to-Text (ASR):** Supports offline (Vosk) and online (OpenAI Whisper) recognition.
    *   **Natural Language Processing (NLP):** Rule-based intent recognition system.
    *   **Text-to-Speech (TTS):** Integrated high-quality synthesis using Hugging Face's MMS models.
    *   Interactive CLI mode (text-based) and Audio-in/Audio-out mode.
*   **Web Interface:**
    *   Gradio-based UI for easy text-to-speech generation.
    *   Multi-language support (English, French, Spanish).

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Echo_Twin_Core.git
    cd Echo_Twin_Core
    ```

2.  **Install Dependencies:**
    (Requires Python 3.11+)
    ```bash
    pip install -r requirements.txt
    ```

    *Note: For the assistant's offline ASR, you may need to download a Vosk model (see Assistant configuration below).*

---

## 🚀 How to Run

### 1. The Full Training Pipeline
Execute the complete workflow: preprocess data, train the NumPy autoencoder, evaluate reconstruction, and run a TTS test.

```bash
python -m scripts.run_full_pipeline
```
*Outputs are saved to `logs/` and `models/checkpoints/`.*

### 2. Web UI (Text-to-Speech)
Launch a user-friendly web interface to generate speech from text. No training required.

```bash
python scripts/ui_app.py
```
*Open the provided local URL (usually `http://127.0.0.1:7860`) in your browser.*

### 3. Voice Assistant
Run the interactive assistant.

**Interactive Text Mode (No microphone/audio file needed):**
```bash
python -m assistant.assistant_demo
```

**Audio File Mode:**
Process an existing audio file through the assistant pipeline:
```bash
python -m assistant.assistant_demo audio/001.wav
```

---

## ⚙️ Configuration

### Assistant Settings
Configuration is managed in `config/settings.json`.

**Speech Recognition Engine:**
*   **Vosk (Offline/Fast):**
    1.  Download a model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models).
    2.  Extract it to `assistant/vosk-model-small/`.
    3.  Set `"asr": "vosk"` in `settings.json`.
*   **Whisper (High Accuracy):**
    1.  Ensure `openai-whisper` is installed.
    2.  Set `"asr": "whisper"` in `settings.json`.

---

## 🧱 Project Structure

```
Echo_Twin_Core/
├── assistant/              # Assistant logic (ASR, NLP, Response)
├── audio/                  # Raw input audio samples
├── config/                 # Configuration files (settings.json)
├── data/                   # Data processing scripts & storage
├── logs/                   # Training logs and generated outputs
├── models/                 # Model architecture (NumPy Autoencoder)
├── scripts/                # Main execution scripts
│   ├── run_full_pipeline.py   # Master pipeline script
│   ├── ui_app.py              # Gradio Web UI
│   ├── train_model.py         # Training logic
│   └── tts_infer.py           # TTS inference logic
├── utils/                  # Utilities (Audio, Watermarking)
└── requirements.txt        # Python dependencies
```

