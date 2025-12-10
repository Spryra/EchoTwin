# assistant/speech_recognition.py
"""
ASR wrapper for Echo Twin assistant.
Supports Vosk (offline) and Whisper (optional).
"""

import os
import logging
from pathlib import Path
import json
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("assistant.asr")

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = os.path.join(BASE_DIR, "config", "settings.json")
with open(CONFIG_PATH, "r") as fh:
    SETTINGS = json.load(fh)

def transcribe_vosk(wav_path, model_path=None):
    try:
        from vosk import Model, KaldiRecognizer
        import json as _json
        model_dir = model_path or SETTINGS["assistant"]["vosk_model_path"]
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Vosk model not found at {model_dir}. Download from Vosk site and unpack.")
        model = Model(model_dir)
        wf, sr = sf.read(wav_path)
        if wf.ndim > 1:
            wf = np.mean(wf, axis=1)
        rec = KaldiRecognizer(model, sr)
        rec.AcceptWaveform(wf.tobytes())
        result = rec.FinalResult()
        res = _json.loads(result)
        text = res.get("text", "")
        return text
    except Exception as e:
        log.warning("Vosk transcription failed: %s", e)
        return ""

def transcribe_whisper(wav_path, model_size="small"):
    try:
        import whisper
        model = whisper.load_model(model_size)
        out = model.transcribe(wav_path)
        return out.get("text", "")
    except Exception as e:
        log.warning("Whisper transcription failed: %s", e)
        return ""

def transcribe_audio(wav_path):
    """
    Transcribes speech from a WAV file using the configured ASR engine.
    """
    if SETTINGS["assistant"]["asr"] == "vosk":
        return transcribe_vosk(wav_path)
    elif SETTINGS["assistant"]["asr"] == "whisper":
        return transcribe_whisper(wav_path)
    else:
        return transcribe_vosk(wav_path)
