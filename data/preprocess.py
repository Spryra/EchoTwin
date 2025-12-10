# File: D:\Echo_Twin_Core\data\preprocess.py
"""
Echo Twin - data/preprocess.py (LJSpeech Full Integration)

Changelog (2025-11-06):
- Added full support for LJSpeech dataset (metadata.csv + normalized text).
- Uses config["dataset"]["use_normalized_text"] to select transcription column.
- Auto-validates dataset paths, ensures directories exist.
- ASCII-safe logging; compatible with both Linux and Windows.
- Backward-compatible: if dataset.type != 'ljspeech', falls back to raw_audio_dir.
"""

from __future__ import annotations
import os
import sys
import csv
import json
import logging
from glob import glob

import numpy as np
from tqdm import tqdm

# Add root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_utils import (
    load_audio,
    trim_silence,
    normalize_audio,
    compute_mel,
    save_mel_spectrogram,
)

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[preprocess] %(message)s")
log = logging.getLogger("preprocess")

# ----------------------------------------------------------------------
# Paths & Configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "settings.json")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

data_cfg = SETTINGS.get("data", {})
dataset_cfg = SETTINGS.get("dataset", {})

RAW_AUDIO_DIR = os.path.join(BASE_DIR, data_cfg.get("raw_audio_dir", "./audio"))
PROCESSED_DIR = os.path.join(BASE_DIR, data_cfg.get("processed_dir", "./data/processed"))
MELS_DIR = os.path.join(BASE_DIR, data_cfg.get("mels_dir", "./data/mels"))
TRANSCRIPT_CSV = os.path.join(BASE_DIR, data_cfg.get("transcript_csv", "./data/transcript.csv"))

SAMPLE_RATE = data_cfg.get("sample_rate", 22050)
N_MELS = data_cfg.get("n_mels", 80)
HOP_LENGTH = data_cfg.get("hop_length", 256)
WIN_LENGTH = data_cfg.get("win_length", 1024)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MELS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# LJSpeech Dataset Loader
# ----------------------------------------------------------------------
def load_ljspeech_dataset(dataset_path: str, metadata_file: str, audio_dir: str, use_normalized_text: bool = True):
    meta_path = os.path.join(dataset_path, metadata_file)
    wav_dir = os.path.join(dataset_path, audio_dir)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"LJSpeech metadata.csv not found: {meta_path}")
    if not os.path.exists(wav_dir):
        raise FileNotFoundError(f"LJSpeech audio directory not found: {wav_dir}")

    entries = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 3:
                wav_id = row[0].strip()
                text = row[2].strip() if use_normalized_text else row[1].strip()
                wav_path = os.path.join(wav_dir, f"{wav_id}.wav")
                if os.path.exists(wav_path):
                    entries.append((wav_path, text))
    log.info(f"Loaded {len(entries)} entries from LJSpeech.")
    return entries

# ----------------------------------------------------------------------
# Main Preprocessing Function
# ----------------------------------------------------------------------
def preprocess_dataset():
    dataset_type = dataset_cfg.get("type", "").lower()
    entries = []

    if dataset_type == "ljspeech" and os.path.exists(dataset_cfg.get("path", "")):
        log.info(f"Detected LJSpeech dataset at {dataset_cfg['path']}")
        entries = load_ljspeech_dataset(
            dataset_cfg["path"],
            dataset_cfg.get("metadata_file", "metadata.csv"),
            dataset_cfg.get("audio_dir", "wavs"),
            use_normalized_text=dataset_cfg.get("use_normalized_text", True),
        )
    else:
        log.info("Using default raw_audio_dir for preprocessing.")
        wav_files = sorted(glob(os.path.join(RAW_AUDIO_DIR, "*.wav")))
        entries = [(wf, "") for wf in wav_files]

    if not entries:
        raise RuntimeError("No audio files found for preprocessing.")

    log.info(f"Processing {len(entries)} audio files → mel spectrograms.")

    transcripts = []

    for idx, (wav_path, text) in enumerate(tqdm(entries, ncols=80)):
        try:
            audio, sr = load_audio(wav_path, sr=SAMPLE_RATE)
            audio = trim_silence(audio)
            audio = normalize_audio(audio)
            mel = compute_mel(audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
            mel_filename = f"{idx+1:05d}.npy"
            mel_path = os.path.join(MELS_DIR, mel_filename)
            save_mel_spectrogram(mel_path, mel)
            transcripts.append([mel_filename, text])
        except Exception as e:
            log.warning(f"Skipping file {wav_path}: {e}")

    with open(TRANSCRIPT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mel_file", "text"])
        writer.writerows(transcripts)

    log.info(f"Preprocessing complete. {len(transcripts)} mel files saved.")
    log.info(f"Transcript file written to: {TRANSCRIPT_CSV}")

# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    preprocess_dataset()
