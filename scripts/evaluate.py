# File: D:\Echo_Twin_Core\scripts\evaluate.py
"""
Echo Twin - scripts/evaluate.py (Improved Mel Evaluation v2)
------------------------------------------------------------
Purpose:
    Evaluate the trained autoencoder model by reconstructing an audio waveform
    from stored mel spectrograms and saving the results for listening tests.

Changelog (2025-11-06):
    - Integrated refined mel reconstruction (mel_db_to_magnitude_db → mel_power_to_waveform).
    - Added dynamic sample rate loading from config.
    - Improved handling of mismatched shapes (auto transpose).
    - Retains full compatibility with 2025-11-03 structure and logging style.
"""

from __future__ import annotations
import os
import sys
import json
import logging
import numpy as np

# Ensure repo root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_architecture import forward, load_checkpoint
from utils.audio_utils import (
    load_mel_file,
    mel_db_to_magnitude_db,
    mel_power_to_waveform,
    save_wav,
)

# ----------------------------------------------------------
# Logging setup (ASCII-safe)
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[evaluate] %(message)s")
log = logging.getLogger("evaluate")

# ----------------------------------------------------------
# Load config safely
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "settings.json")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

def _cfg_path(*keys, fallback=None):
    cur = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return fallback
    return cur

mels_dir = _cfg_path("paths", "mels_dir", fallback=_cfg_path("data", "mels_dir"))
checkpoints_dir = _cfg_path("paths", "checkpoints_dir", fallback=_cfg_path("model", "checkpoint_dir"))
logs_dir = _cfg_path("paths", "logs_dir", fallback=_cfg_path("logging", "log_dir"))
sample_rate = cfg.get("data", {}).get("sample_rate", 22050)

if not mels_dir or not checkpoints_dir:
    raise KeyError("Could not resolve mels_dir or checkpoints_dir from config.")

mels_dir = os.path.join(BASE_DIR, mels_dir)
checkpoints_dir = os.path.join(BASE_DIR, checkpoints_dir)
logs_dir = os.path.join(BASE_DIR, logs_dir)
os.makedirs(logs_dir, exist_ok=True)
eval_dir = os.path.join(logs_dir, "evaluations")
os.makedirs(eval_dir, exist_ok=True)

# ----------------------------------------------------------
# Load latest model checkpoint
# ----------------------------------------------------------
npz_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".npz")]
if not npz_files:
    raise FileNotFoundError(f"No model checkpoints found in {checkpoints_dir}")

npz_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)))
latest_ckpt = os.path.join(checkpoints_dir, npz_files[-1])
log.info(f"Using checkpoint: {latest_ckpt}")

params = load_checkpoint(latest_ckpt)

# ----------------------------------------------------------
# Load a mel spectrogram sample
# ----------------------------------------------------------
mel_files = [f for f in os.listdir(mels_dir) if f.endswith(".npy")]
if not mel_files:
    raise FileNotFoundError(f"No mel spectrograms found in {mels_dir}")

mel_path = os.path.join(mels_dir, mel_files[0])
mel_db = load_mel_file(mel_path)
log.info(f"Loaded mel spectrogram: {mel_path}")

# Ensure correct orientation (time, features)
if mel_db.shape[0] < mel_db.shape[1]:
    mel_db = mel_db.T

mel_input = mel_db.astype(np.float32)
mel_reconstructed, _ = forward(params, mel_input)
mel_reconstructed = mel_reconstructed.T

# ----------------------------------------------------------
# Convert mel → waveform
# ----------------------------------------------------------
mel_power = mel_db_to_magnitude_db(mel_reconstructed)
waveform = mel_power_to_waveform(
    mel_power,
    sr=sample_rate,
    n_iter=64,  # more Griffin-Lim iterations = clearer sound
    hop_length=256,
    win_length=1024,
)

# ----------------------------------------------------------
# Save results
# ----------------------------------------------------------
output_path = os.path.join(eval_dir, "reconstructed.wav")
save_wav(output_path, waveform, sr=sample_rate)
log.info(f"Reconstructed audio saved to: {output_path}")

if __name__ == "__main__":
    log.info("Evaluation completed successfully.")
