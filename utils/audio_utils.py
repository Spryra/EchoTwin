# File: D:\Echo_Twin_Core\utils\audio_utils.py
"""
Echo Twin - utils/audio_utils.py (Full Replacement, 2025-11-06)
----------------------------------------------------------------
Purpose:
    Unified and stable audio processing utilities.

Fix Summary:
- Restored log_safe() helper to fix ImportError from scripts.train_model.
- Keeps augment_*(), spec_augment(), compute_mel(), and inverse transforms.
- Preserves all existing functionality (no removals).
- Ensures full backward compatibility for older imports.
- ASCII-safe logging, UTF-8 reconfiguration for Windows terminals.
"""

from __future__ import annotations
import os
import sys
import random
import logging
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import librosa

# ------------------------------------------------------------
# UTF-8 Safe Logging Setup
# ------------------------------------------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[audio_utils] %(message)s")
log = logging.getLogger("audio_utils")

# ------------------------------------------------------------
# ASCII-safe log wrapper for backward compatibility
# ------------------------------------------------------------
def log_safe(msg: str):
    """Backward-compatible safe logger used in older scripts."""
    try:
        log.info(str(msg))
    except Exception:
        # fallback to print in case logging fails
        print(f"[audio_utils] {msg}")

# ------------------------------------------------------------
# Core Audio I/O
# ------------------------------------------------------------
def load_audio(path: str, sr: int = 22050, mono: bool = True) -> Tuple[np.ndarray, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, orig_sr = sf.read(path, always_2d=False)
    if y.ndim > 1 and mono:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr, sr)
        orig_sr = sr
    return y.astype(np.float32), orig_sr


def save_wav(path: str, audio: np.ndarray, sr: int = 22050):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    log_safe(f"Saved WAV: {path}")


# Backward-compatible aliases
def load_wav(path: str, sr: int = 22050, mono: bool = True):
    return load_audio(path, sr, mono)

def read_wav(path: str, sr: int = 22050, mono: bool = True):
    return load_audio(path, sr, mono)

def write_wav(path: str, audio: np.ndarray, sr: int = 22050):
    return save_wav(path, audio, sr)

# ------------------------------------------------------------
# Basic audio processing
# ------------------------------------------------------------
def normalize_audio(audio: np.ndarray, target_db: float = -30.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2))
    if rms <= 0:
        return audio
    scalar = 10 ** (target_db / 20.0) / (rms + 1e-9)
    return np.clip(audio * scalar, -1.0, 1.0)

def trim_silence(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    except Exception as e:
        log.warning(f"trim_silence failed: {e}")
        return audio

# ------------------------------------------------------------
# Mel Spectrogram computation
# ------------------------------------------------------------
def compute_mel(audio: np.ndarray, sr: int = 22050,
                n_mels: int = 80, hop_length: int = 256,
                win_length: int = 1024, fmin: float = 0.0,
                fmax: Optional[float] = None, power: float = 2.0) -> np.ndarray:
    try:
        S = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=win_length,
            n_mels=n_mels, hop_length=hop_length,
            win_length=win_length, fmin=fmin, fmax=fmax, power=power
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db.astype(np.float32)
    except Exception as e:
        log.error(f"compute_mel failed: {e}")
        raise

def save_mel_spectrogram(path: str, mel_db: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, mel_db)
    log_safe(f"Saved mel spectrogram: {path}")

def load_mel_file(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mel file not found: {path}")
    return np.load(path)

# ------------------------------------------------------------
# Inverse transforms
# ------------------------------------------------------------
def mel_db_to_magnitude_db(mel_db: np.ndarray) -> np.ndarray:
    try:
        mel_power = librosa.db_to_power(mel_db)
        return mel_power
    except Exception as e:
        log.error(f"mel_db_to_magnitude_db failed: {e}")
        raise

def mel_power_to_waveform(mel_power: np.ndarray, sr: int = 22050,
                          n_iter: int = 32, hop_length: int = 256,
                          win_length: int = 1024) -> np.ndarray:
    try:
        y = librosa.feature.inverse.mel_to_audio(
            mel_power, sr=sr, n_iter=n_iter,
            hop_length=hop_length, win_length=win_length
        )
        return y.astype(np.float32)
    except Exception as e:
        log.error(f"mel_power_to_waveform failed: {e}")
        raise

def griffin_lim(S_mag: np.ndarray, sr: int = 22050, n_iter: int = 32,
                hop_length: int = 256, win_length: int = 1024) -> np.ndarray:
    try:
        y = librosa.griffinlim(S_mag, n_iter=n_iter,
                               hop_length=hop_length, win_length=win_length)
        return y.astype(np.float32)
    except Exception as e:
        log.error(f"griffin_lim failed: {e}")
        raise

# ------------------------------------------------------------
# Augmentation functions
# ------------------------------------------------------------
def spec_augment(mel: np.ndarray, time_mask_max_pct=0.1,
                 freq_mask_max_pct=0.15, num_time_masks=1,
                 num_freq_masks=1):
    """Apply SpecAugment to mel spectrogram."""
    if mel.ndim != 2:
        log.warning(f"spec_augment expected 2D mel, got shape {mel.shape}. Skipping.")
        return mel
    m = mel.copy()
    n_mels, n_time = m.shape
    for _ in range(num_freq_masks):
        f = int(random.uniform(0, freq_mask_max_pct) * n_mels)
        f0 = random.randint(0, max(0, n_mels - f))
        m[f0:f0+f, :] = 0
    for _ in range(num_time_masks):
        t = int(random.uniform(0, time_mask_max_pct) * n_time)
        t0 = random.randint(0, max(0, n_time - t))
        m[:, t0:t0+t] = 0
    return m

def augment_time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    try:
        return librosa.effects.time_stretch(audio, rate)
    except Exception as e:
        log.warning(f"augment_time_stretch failed: {e}")
        return audio

def augment_pitch_shift(audio: np.ndarray, sr: int, n_steps: int = 2) -> np.ndarray:
    try:
        return librosa.effects.pitch_shift(audio, sr, n_steps)
    except Exception as e:
        log.warning(f"augment_pitch_shift failed: {e}")
        return audio

def add_background_noise(audio: np.ndarray, noise_level_db: float = -30.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2)) + 1e-9
    noise_rms = rms * (10 ** (noise_level_db / 20.0))
    noise = np.random.normal(0.0, noise_rms, size=audio.shape)
    return np.clip(audio + noise, -1.0, 1.0)

# ------------------------------------------------------------
# Loss Helpers
# ------------------------------------------------------------
def compute_mse(pred, target):
    return ((pred - target) ** 2).mean()

def compute_l1(pred, target):
    return np.mean(np.abs(pred - target))

# ------------------------------------------------------------
# Smoke Test
# ------------------------------------------------------------
if __name__ == "__main__":
    log_safe("Running audio_utils smoke test...")
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
    mel = compute_mel(test_signal)
    mel_aug = spec_augment(mel)
    rec = mel_power_to_waveform(librosa.db_to_power(mel_aug))
    save_wav("logs/test_audio.wav", rec)
    log_safe("audio_utils smoke test complete.")
