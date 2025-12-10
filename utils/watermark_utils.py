# File: ECHO_TWIN_CORE/utils/watermark_utils.py
# Purpose: Handles audio watermark embedding and verification for Echo Twin Core.
# Compatible with Python 3.11 and updated audio_utils.py structure.

import os
import numpy as np
import logging
from scipy.signal import correlate

from utils.audio_utils import load_wav, save_wav

log = logging.getLogger("watermark_utils")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# -----------------------------
# Configurable Parameters
# -----------------------------
DEFAULT_WATERMARK_FREQ = 19000  # inaudible high frequency (Hz)
DEFAULT_WATERMARK_DB = -40      # low-level watermark strength (dB)
DEFAULT_SAMPLE_RATE = 22050
WATERMARK_DURATION = 1.0        # seconds

# -----------------------------
# Core Functions
# -----------------------------
def generate_watermark(sr=DEFAULT_SAMPLE_RATE, freq=DEFAULT_WATERMARK_FREQ, db=DEFAULT_WATERMARK_DB):
    """Generate a low-amplitude sine-wave watermark."""
    t = np.linspace(0, WATERMARK_DURATION, int(sr * WATERMARK_DURATION), endpoint=False)
    watermark = np.sin(2 * np.pi * freq * t)
    rms = np.sqrt(np.mean(watermark ** 2))
    scalar = 10 ** (db / 20.0) / (rms + 1e-9)
    return watermark * scalar


def embed_watermark(input_path, output_path=None, sr=DEFAULT_SAMPLE_RATE, freq=DEFAULT_WATERMARK_FREQ):
    """Embed watermark into an audio file."""
    try:
        audio, sr = load_wav(input_path, sr=sr)
        watermark = generate_watermark(sr, freq)
        padded_wm = np.zeros_like(audio)
        L = min(len(audio), len(watermark))
        padded_wm[:L] = watermark[:L]
        combined = np.clip(audio + padded_wm, -1.0, 1.0)

        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = base + "_wm" + ext

        save_wav(output_path, combined, sr)
        log.info(f"✅ Watermark embedded successfully: {output_path}")
        return output_path
    except Exception as e:
        log.error(f"❌ Failed to embed watermark: {e}")
        raise


def extract_watermark(input_path, sr=DEFAULT_SAMPLE_RATE, freq=DEFAULT_WATERMARK_FREQ):
    """Extract watermark correlation signal from audio."""
    try:
        audio, sr = load_wav(input_path, sr=sr)
        watermark = generate_watermark(sr, freq)
        corr = correlate(audio, watermark, mode="valid")
        return corr
    except Exception as e:
        log.error(f"❌ Failed to extract watermark: {e}")
        raise


def verify_watermark(input_path, sr=DEFAULT_SAMPLE_RATE, freq=DEFAULT_WATERMARK_FREQ, threshold=0.3):
    """Verify if a watermark is present in the given audio."""
    try:
        corr = extract_watermark(input_path, sr=sr, freq=freq)
        peak = np.max(np.abs(corr))
        norm_corr = peak / (np.linalg.norm(corr) + 1e-9)
        presence = norm_corr > threshold
        log.info(f"🔍 Watermark detection strength: {norm_corr:.4f}")
        if presence:
            log.info("✅ Watermark verification: PASSED")
        else:
            log.warning("⚠️ Watermark verification: FAILED")
        return presence, norm_corr
    except Exception as e:
        log.error(f"❌ Watermark verification failed: {e}")
        return False, 0.0


# -----------------------------
# Test Entry Point
# -----------------------------
if __name__ == "__main__":
    test_path = "../audio/001.wav"
    if os.path.exists(test_path):
        wm_out = embed_watermark(test_path)
        verify_watermark(wm_out)
    else:
        log.info("⚠️ No test file found in ../audio/")
