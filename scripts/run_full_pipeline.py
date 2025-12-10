# File: D:\Echo_Twin_Core\scripts\run_full_pipeline.py
"""
Echo Twin - scripts/run_full_pipeline.py (Extended: TTS Integration v2)
-----------------------------------------------------------------------
Purpose:
    Execute the full Echo Twin workflow sequentially:
        1. Preprocess raw audio → mel spectrograms
        2. Train model on mel data
        3. Evaluate model to reconstruct waveform
        4. Run TTS inference for text generation

Changelog (2025-11-06):
    - Added TTS inference stage (scripts.tts_infer).
    - Added robust subprocess isolation and duration timing.
    - Preserves all prior error handling and logging style.
    - Fully ASCII-safe (no emoji) and Windows-compatible.

Usage:
    python -m scripts.run_full_pipeline
"""

from __future__ import annotations
import os
import sys
import subprocess
import logging
import time

# ----------------------------------------------------------
# Logging setup
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[pipeline] %(message)s")
log = logging.getLogger("pipeline")

# ----------------------------------------------------------
# Project paths
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ----------------------------------------------------------
# Helper to run subprocess safely
# ----------------------------------------------------------
def run_step(command: list[str], step_name: str):
    log.info(f"--- Starting step: {step_name} ---")
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable] + command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        duration = time.time() - start_time

        if result.returncode == 0:
            log.info(f"[{step_name}] Completed successfully in {duration:.2f}s")
            if result.stdout.strip():
                log.debug(result.stdout.strip())
        else:
            log.error(f"[{step_name}] Failed with code {result.returncode}")
            if result.stderr.strip():
                log.error(result.stderr.strip())
            raise subprocess.CalledProcessError(result.returncode, command)

    except Exception as e:
        log.error(f"[{step_name}] Unexpected error: {e}")
        raise

# ----------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------
def main():
    log.info("==============================================")
    log.info(" Echo Twin - Full Pipeline Execution Started ")
    log.info("==============================================")
    log.info(f"Working directory: {BASE_DIR}")

    # Sequential pipeline steps
    steps = [
        (["-m", "data.preprocess"], "Preprocessing Audio Data"),
        (["-m", "scripts.train_model"], "Training Model"),
        (["-m", "scripts.evaluate"], "Evaluating Model"),
    ]

    for cmd, name in steps:
        run_step(cmd, name)

    # ------------------------------------------------------
    # TTS Inference Stage
    # ------------------------------------------------------
    log.info("--- Starting step: Text-to-Speech Inference ---")
    try:
        # You can change this phrase anytime
        tts_text = "This is a voice test by Echo Twin using the full pipeline."
        tts_command = ["-m", "scripts.tts_infer", tts_text]
        result = subprocess.run(
            [sys.executable] + tts_command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        if result.returncode == 0:
            log.info("[TTS Inference] Completed successfully.")
            if result.stdout.strip():
                log.debug(result.stdout.strip())
        else:
            log.error(f"[TTS Inference] Failed with code {result.returncode}")
            if result.stderr.strip():
                log.error(result.stderr.strip())

    except Exception as e:
        log.error(f"[TTS Inference] Unexpected error: {e}")

    log.info("==============================================")
    log.info(" Full Pipeline completed successfully. ")
    log.info(" Outputs saved under: /logs/evaluations and /logs/evaluations_tts ")
    log.info("==============================================")

# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
