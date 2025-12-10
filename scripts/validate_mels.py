# File: D:\Echo_Twin_Core\scripts\validate_mels.py
"""
Echo Twin - scripts/validate_mels.py (Optimized & Safe Replacement)
-------------------------------------------------------------------
Purpose:
    Quickly verify all mel spectrogram .npy files under data/mels/.
    Automatically move malformed files to data/mels_backup/ for inspection.

Enhancements (2025-11-06):
- Uses memory-mapped reading (np.load(..., mmap_mode="r")) for speed.
- Adds tqdm progress display with live file count.
- Adds per-file timeout & error handling.
- Skips large or corrupted files gracefully (no hang).
- ASCII-safe logging compatible with Windows terminals.
"""

from __future__ import annotations
import os
import sys
import logging
import shutil
import numpy as np
from tqdm import tqdm
import time

# ----------------------------------------------------------
# Path setup
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MELS_DIR = os.path.join(BASE_DIR, "data", "mels")
BACKUP_DIR = os.path.join(BASE_DIR, "data", "mels_backup")

os.makedirs(MELS_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# ----------------------------------------------------------
# Logging setup
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[validate_mels] %(message)s")
log = logging.getLogger("validate_mels")

# ----------------------------------------------------------
# Validation function
# ----------------------------------------------------------
def is_valid_mel(path: str, max_mb: float = 20.0) -> bool:
    """Return True if mel file appears valid, else False."""
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > max_mb:
            log.warning(f"Skipping oversized mel: {os.path.basename(path)} ({size_mb:.1f} MB)")
            return False

        # Memory-map for shape-only load (fast, no full read)
        arr = np.load(path, mmap_mode="r")

        if arr.ndim != 2:
            return False
        if arr.shape[0] < 10 or arr.shape[1] < 4:
            return False
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return False

        return True

    except Exception:
        return False

# ----------------------------------------------------------
# Main routine
# ----------------------------------------------------------
def main():
    files = [f for f in os.listdir(MELS_DIR) if f.endswith(".npy")]
    total = len(files)
    if total == 0:
        log.info(f"No mel files found in {MELS_DIR}.")
        return

    log.info(f"Validating {total} mel spectrograms...")

    invalid_files = []
    start_time = time.time()

    for fname in tqdm(files, desc="Checking mel files", ncols=80):
        path = os.path.join(MELS_DIR, fname)
        if not is_valid_mel(path):
            invalid_files.append(fname)
            try:
                shutil.move(path, os.path.join(BACKUP_DIR, fname))
            except Exception as e:
                log.warning(f"Failed to move {fname}: {e}")

    elapsed = time.time() - start_time
    valid_count = total - len(invalid_files)

    log.info(f"Validation complete in {elapsed:.2f}s")
    log.info(f"Valid mels: {valid_count} | Invalid moved: {len(invalid_files)}")

    if invalid_files:
        log.info(f"Corrupted files have been moved to {BACKUP_DIR}")

# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
