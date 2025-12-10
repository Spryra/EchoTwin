# File: D:\Echo_Twin_Core\utils\__init__.py
"""
Echo Twin - utils/__init__.py (Full replacement, 2025-11-06)

Purpose:
- Expose all public utility functions safely.
- Prevent ImportError when imported by scripts/train_model or validate_mels.
- Ensures backwards compatibility for augment_time_stretch, spec_augment, etc.

Changelog:
- 2025-11-06: Fixed ImportError caused by missing augment_time_stretch export.
- Preserves all symbols used by other modules.
- ASCII-safe, no emoji.
"""

from .audio_utils import (
    load_audio,
    save_wav,
    load_wav,
    read_wav,
    write_wav,
    normalize_audio,
    trim_silence,
    compute_mel,
    save_mel_spectrogram,
    load_mel_file,
    mel_db_to_magnitude_db,
    mel_power_to_waveform,
    griffin_lim,
    augment_time_stretch,
    augment_pitch_shift,
    add_background_noise,
    spec_augment,
    compute_mse,
    compute_l1,
)

__all__ = [
    "load_audio",
    "save_wav",
    "load_wav",
    "read_wav",
    "write_wav",
    "normalize_audio",
    "trim_silence",
    "compute_mel",
    "save_mel_spectrogram",
    "load_mel_file",
    "mel_db_to_magnitude_db",
    "mel_power_to_waveform",
    "griffin_lim",
    "augment_time_stretch",
    "augment_pitch_shift",
    "add_background_noise",
    "spec_augment",
    "compute_mse",
    "compute_l1",
]
