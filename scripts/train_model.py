# File: D:\Echo_Twin_Core\scripts\train_model.py
"""
Echo Twin - scripts/train_model.py (improved stability edition)
----------------------------------------------------------------
Changelog:
- 2025-11-06: Added Adam optimizer (NumPy) and hybrid loss (MSE+L1).
- Integrated SpecAugment from audio_utils.
- Keeps full compatibility with prior training setup, paths, logs, and checkpoints.
- ASCII-safe logging, no emojis.
"""

from __future__ import annotations
import os
import sys
import json
import logging
import numpy as np
from glob import glob
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_architecture import (
    init_weights, forward, backward, save_checkpoint
)
from utils.audio_utils import (
    load_mel_file, compute_mse, compute_l1, spec_augment, log_safe
)

# ----------------------------------------------------------
# Logging setup
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[train_model] %(message)s")
log = logging.getLogger("train_model")

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "settings.json")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file missing: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

MEL_DIR = os.path.join(BASE_DIR, "data", "mels")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPOCHS = SETTINGS.get("training", {}).get("epochs", 50)
LR = SETTINGS.get("training", {}).get("learning_rate", 0.001)
BATCH_SIZE = SETTINGS.get("training", {}).get("batch_size", 64)
HIDDEN_DIM = SETTINGS.get("model", {}).get("hidden_dim", 128)

# ----------------------------------------------------------
# Adam Optimizer (NumPy)
# ----------------------------------------------------------
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ----------------------------------------------------------
# Load mel data
# ----------------------------------------------------------
mel_files = sorted(glob(os.path.join(MEL_DIR, "*.npy")))
if not mel_files:
    raise RuntimeError(f"No mel spectrograms found in {MEL_DIR}")

log_safe(f"Found {len(mel_files)} mel spectrograms for training.")

mel_data = []
for mf in mel_files:
    mel = load_mel_file(mf)
    mel_data.append(mel.T)
mel_data = np.concatenate(mel_data, axis=0).astype(np.float32)
log_safe(f"Training samples shape: {mel_data.shape}")

# ----------------------------------------------------------
# Initialize model
# ----------------------------------------------------------
input_dim = mel_data.shape[1]
params = init_weights(input_dim, HIDDEN_DIM)
optimizer = Adam(params, lr=LR)
log_safe(f"Model initialized (input_dim={input_dim}, hidden_dim={HIDDEN_DIM})")

# ----------------------------------------------------------
# Training Loop
# ----------------------------------------------------------
num_batches = len(mel_data) // BATCH_SIZE
log_safe(f"Starting training for {EPOCHS} epochs using Adam optimizer")

for epoch in range(1, EPOCHS + 1):
    indices = np.random.permutation(len(mel_data))
    epoch_loss = 0.0

    for i in tqdm(range(num_batches), desc=f"Epoch {epoch}/{EPOCHS}", ncols=80):
        batch_idx = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch = mel_data[batch_idx]

        # SpecAugment - Skip for frame-level training
        # The training uses individual mel frames (1D: n_mels), not full spectrograms (2D: n_mels x time)
        # SpecAugment requires 2D spectrograms, so we skip it for frame-level training
        # If you want augmentation, you'd need to train on full spectrograms instead of frames
        # batch = np.stack([spec_augment(b.T).T for b in batch])  # Disabled for frame-level training

        pred, h = forward(params, batch)
        loss_mse = compute_mse(pred, batch)
        loss_l1 = compute_l1(pred, batch)
        loss = 0.5 * loss_mse + 0.5 * loss_l1

        grads = backward(params, batch, pred, h, batch)
        optimizer.step(params, grads)
        epoch_loss += loss

    epoch_loss /= max(1, num_batches)
    log_safe(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {epoch_loss:.6f}")

    if epoch % 5 == 0 or epoch == EPOCHS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.npz")
        save_checkpoint(params, ckpt_path)
        log_safe(f"Saved checkpoint: {ckpt_path}")

final_path = os.path.join(CHECKPOINT_DIR, "model_latest.npz")
save_checkpoint(params, final_path)
log_safe(f"Training complete. Final model saved at: {final_path}")
