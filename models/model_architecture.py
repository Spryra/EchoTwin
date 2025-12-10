# File: ECHO_TWIN_CORE/models/model_architecture.py
# Purpose: Minimal MLP autoencoder implemented with NumPy (no PyTorch).
# We train a small feed-forward autoencoder to reconstruct mel-frame vectors.
# Model parameters are NumPy arrays saved/loaded as .npz files to models/checkpoints.

import numpy as np
import os

def init_weights(input_dim, hidden_dim, seed=42):
    rng = np.random.RandomState(seed)
    W1 = rng.normal(0, 0.1, (hidden_dim, input_dim)).astype(np.float32)
    b1 = np.zeros((hidden_dim, ), dtype=np.float32)
    W2 = rng.normal(0, 0.1, (input_dim, hidden_dim)).astype(np.float32)
    b2 = np.zeros((input_dim,), dtype=np.float32)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward(params, x):
    """
    x: shape (batch, input_dim)
    returns reconstruction shape (batch, input_dim)
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    h = np.tanh(x.dot(W1.T) + b1)  # (batch, hidden)
    out = h.dot(W2.T) + b2
    return out, h

def sgd_update(params, grads, lr):
    params["W1"] -= lr * grads["W1"]
    params["b1"] -= lr * grads["b1"]
    params["W2"] -= lr * grads["W2"]
    params["b2"] -= lr * grads["b2"]

def backward(params, x, pred, h, target):
    """
    Compute gradients for autoencoder with mean squared error loss.
    """
    batch = x.shape[0]
    # dLoss/dOut = 2*(pred - target) / N
    dOut = 2.0 * (pred - target) / (batch * x.shape[1])
    # grads for W2,b2
    gradsW2 = dOut.T.dot(h)  # (input_dim, hidden_dim)
    gradsb2 = dOut.sum(axis=0)
    # backprop through tanh: dh = dOut.dot(W2) * (1 - h^2)
    dh = dOut.dot(params["W2"]) * (1.0 - h**2)
    gradsW1 = dh.T.dot(x)
    gradsb1 = dh.sum(axis=0)
    grads = {"W1": gradsW1.astype(np.float32), "b1": gradsb1.astype(np.float32),
             "W2": gradsW2.astype(np.float32), "b2": gradsb2.astype(np.float32)}
    return grads

def save_checkpoint(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, W1=params["W1"], b1=params["b1"], W2=params["W2"], b2=params["b2"])

def load_checkpoint(path):
    npz = np.load(path)
    return {"W1": npz["W1"].astype(np.float32), "b1": npz["b1"].astype(np.float32),
            "W2": npz["W2"].astype(np.float32), "b2": npz["b2"].astype(np.float32)}
