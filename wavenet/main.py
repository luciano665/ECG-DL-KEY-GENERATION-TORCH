import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import wandb

# ─────────────────────────────────────────────────────────────────────────────
# 0. Initialize Weights & Biases for experiment tracking
# ─────────────────────────────────────────────────────────────────────────────
wandb.init(
    entity="lumr0067-west-virginia-university",
    project="ECG-WAVENET-KEY-GENERATION",
    config={
        # Training Parameters
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "patience": 15,
        # Data Parameters
        "seq_len": 170,
        # Wavenet architecture params
        "num_filters": 64,
        "num_wavenet_blocks": 3,
        "kernel_size": 3,
        "dropout_rate": 0.1,
        # Output key size
        "key_bits": 256

    }
)
config = wandb.config

