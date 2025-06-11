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

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────
class ECGKeyLoader:
    def __init__(self, data_dir, key_path):
        # Store data directory and load JSON file of ground-truth keys
        self.data_dir = data_dir
        self.key_map = self._load_keys(key_path)
        # Walk through each Person_XX folder and pull in valid ECG segments
        self.persons = self._load_persons()
        # Ensure that at least 10 segments total exist across all persons
        self._validate_dataset()

    def _load_keys(self, key_path):
        # Read the JSON of per-person 256-bit keys, convert tp float32 np-arrays
        with open(key_path) as f:
            raw = json.load(f)
        return {
            int(k.split("_")[-1]): np.array(v, dtype=np.float32)
            for k, v in raw.items()
        }

    def _load_persons(self):
        persons = []
        valid_ids = set(self.key_map.keys())

        for d in sorted(os.listdir(self.data_dir)):
            if not d.startswith("Person_"):
                continue
            # Extract the ID, handling leading zeros
            try:
                pid = int(d.split("_")[-1].lstrip("0")) or int(d.split("_")[-1])
            except ValueError:
                continue
            if pid in valid_ids:
                continue

            # Gather that person's valid 170-sample ECG segments
            segments = self._load_segments(os.path.join(self.data_dir, d))
            if len(segments) == 0:
                continue

            persons.append({
                "id": pid,
                "segments": segments,
                "key": self.key_map[pid]
            })
            print(f"Loaded {len(segments)} segments from {d}")

        return persons

    def _validate_dataset(self):
        # Raise if no person of fewer than 10 segments total
        if not self.persons:
            raise ValueError("No valid persons with ECG data found")
        total = sum(len(p["segments"]) for p in self.persons)
        print(f"Dataset contains {len(self.persons)} persons with {total} total segments.")
        if total < 10:
            raise ValueError("Insufficient data available for training")

    def _load_segments(self, person_path, seq_len=170):
        segments = []
        # Recursively scan CSV files under each recording folder
        for root, _, files in os.walk(person_path):
            for fn in files:
                if not fn.endswith(".csv"):
                    continue
                file_path = os.path.join(root, fn)
                try:
                    # Skip header row, enforce exactly seq_len points
                    ecg = np.loadtxt(file_path, delimiter=",", skiprows=1, ndmin=1)
                    if ecg.ndim != 1 or len(ecg) != seq_len:
                        continue
                    segments.append(ecg.astype(np.float32))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        return np.array(segments)

    def get_train_data(self, test_size=0.2):
        # Aggregate all segments & keys, stratify split by person ID
        X, Y, ids = [], [], []
        for p in self.persons:
            X.extend(p["segments"])
            Y.extend([p["key"]] * len(p["segments"]))
            ids.extend([p["id"]] * len(p["segments"]))

        X = np.array(X).reshape(-1, config.seq_len, 1)
        Y = np.array(Y)
        return train_test_split(X, Y, train_size=test_size, stratify=ids)

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. WaveNet Residual Block in PyTorch
    # ─────────────────────────────────────────────────────────────────────────────