import os
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.fx.experimental.unification.unification_tools import first
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Loader: ECGKeyLoader
#    - Reads ECG segments and associated ground-truth keys
#    - Filters directories and files, validates segment length
#    - Splits data into training and validation sets
# ─────────────────────────────────────────────────────────────────────────────
class ECGKeyLoader:
    """
        Handles loading ECG signal segments and their corresponding ground truth Key (256-bits)
    """
    def __init__(self, data_dir, key_path):
        # Directory per-person ECG CSV files
        self.data_dir = data_dir
        # Load Ground Keys into dict -> {Person_id : [key]}
        self.key_map = self._load_keys(key_path)
        # Parse folder structure to gather segments per person
        self.persons = self._load_persons()
        # Ensuring minimal requirements for the dataset are met
        self._validate_dataset()

    def _load_keys(self, key_path):
        """
        Load JSON file of random binary keys
        Format: {Person_01: []...}
        Convert intp a dict [int, np.ndarray] for fast lookup.
        """
        with open(key_path) as f:
            raw = json.load(f)

        # Extract the ID from the Key name and convert values to floats32 array
        return {
            int(k.split("_")[-1]): np.array(v, dtype=np.float32)
            for k, v in raw.items()
        }

    def _load_persons(self):
        """
        Traverse data_dir, locate "Person_<ID>" folders,
        load valid segments, skip if no key or segments is invalid.
        Returns a list of dicts with keys:
            - 'id': person ID
            - 'segments': np.ndarray of shape (N_segments, seq_len)
            - 'key': ground-truth key vector
        """
        persons = []
        valid_ids = set(self.key_map.keys())
        for dir_name in sorted(os.listdir(self.data_dir)):
            # Only consider dirs starting with 'Person_'
            if not dir_name.startswith("Person_"):
                continue

            # Parse the integer ID, handle leading 0
            try:
                pid = int(dir_name.split('_')[-1].lstrip('0'))
            except ValueError:
                print(f"Skipping in valid directory: {dir_name}")
                continue

            # Skip if no valid key exit for this person
            if pid not in valid_ids:
                print(f"No key for {dir_name}, skipping")
                continue

            # Load ECG segments form nested CSV files
            segments = self._load_segments(os.path.join(self.data_dir, dir_name))
            #Skip if no segments loaded
            if len(segments) == 0:
                print(f"No valid segments in {dir_name}, skipping")
                continue

            # Append structured info
            persons.append({
                'id': pid,
                'segments': segments,
                'key': self.key_map[pid]
            })
            print(f"Loaded {len(segments)} segments from {dir_name}")

        return persons

    def _validate_dataset(self):
        """
        Ensure we have at least one person and >= 20 total segments.
        """
        if not self.persons:
            raise  ValueError("No valid persons with both keys and ECG segments")
        total = sum(len(p['segments']) for p in self.persons)
        print(f"Dataset:  {len(self.persons)} persons, {total} total segments")
        if total < 10:
            raise ValueError("Need ≥10 valid segments for training")

    def _load_segments(self, person_path, seq_len=170):
        """
        Read all .csv files under person_path/recording_n,
        skip row header for each CSV file, verify each segment equals = seq_len
        Returns np.ndarray shape (N_segments, seq_len)
        """
        segments = []
        for root, _, files in os.walk(person_path):
            for file_name in files:
                if not file_name.endswith(".csv"):
                    continue
                file_path = os.path.join(root, file_name)
                try:
                    # Skip header row since CSVs have empty header row
                    arr = np.loadtxt(file_path, delimiter=',', skiprows=1)
                    # Check shape integrity
                    if arr.ndim != 1 or len(arr) != seq_len:
                        print(f"Invalid ECG in {file_path}")
                        continue
                    segments.append(arr.astype(np.float32))
                except Exception as e:
                    print(f" Error loading {file_path}: {e}")

        return np.array(segments)

    def get_train_data(self, test_size=0.2):
        """
        Aggregate X (segments), y (binary keys), and ids (for stratification).
        Reshape X to (N, seq_len, 1) and return sklearn train_test_split
        """
        X, y, ids = [], [], []
        for p in self.persons:
            X.extend(p['segments'])
            y.extend([p['key']] * len(p['segments']))
            ids.extend([p['id']] * len(p['segments']))

        # Convert to arrays and reshape
        X = np.array(X).reshape(-1, 170, 1)
        y = np.array(y)

        if len(X) < 2:
            raise ValueError(f"Need >= 2 samples, got {len(X)}")

        # Stratified split to maintain class distribution (by person)
        return train_test_split(X, y, test_size=test_size, stratify=ids)


# -----------------------------------------------------------------------------
# 2. Model Definition: BioKeyTransformer
#    - 1D CNN encoder to extract features from ECG
#    - Multi-head self-attention to learn temporal dependencies
#    - LayerNorm + residual for stability
#    - Global average pooling → binary key projection
# -----------------------------------------------------------------------------
class BioKeyTransformer(nn.Module):
    """
    Hybrid CNN-Transformer for ECG-based binary key generation
    Input: (batch, seq_len, 1)
    Output: (batch, key_bits) with values 0/1
    """
    def __init__(self, key_bits=256):
        super().__init__()
        # Convolutional stack: progressively increase channel depth
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=15, padding=7), nn.ReLU(),
            nn.Conv1d(64, 128, 10, padding=5), nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=2), nn.ReLU()
        )
        # Self-Attention: embed_dim=256, num_heads=8
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        # Post-attention normalization + residual
        self.norm = nn.LayerNorm(256)
        # Linear layer to map pooled features to key bits
        self.key_proj = nn.Linear(256, key_bits)
        # Sigmoid to obtain probabilities for each bit
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, 1) -> permute for Conv1D: (bacth, 1, seq_len)
        x = x.permute(0, 2, 1)
        # Convolutional feature extraction
        x = self.conv(x)        # -> (batch, 256, seq_len)
        # Back to (batch, seq_len, 256) for attention
        x = x.permute(0, 2, 1)
        # Self-attention (query=key=value)
        attn_out, _ = self.attn(x, x, x)
        # Normalization layer , Add & norm (residual connection)
        x = self.norm(x + attn_out)
        # Global average pooling along time dimension
        x = x.mean(dim=1)
        # Map the key bits and apply sigmoid
        return  self.sigmoid(self.key_proj(x))





