import os
import json
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import wandb

# ─────────────────────────────────────────────────────────────────────────────
# 0. Initialize wandb client
#   - A training monitoring tool, visualization of loss
# ─────────────────────────────────────────────────────────────────────────────

# Log into wandb library
wandb.login(relogin=True)

# Init wandb to record results
wandb.init(
    entity="lumr0067-west-virginia-university",
    project="ecg-key_generation",
    config={
        "epochs": 100,
        "batch_size": 32,
        "patience": 10,
        "seq_len": 170,
        "patch_size": 10,
        "embed_dim": 64,
        "num_heads": 8,
        "mlp_dim": 128,
        "num_transformer_blocks": 4,
        "key_bits": 256,
        "dropout_rate": 0.1
    }
)
config = wandb.config


# =====================================================================
# 1. Data Loader with rec_2_filtered Handling (mirrors TF version logic)
# =====================================================================
class ECGKeyLoader:
    # Constructor
    def __init__(self, data_dir, key_path):
        self.data_dir = data_dir # root folder with 'Person_XX'
        self.key_map = self._load_keys(key_path) # load ground-truth keys
        self.persons = self._load_persons() # load ECG segments per person
        self._validate_datasets() # ensure there is enough data

    def _load_keys(self, key_path):
        # Open JSON file with ground-truth keys
        with open(key_path) as f:
            raw = json.load(f) # parse JSON content
            # Convert keys into IDs and vals to float32 np-arrays
        return {int(k.split("_")[-1]): np.array(v, dtype=np.float32) for k , v in raw.items()}

    def _load_persons(self):
        persons = [] # array for each person's data
        valid_ids = set(self.key_map.keys())

        # loop through entries
        for dir_name in sorted(os.listdir(self.data_dir))
            if not dir_name.startswith("Person_") # skip invalid directories
                continue
            try:
                # Extract ID and handle extra 0's
                pid = int(dir_name.split('_')[-1].lstrip('0'))
                if pid == 0:
                    # Case of Person_00
                    pid = int(dir_name.split('_')[-1])
            except ValueError:
                print(f"Skipping invalid directory {dir_name}")
                continue

            if pid not in  valid_ids: # Skip if no key exists
                print(f"No key found for {dir_name}, skipping")
                continue

            person_path = os.path.join(self.data_dir, dir_name) # Full path
            segments = self._load_segments(person_path) # load .csv files (ECG-segments)
            if len(segments) == 0: # Skip invalid segments -> (MAY ADD CONDITION < 170)
                print(f"No valid segments in {dir_name}, skipping.")
                continue

            # store ID, segments arrays, and associated key
            persons.append({
                'id': pid,
                'segments': segments,
                'key': self.key_map[pid]
            })
            print(f"Loaded {len(segments)} segments from {dir_name}")

        return persons   # list of dictionaries

    def _validate_dataset(self):
        # At least one person must exist
        if not self.persons:
            raise  ValueError("No valid persons with both keys and ECG segments found")
        total_segments = sum(len(p['segments']) for p in self.persons) # Count of all segments loaded
        print(f"Dataset contains {len(self.persons)} persons with {total_segments} segments.")
        if total_segments < 10: # At least >= 10 segments required
            raise ValueError("Insufficient data for training (<10).")

    def _load_segments(self, person_path, seq_len=170):
        # List to store valid segments-arrays
        segments = []

        # Recursive traversal
        for root, dirs, files in os.walk(person_path):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                file_path = os.path.join(root, file)
                try:
                    ecg = np.loadtxt(file_path, delimiter=',', skiprows=1) # Skip header row
                    # Check to ensure correct shape
                    if ecg.ndim != 1 or len(ecg) != seq_len:
                        print(f"Invalid ECG format in {file_path}")
                        continue
                    segments.append(ecg.astype(np.float32)) # Store valid segment
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        return np.array(segments)

    def get_train_data(self, test_size=0.2):
        X, Y, ids = [], [], []
        for p in self.persons:
            X.extend(p['segments']) # add all segments
            Y.extend([p['key']] * len(p['segments'])) # Repeat the person's key n-times
            ids.extend([p['id']] * len(p['segments'])) # Person ID for stratify

        X = np.array(X).reshape(-1, config.seq_len, 1) # shape: (N, seq_len, 1)
        Y = np.array(Y) # shape: (N, key_bits)
        if len(X) < 2:
            raise ValueError(f"Need at least 2 samples, got {len(X)}") # sanity check

        # Perform stratified train/test split
        return train_test_split(X, Y, test_size=test_size, stratify=ids)

# =====================================================================
# 2. Transformer-Based Model for ECG Key Generation (PyTorch)
# =====================================================================
class TransformerBlock(nn.Module):
    # Constructor
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super().__init__() # Init base module
        # Self-attention
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        # 1st Dropout layer
        self.dropout1 = nn.Dropout(dropout_rate)
        # 1st Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        # 1st MLP layer
        self.dense1 = nn.Linear(embed_dim, mlp_dim)
        # Activation layer
        self.activation = nn.ReLU()
        # 2nd MLP layer
        self.dense2 = nn.Linear(mlp_dim, embed_dim)
        # 2nd Dropout layer
        self.dropout2 = nn.Dropout(dropout_rate)
        # 2nd Layer normalization
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x): # x shape: (batch, seq_len, embed_dim)
        # Sell-attention mechanism
        attn_output = self.attn(x, x, x)
        # Apply dropout
        attn_output = self.dropout1(attn_output)
        # Residual + layer-norm
        out1 = self.norm1(x + attn_output)
        # Feed-forward part-1
        ffn = self.activation(self.dense1(out1))
        # Feed-forward part-2
        ffn = self.dense2(ffn)
        # Apply dropout
        ffn = self.dropout2(ffn)
        # Residual + layer-norm
        return self.norm2(out1 + ffn)

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        # Length of each patch
        self.patch_size = patch_size
        # 1D conv to project patched to embedding dim
        self.proj = nn.Conv1d(in_channels=1,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              padding=0)

    def forward(self, x): # x shape: (batch, seq_len, channels)
        # Check if missing channel dim
        if x.ndim == 2:
            x = x.unsqueeze(-1) # add channel=1
        x = x.permute(0, 2, 1) # -> (batch, channels, seq_len) for CONV
        x = self.proj(x) # -> (batch, embed_dim, num_patches)
        x = x.permute(0, 2, 1) # -> (batch, num_patches, embed_dim)
        return x

class TransformerKeyGenerator(nn.Module):
    # Constructor
    def __init__(self, seq_len=170, patch_size=10, embed_dim=64,
                 num_heads=4, mlp_dim=128, num_transformer_blocks=4,
                 key_bits=256, dropout_rate=0.1):
        super().__init__()
        # Input seq_len
        self.seq_len = seq_len
        # Patch len
        self.patch_size = patch_size
        # Embedding dim
        self.embed_dim = embed_dim
        # Num of patches
        self.num_patches = seq_len // patch_size
        # Length of output key
        self.key_bits = key_bits

        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        # Positional embeddings parameter
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        # Stack of transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ])
        # Global ave pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        # Final Projection
        self.key_proj = nn.Linear(embed_dim, key_bits)
        # Sigmoid for binary output (thresholding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x shape: (batch, seq_len, channels)




