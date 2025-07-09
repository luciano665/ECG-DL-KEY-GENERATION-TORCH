import os
import json
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import wandb


# 0. Initialize wandb client
#   - A training monitoring tool, visualization of loss

# Log into wandb library
#wandb.login(relogin=True)

# Init wandb to record results
wandb.init(
    entity="lumr0067-west-virginia-university",
    project="ECG-VIT-KEY-GENERATION",
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
        "dropout_rate": 0.1,
        "learning_rate": 1e-3
    }
)
config = wandb.config


# 1. Data Loader with rec_2_filtered Handling
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
        for dir_name in sorted(os.listdir(self.data_dir)):
            if not dir_name.startswith("Person_"): # skip invalid directories
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

    def _validate_datasets(self):
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
        attn_output, _ = self.attn(x, x, x)
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
        # Patch + Projection
        x = self.patch_embed(x)
        # Add positional info
        x = x + self.pos_embed
        # Transformer stack
        for block in self.transformer_blocks:
            x = block(x) # -> to each block
        x = x.permute(0, 2, 1) # -> (batch, embed_dim, num_patches) for pooling
        x = self.gap(x).squeeze(-1) # -> (batch, embed_dim)
        x = self.key_proj(x) # (batch, key_bits)
        return self.sigmoid(x)

# 3. Training and Key Generation System  with WandB
class KeyGenerationSystem:
    # Constructor
    def __init__(self, data_dir, key_path, device=None):
        # Init data loader
        self.loader = ECGKeyLoader(data_dir, key_path)
        # Determine compute device
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # Placeholder for model
        self.model = None

    def train(self, epochs=config.epochs, batch_size=config.batch_size, lr=config.learning_rate, patience=config.patience):
        # Get train/validation splits
        X_train, X_val, Y_train, Y_val = self.loader.get_train_data()
        print("X_train shape:", X_train.shape)
        print("X_val shape:", X_val.shape)
        print("Y_train shape:", Y_train.shape)
        print("Y_val shape:", Y_val.shape)

        # Instance model with correct key length
        key_bits = Y_train.shape[1] if Y_train.ndim > 1 else config.key_bits
        self.model = TransformerKeyGenerator(
            seq_len=config.seq_len,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            num_transformer_blocks=config.num_transformer_blocks,
            key_bits=key_bits,
            dropout_rate=config.dropout_rate
        )

        # Watch model with WandB to track gradients and parameters
        wandb.watch(self.model, log="all", log_freq=50)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss() # Binary cross-emtropy loss

        # Track best validation loss
        best_loss =  float('inf')
        epochs_no_improve = 0 # early stopping counter
        history = {'train_loss': [], 'val_loss': []} # record losses

        # Creation of Datasets
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        for ep in range(1, epochs + 1):
            # training loop
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                # Forward pass
                predictions = self.model(xb)
                # Compute loss
                loss = criterion(predictions, yb)
                # Backpropagation
                loss.backward()
                # Update of parameters
                optimizer.step()
                # Accumulate
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation loop
            self.model.eval() # set to eval mode
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb, in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += criterion(self.model(xb), yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)


            # Log metrics to WandB
            wandb.log({
                'epoch': ep,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            print(f"Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss # Update best
                best_state = self.model.state_dict()  # save weights
                epochs_no_improve = 0 # reset counter
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Stopping early at epoch {ep}")
                    break

        # Restore best weights
        self.model.load_state_dict(best_state)
        return history

    def generate_key(self, ecg_segments, threshold=0.5):
        if ecg_segments is None or len(ecg_segments) == 0:
            raise  ValueError("No ECG segment provided for key generation")
        array = np.array(ecg_segments, dtype=np.float32)
        if array.ndim == 2: # if missing channel dim
            array = array.reshape(array.shape[0], array.shape[1], 1) # add channels
        tensor = torch.from_numpy(array).to(self.device) # To tensor on device

        # Eval mode
        self.model.eval()
        with torch.no_grad():
            probs = self.model(tensor).cpu().numpy() # model output
        avg = probs.mean(axis=0) # average probabilities across segments
        # Threshold for final binary output
        return (avg > threshold).astype(np.int32)


# 4. Main Execution with Error Handling
if __name__ == '__main__':
    DATA_DIR = "/Users/lucianomaldonado/ECG_KEY-PYTORCH/segmented_ecg_data_torch_use"
    KEY_FILE = "/Users/lucianomaldonado/ECG_KEY-PYTORCH/GROUND_TRUTH_KEYS/secrets_random_keys_torch.json"

    try:
        print("Initializing system....")
        kgs = KeyGenerationSystem(DATA_DIR, KEY_FILE)

        print("\nStarting training....")
        kgs.train(epochs=config.epochs) # train model using WandB hyperparameters

        # Dummy forward to build model
        _ = kgs.model(torch.zeros((1, config.seq_len, 1), device=kgs.device))

        # ----------------------------
        # Test key generation for all persons
        # ----------------------------
        print("\nTesting key generation for all persons:")
        aggregated_keys = {} # store aggregated keys
        all_intra = [] # intra-person distances H

        for person in kgs.loader.persons:
            segments = person['segments']
            if segments.ndim == 2:
                segments = segments.reshape(segments.shape[0], segments.shape[1], 1)

            agg_key = kgs.generate_key(segments)
            ground_keys = person['key'].astype(np.int32)
            acc = np.mean(agg_key == ground_keys) # accuracy
            print(f"\nPerson {person['id']}: Aggregated Key Accuracy: {acc:.2%}")
            aggregated_keys[person['id']] = agg_key

            # compute intra-person Hamming distances
            with torch.no_grad():
                preds = (kgs.model(torch.from_numpy(segments).to(kgs.device)).cpu().numpy() > 0.5).astype(int)
            dists = []
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    dists.append(np.sum(preds[i] != preds[j]))
            if dists:
                m, s = np.mean(dists), np.std(dists)
                print(f"  Intra-person avg Hamming distance: {m:.2f} ± {s:.2f} bits")
                all_intra += dists
            else:
                print("  Not enough segments to compute intra-person HD.")

        if all_intra:
            print(f"\nOverall Intra-person HD: mean={np.mean(all_intra):.2f}, std={np.std(all_intra):.2f}")
        else:
            print("\nNo intra-person HD data.")

        # Inter-person Hamming distances
        print("\nInter-person Hamming distances:")
        ids = sorted(aggregated_keys)
        inter = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = int(np.sum(aggregated_keys[ids[i]] != aggregated_keys[ids[j]]))
                inter.append(d)
                print(f"  {ids[i]} vs {ids[j]}: {d} bits")
        if inter:
            print(f"\nOverall Inter-person HD: mean={np.mean(inter):.2f}, std={np.std(inter):.2f}")
        else:
            print("\nNo inter-person HD data.")

        # save raw distance arrays
        with open("all_intra_distances.pkl", "wb") as f:
            pickle.dump(all_intra, f)
        with open("person_inter_dists.pkl", "wb") as f:
            pickle.dump(inter, f)

    except Exception as e:
        print(f"\nError: {e}")
        print("Verification Checklist:")
        print("1. Directory structure: Person_XX/rec_N_filtered/*.csv")
        print("2. CSV files contain exactly 170 values, no headers")
        print("3. JSON keys match Person_XX numbering (1-89)")
        print("4. Minimum 10 segments across all persons")
        print("Something went wrong; please verify your paths and data.")




