import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import wandb


# 0. Initialize Weights & Biases for experiment tracking

wandb.init(
    entity="lumr0067-west-virginia-university",
    project="ECG-WN-KEY-GENERATION",
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


# 1. Data loading

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
            if pid not in valid_ids:
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


# 2. WaveNet Residual Block
class WaveNetResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        # Casual dilated convolution for filter & gate
        #pad = (kernel_size - 1) * dilation
        self.conv_filter = nn.Conv1d(channels, channels, kernel_size, padding="same", dilation=dilation)
        self.conv_gate = nn.Conv1d(channels, channels, kernel_size, padding="same", dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        # 1x1 convolutions for residual & skip outputs
        self.residual = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # x shape: (batch, channels, time)
        f = torch.tanh(self.conv_filter(x))
        g = torch.sigmoid(self.conv_gate(x))
        z = f * g
        z = self.dropout(z)
        residual = self.residual(z) + x
        skip = self.skip(z)
        return residual, skip


# 3. Full WaveNet Key Generator Model
class WaveNetKeyGeneration(nn.Module):
    def __init__(self, seq_len, channels, n_blocks, kernel_size, key_bits, dropout):
        super().__init__()
        # Project input (1 channel) into high-dimensional feature space
        self.initial = nn.Conv1d(1, channels, 1)
        # Stack of dilated residual blocks
        self.blocks = nn.ModuleList([
            WaveNetResidualBlock(channels, kernel_size, 2**i, dropout)
            for i in range(n_blocks)
        ])
        self.post_relu = nn.ReLU()
        self.post_conv = nn.Conv1d(channels, channels, 1, padding='same')
        # Global average pool over time -> (batch, channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Final dense projection to kye_bits and sigmoid for binary output
        self.fc = nn.Linear(channels, key_bits)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x shape: (batch, seq_len, 1) ->permute to (batch, 1, seq_len)
        x = x.permute(0, 2, 1)
        x = self.initial(x)
        skip_sum = 0
        # Accumulate all skip connections
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        x = self.post_relu(skip_sum)
        x = self.post_conv(x)
        x = self.post_relu(x)
        # Pool down to (batch, channels)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return self.sigmoid(x)


# 4. Training & Generation System: mirrors TF main exactly
class KeyGenerationSystem:
    def __init__(self, data_dir, key_path, device=None):
        # 1) Initialize data loader
        self.loader = ECGKeyLoader(data_dir, key_path)
        # 2) Set compute device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 3) Buidl the Pytorch WaveNet model and move to device
        self.model = WaveNetKeyGeneration(
            seq_len=config.seq_len,
            channels=config.num_filters,
            n_blocks=config.num_wavenet_blocks,
            kernel_size=config.kernel_size,
            key_bits=config.key_bits,
            dropout=config.dropout_rate
        ).to(self.device)

    def train(self, epochs=config.epochs, batch_size=config.batch_size,
              lr=config.learning_rate, patience=config.patience):
        # 1) Get train/val splits
        X_train, X_val, Y_train, Y_val = self.loader.get_train_data()
        print("X_train shape:", X_train.shape)
        print("X_val   shape:", X_val.shape)
        print("Y_train shape:", Y_train.shape)
        print("Y_val   shape:", Y_val.shape)

        # 2) Wrap in DataLoader
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # 3) Optimizer, loss, and WandB monitoring
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        wandb.watch(self.model, log="all", log_freq=50)

        best_val = float("inf")
        no_improve = 0

        for ep in range(1, epochs + 1):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                output = self.model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += criterion(self.model(xb), yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            # Log & early stopping
            wandb.log({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})
            print(f"Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), "best_wavenet.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Stopping early at epoch {ep}")
                    break

        # Restore best weights
        self.model.load_state_dict(torch.load("best_wavenet.pt"))

    def generate_key(self, ecg_segments, threshold=0.5):
        # Identical logic: average per-segment predictions, threshold
        array = np.array(ecg_segments, dtype=np.float32)
        if array.ndim == 2:
            array = array.reshape(-1, config.seq_len, 1)
        tensor = torch.from_numpy(array).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(tensor).cpu().numpy()
        avg = probs.mean(axis=0)
        return (avg > threshold).astype(np.int32)


# 5. Main
if __name__ == "__main__":
    DATA_DIR = ""
    KEY_FILE = ""

    try:
        print("Initializing system....")
        kgs = KeyGenerationSystem(DATA_DIR, KEY_FILE)

        print("\nStarting training....")
        kgs.train()

        # Dummy forward pass to build model
        _ = kgs.model(torch.zeros((1, config.seq_len, 1), device=kgs.device))

        # ----------------------------
        # Test key generation for all persons
        # ----------------------------
        print("\nTesting key generation for all persons:")

        aggregated_keys = {}     # person_id -> aggregated (final) key
        all_intra_distances = [] # flat list of ALL intra-person pair distances

        for person in kgs.loader.persons:
            segments = person["segments"]
            if segments.ndim == 2:  # ensure (N_segments, seq_len, 1)
                segments = segments.reshape(segments.shape[0], segments.shape[1], 1)

            # Aggregated key (average over segments then threshold)
            aggregated_key = kgs.generate_key(segments)
            ground_truth   = person["key"].astype(np.int32)
            acc = np.mean(aggregated_key == ground_truth)

            print(f"\nPerson {person['id']}:")
            print(f"  Aggregated Key Accuracy: {acc:.2%}")
            print(f"  Aggregated Key (first 24 bits): {aggregated_key[:24]}...")
            print(f"  Ground Truth   (first 24 bits): {ground_truth[:24]}...")

            aggregated_keys[person["id"]] = aggregated_key

            # ---- Intra-person pairwise Hamming distances (segment-level) ----
            with torch.no_grad():
                seg_tensor = torch.from_numpy(segments).to(kgs.device)
                probs = kgs.model(seg_tensor).cpu().numpy()
            segment_keys = (probs > 0.5).astype(np.int32)  # shape: (num_segments, key_bits)

            n_seg = segment_keys.shape[0]
            if n_seg > 1:
                # Pairwise comparisons
                for i in range(n_seg):
                    ki = segment_keys[i]
                    for j in range(i + 1, n_seg):
                        d = int(np.sum(ki != segment_keys[j]))
                        all_intra_distances.append(d)
                per_person_mean = float(np.mean([
                    int(np.sum(segment_keys[i] != segment_keys[j]))
                    for i in range(n_seg) for j in range(i + 1, n_seg)
                ]))
                print(f"  Intra-person average Hamming distance: {per_person_mean:.2f} bits")
            else:
                print("  Not enough segments for intra-person Hamming distance.")

        # ---- Overall intra statistics (flat list) ----
        if all_intra_distances:
            overall_intra_mean = np.mean(all_intra_distances)
            overall_intra_std  = np.std(all_intra_distances)
            print("\nOverall Intra-person Hamming Distance (pooled pairs): "
                  f"mean= {overall_intra_mean:.2f} bits, std= {overall_intra_std:.2f} bits, "
                  f"n_pairs={len(all_intra_distances)}")
        else:
            print("\nNo intra-person distances collected.")

        # ----------------------------
        # Inter-person Hamming distances (aggregated keys)
        # ----------------------------
        print("\nInter-person Hamming distances (aggregated keys):")
        person_ids = sorted(aggregated_keys.keys())

        # dict person_id -> list of distances to *each other* person
        person_inter_dists = {pid: [] for pid in person_ids}

        for i in range(len(person_ids)):
            pid_i = person_ids[i]
            key_i = aggregated_keys[pid_i]
            for j in range(i + 1, len(person_ids)):
                pid_j = person_ids[j]
                key_j = aggregated_keys[pid_j]
                d = int(np.sum(key_i != key_j))
                # store distance for both persons (duplicated style you already use)
                person_inter_dists[pid_i].append(d)
                person_inter_dists[pid_j].append(d)
                print(f"  {pid_i} vs {pid_j}: {d} bits")

        # Overall inter (flatten duplicated lists)
        all_inter_dup = [d for lst in person_inter_dists.values() for d in lst]
        if all_inter_dup:
            inter_mean = np.mean(all_inter_dup)
            inter_std  = np.std(all_inter_dup)
            print(f"\nOverall Inter-person Hamming Distance (duplicated entries): "
                  f"mean= {inter_mean:.2f} bits, std= {inter_std:.2f} bits, n_entries={len(all_inter_dup)}")
        else:
            print("\nNo inter-person distances computed.")

        # ----------------------------
        # Save PKL files (formats compatible with your plotting script)
        # ----------------------------
        with open("all_intra_distances.pkl", "wb") as f:
            pickle.dump(all_intra_distances, f)

        with open("person_inter_dists.pkl", "wb") as f:
            pickle.dump(person_inter_dists, f)

        print("\nSaved PKL files:")
        print("  all_intra_distances.pkl  (flat list of intra pair distances)")
        print("  person_inter_dists.pkl   (dict person_id -> list of inter distances)")

    except Exception as e:
        print(f"\nError: {e}")
        print("Verification Checklist:")
        print("1. Directory structure: Person_XX/rec_N_filtered/*.csv")
        print("2. CSV files contain exactly 170 values, no headers")
        print("3. JSON keys match Person_XX numbering (1-89)")
        print("4. Minimum 10 segments across all persons")
        print("Something went wrong; please verify your paths and data.")

