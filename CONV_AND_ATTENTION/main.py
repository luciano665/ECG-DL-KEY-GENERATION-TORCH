import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
import os


# 0. Initialize wandb client
#   - A training monitoring tool, visualization of loss

# Get wandb API_key
# Login into wandb library
#wandb.login(relogin=True)

# Init wandDB to record results
wandb.init(
    entity = "lumr0067-west-virginia-university",

    project = "ecg-key_generation",

    config = {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "patience": 10
    }
)
config = wandb.config # access config values throughout

# 1. Data Loader: ECGKeyLoader
#    - Reads ECG segments and associated ground-truth keys
#    - Filters directories and files, validates segment length
#    - Splits data into training and validation sets

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



# 2. Model Definition: BioKeyTransformer
#    - 1D CNN encoder to extract features from ECG
#    - Multi-head self-attention to learn temporal dependencies
#    - LayerNorm + residual for stability
#    - Global average pooling → binary key projection
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
        # x: (batch, seq_len, 1) -> permute for Conv1D: (batch, 1, seq_len)
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


# 3. Training System: KeyGenerationSystem
#    - Integrates data loading, model instantiation, training loop
#    - Implements early stopping based on validation loss
#    - Provides key generation (inference) method
class KeyGenerationSystem:
    """
    Orchestrates data loading, model training, and key generation
    """
    def __init__(self, data_dir, key_path, device=None):
        # Init data loader
        self.loader = ECGKeyLoader(data_dir, key_path)
        # Determine compute device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Instantiate model with correct key lenght
        key_bits = self.loader.persons[0]['key'].shape[0]
        self.model = BioKeyTransformer(key_bits=key_bits).to(self.device)
        wandb.watch(self.model, log='all', log_freq=50)  # Monitor model logs

    def train(self, epochs=100, batch_size=32, lr=1e-4, patience=10):
        """
        Train the model with:
            - Adam Optimizer
            - Binary cross-entropy loss
            - Early stopping after 'patience' epochs without val_loss improvment
        Returns training history dict
        """

        # Get splits
        X_train, X_val, y_train, y_val = self.loader.get_train_data()
        # Wrap in TensorDatasets
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        # Dataloaders for batch iteration
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Initialize best_state to current weights
        best_state = self.model.state_dict()
        best_loss = float('inf')
        epochs_no_improve = 0
        #best_state = None
        history = {'train_loss': [], 'val_loss': []}

        for ep in range(1, epochs+1):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(xb)
                loss = criterion(predictions, yb)
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

            # Log metrics to Wandb
            wandb.log({
                'epoch': ep,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            print(f"Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Early stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Stopping early at epoch {ep}")
                    break
        # Restore best model weights
        self.model.load_state_dict(best_state)
        return history

    def generate_key(self, ecg_segments, threshold=0.5):
        """
        Inference: aggregate predictions over multiple segments to
        produce final binary key via thresholding
        """
        if ecg_segments is None or len(ecg_segments) == 0:
            raise  ValueError("No ECG segments provided")
        # Convert list/array to tensor
        arr = np.array(ecg_segments, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, -1)
        tensor = torch.from_numpy(arr).to(self.device)

        # Model forward pass
        self.model.eval()
        with torch.no_grad():
            probs = self.model(tensor).cpu().numpy()
        # Average probabilities across segments
        avg = probs.mean(axis=0)
        # Threshold operation for to get binary key
        return  (avg > threshold).astype(np.int32)


# 4. Main Execution: script entrypoint
#    - Configurable data paths
#    - Error handling and reporting
#    - Computes intra- and inter-person Hamming distances
#    - Saves raw distance data for subsequent analysis

if __name__ == "__main__":
    DATA_DIR = ""
    KEY_FILE = ""

    try:
        print("Initializing system...")
        kgs = KeyGenerationSystem(DATA_DIR, KEY_FILE)

        print("\nStarting training...")
        kgs.train(epochs=100)

        # ---------------------------------------------------------------------
        # Testing and evaluation: compute Hamming distances
        # ---------------------------------------------------------------------

        print("\nTesting key generation for all persons:")

        # Dictionary to hold aggregated keys for inter-person comparisons
        aggregated_keys = {}
        all_intra_keys = []

        for person in kgs.loader.persons:
            segments = person['segments']
            # Ensure 3D tensor shape (N_segments, seq_len, 1)
            if segments.ndim == 2:
                segments = segments.reshape(segments.shape[0], segments.shape[1], 1)

            # Generate aggregated key from all segments for this person
            aggregated_key = kgs.generate_key(segments)
            ground_truth = person['key'].astype(np.int32)
            accuracy = np.mean(aggregated_key == ground_truth)

            print(f"\nPerson {person['id']}:")
            print(f"  Aggregated Key Accuracy: {accuracy:.2%}")
            print(f"  Aggregated Key: {aggregated_key[:24]}...")
            print(f"  Ground Truth:   {ground_truth[:24]}...")

            aggregated_keys[person['id']] = aggregated_key

            # ----------------------------
            # Compute Intra-Person Hamming Distance
            # ----------------------------
            # Segment-level forward pass (PyTorch)
            with torch.no_grad():
                seg_tensor = torch.from_numpy(segments).to(kgs.device)
                probs = kgs.model(seg_tensor).cpu().numpy()
            individual_keys = (probs > 0.5).astype(np.int32)  # shape: (num_segments, key_bits)
            num_keys = individual_keys.shape[0]

            if num_keys > 1:
                distances = []
                for i in range(num_keys):
                    for j in range(i + 1, num_keys):
                        d = int(np.sum(individual_keys[i] != individual_keys[j]))
                        distances.append(d)
                avg_distance = np.mean(distances)
                print(f"  Intra-person average Hamming distance: {avg_distance:.2f} bits")
                all_intra_keys.extend(distances)
            else:
                print("  Not enough segments to compute intra-person Hamming distance.")

        # Overall intra HD statistics
        if all_intra_keys:
            overall_intra_mean = np.mean(all_intra_keys)
            overall_intra_std = np.std(all_intra_keys)
            print("\nOverall Intra-person Hamming Distance: "
                  f"mean= {overall_intra_mean:.2f} bits, std= {overall_intra_std:.2f} bits")
        else:
            print("\nNo data available to compute mean and std for Intra-person Hamming Distance.")

        # ----------------------------
        # Compute Inter-Person Hamming Distances (aggregated keys)
        # ----------------------------
        person_ids = sorted(aggregated_keys.keys())
        person_inter_dists = {p: [] for p in person_ids}
        print("\nInter-person Hamming distances (aggregated keys):")

        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                key1 = aggregated_keys[person_ids[i]]
                key2 = aggregated_keys[person_ids[j]]
                d = int(np.sum(key1 != key2))
                # Save the distance for both persons
                person_inter_dists[person_ids[i]].append(d)
                person_inter_dists[person_ids[j]].append(d)
                print(f"  Distance between Person {person_ids[i]} and Person {person_ids[j]}: {d} bits")

        # Compute overall inter-person statistics (double-counted pairs in dict)
        all_inter_distances = []
        for dist_list in person_inter_dists.values():
            all_inter_distances.extend(dist_list)

        if all_inter_distances:
            overall_inter_mean = np.mean(all_inter_distances)
            overall_inter_std = np.std(all_inter_distances)
            print("\nOverall Inter-person Hamming Distance (double-counted lists): "
                  f"mean = {overall_inter_mean:.2f} bits, std = {overall_inter_std:.2f} bits")
            # If you want true unordered pair stats, compute once-through:
            unordered = []
            for i in range(len(person_ids)):
                for j in range(i + 1, len(person_ids)):
                    unordered.append(int(np.sum(aggregated_keys[person_ids[i]] != aggregated_keys[person_ids[j]])))
            print(
                f"  (Unique pairs) mean = {np.mean(unordered):.2f} bits, std = {np.std(unordered):.2f} bits, n_pairs={len(unordered)}")
        else:
            print("\nNo data available to compute inter-person Hamming Distance statistics.")

        # ----------------------------
        # Save the raw distance data for later plotting:
        # ----------------------------
        with open("all_intra_distances.pkl", "wb") as f:
            pickle.dump(all_intra_keys, f)

        with open("person_inter_dists.pkl", "wb") as f:
            pickle.dump(person_inter_dists, f)

        print("\nSaved PKL files:")
        print("  all_intra_distances.pkl  (flat list of intra pair distances)")
        print("  person_inter_dists.pkl   (dict person_id -> list of inter distances)")

    except Exception as e:
        print(f"\nError: {e}")
        print("Checklist:")
        print("1. Person_XX/rec_2_filtered/*.csv present")
        print("2. Each CSV: 170 values, no header")
        print("3. JSON keys match Person_XX IDs")
        print("4. ≥10 segments total")



