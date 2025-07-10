import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import warnings
import torch.nn.init as init
import torch.optim as optim
from itertools import product
from scipy.stats import entropy
from scipy.fft import fft


warnings.filterwarnings("ignore")

# === CONFIGURATION ===
NUM_SUBJECTS = 46
NUM_CLASSES = 3
BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 0.005
SEED = 42
VAL_SPLIT = 0.1
PATIENCE = 1

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sampling parameters
SAMPLING_RATE = 256
WINDOW_SIZE_SEC = 2  # 2-second window
STEP_SIZE_SEC = 1    # 1-second overlap

EXPECTED_COLUMNS = [
    'Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10',
    'Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10',
    'Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10',
    'Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10',
    'Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10'
]

# === DeepConvNet ===
class ConvBiLSTMNet(nn.Module):
    def __init__(self, num_classes=3, input_channels=4, input_samples=5, hidden_size=64, num_layers=1):
        super(ConvBiLSTMNet, self).__init__()

        # Conv layer to extract spatial features
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.3)
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM to process temporal dynamics
        self.bilstm = nn.LSTM(
            input_size=16 * input_channels,  # 16 filters × 4 channels
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # x shape: [batch, 1, 4, 5]
        x = self.conv(x)  # [batch, 16, 4, 5]
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, time=5, filters=16, channels=4]
        x = x.view(x.size(0), x.size(1), -1)    # [batch, 5, 64]

        output, _ = self.bilstm(x)  # [batch, 5, hidden*2]
        out = output[:, -1, :]      # Take output from last time step
        out = self.fc(out)          # [batch, num_classes]
        return out



# === Dataset ===
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

# === Compute Shannon Entropy ===
def compute_shannon_entropy(signal, bins=30):
    # Compute the histogram and normalize it
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist += 1e-6  # Add small constant to avoid log(0)
    return entropy(hist)

# === Compute FFT Features ===
# Define the FFT feature extraction function
def compute_fft_features(signal, sampling_rate=256):
    n = len(signal)  # Length of the signal
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)  # Compute frequency bins
    fft_values = np.abs(fft(signal))  # Compute the magnitude of the FFT
    pos_mask = freqs > 0  # We only care about positive frequencies

    # Filter out negative frequencies
    freqs = freqs[pos_mask]
    fft_values = fft_values[pos_mask]

    # Compute FFT features
    dominant_freq = freqs[np.argmax(fft_values)]  # The frequency with the highest magnitude
    mean_power = np.mean(fft_values)  # Mean power of the signal
    var_power = np.var(fft_values)  # Variance of the power

    # Return the features as a list
    return [dominant_freq, mean_power, var_power]


def compute_engineered_features(df):
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    channels = ['TP9', 'AF7', 'AF8', 'TP10']

    features = []

    # Basic band-wise features
    for band in bands:
        band_cols = [f"{band}_{ch}" for ch in channels]
        band_values = df[band_cols].values
        band_values_flat = band_values.flatten()  # Flatten to ensure consistent size
        features.extend(band_values_flat)

    # === Band Ratios ===
    alpha = df[[f"Alpha_{ch}" for ch in channels]].mean(axis=1)
    beta = df[[f"Beta_{ch}" for ch in channels]].mean(axis=1)
    theta = df[[f"Theta_{ch}" for ch in channels]].mean(axis=1)
    delta = df[[f"Delta_{ch}" for ch in channels]].mean(axis=1)

    theta_alpha_ratio = (theta + 1e-6) / (alpha + 1e-6)
    alpha_beta_ratio = (alpha + 1e-6) / (beta + 1e-6)
    theta_beta_ratio = (theta + 1e-6) / (beta + 1e-6)

    # Ensure all ratios are consistent
    features.extend([theta_alpha_ratio.mean(), alpha_beta_ratio.mean(), theta_beta_ratio.mean()])

    # === Asymmetry Scores ===
    asym_alpha = df['Alpha_AF8'] - df['Alpha_AF7']
    asym_beta = df['Beta_TP10'] - df['Beta_TP9']
    features.extend([asym_alpha.mean(), asym_beta.mean()])

    # === Relative Band Powers per Channel ===
    for ch in channels:
        total_power = sum([df[f"{band}_{ch}"] for band in bands])
        for band in bands:
            rel_power = df[f"{band}_{ch}"] / (total_power + 1e-6)
            features.append(rel_power.mean())

    # === Global Band Powers ===
    global_band_powers = []
    for band in bands:
        band_cols = [f"{band}_{ch}" for ch in channels]
        band_power = df[band_cols].mean(axis=1)
        global_band_powers.append(band_power.mean())

    features.extend(global_band_powers)

    global_band_means = []
    for band in bands:
        band_cols = [f"{band}_{ch}" for ch in channels]
        mean_power = df[band_cols].mean(axis=1)
        global_band_means.append(mean_power.mean())

    total_global_power = sum(global_band_means) + 1e-6  # avoid division by zero
    global_relative_psd = [band / total_global_power for band in global_band_means]

    features.extend(global_relative_psd)

    # Ensure the features are flattened properly and have consistent size
    features_flattened = np.array(features).flatten()

    # Debugging check: Print the length of the features
    print(f"Feature vector length: {len(features_flattened)}")

    return features_flattened  # Ensure it's a 1D array

# === Extract Features from a Window of Data ===
def extract_window_features(df_window, sampling_rate=256):
    features = []
    for col in EXPECTED_COLUMNS:
        signal = df_window[col].values
        mean_val = np.mean(signal)
        ent = compute_shannon_entropy(signal)
        fft_feats = compute_fft_features(signal, sampling_rate)
        features.extend([mean_val, ent] + fft_feats)

    # Add engineered features
    engineered_feats = compute_engineered_features(df_window)
    features.extend(engineered_feats.tolist())

    return features

def sliding_window_feature_extraction(df, window_size_sec=2, step_size_sec=1, sampling_rate=256):
    window_size = window_size_sec * sampling_rate
    step_size = step_size_sec * sampling_rate

    X = []
    y = []

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        df_window = df.iloc[start:end]
        if df_window['Label'].nunique() > 1:
            continue  # skip mixed label windows
        label = df_window['Label'].iloc[0]
        features = extract_window_features(df_window, sampling_rate)

        # Debugging: Print the length of features for each window
        print(f"Extracted {len(features)} features from window starting at index {start}")

        X.append(features)
        y.append(label)

    print(f"Feature vector shape: {len(X[0])} features per window")
    
    return np.array(X), np.array(y)

# === Load All Data ===
def load_all_data():
    data_per_subject = []
    for i in range(1, NUM_SUBJECTS + 1):
        file_path = f'Participant_{i}.csv'
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        if not set(EXPECTED_COLUMNS + ['Label']).issubset(df.columns):
            print(f"Missing columns in file {file_path}")
            continue

        X, y = sliding_window_feature_extraction(df, window_size_sec=WINDOW_SIZE_SEC, step_size_sec=STEP_SIZE_SEC, sampling_rate=SAMPLING_RATE)
        if len(X) == 0:
            print(f"No valid windows from subject {i}. Skipping.")
            continue

        data_per_subject.append((X, y))

    return data_per_subject

def normalize_subject_data(data_per_subject):
    normalized_data = []
    for X, y in data_per_subject:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        normalized_data.append((X_scaled, y))
    return normalized_data



# === TRAINING + EVALUATION ===
def train_and_evaluate(data):
    NUM_SUBJECTS = len(data)
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for test_idx in range(NUM_SUBJECTS):
        print(f"\n=== LOSO Fold {test_idx + 1}/{NUM_SUBJECTS} ===")
        train_data = [data[i] for i in range(NUM_SUBJECTS) if i != test_idx]
        test_data = data[test_idx]

        X_train = np.vstack([d[0] for d in train_data])
        y_train = np.hstack([d[1] for d in train_data])
        X_test, y_test = test_data

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        total_features = X_train.shape[1]
        num_samples = 5
        if total_features % num_samples != 0:
            raise ValueError(f"Feature count ({total_features}) must be divisible by num_samples ({num_samples})")
        num_channels = total_features // num_samples

        X_train = X_train.reshape(-1, 1, num_channels, num_samples)
        X_test = X_test.reshape(-1, 1, num_channels, num_samples)

        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=VAL_SPLIT, stratify=y_train)

        train_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE)
        test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE)

        model = ConvBiLSTMNet(
            num_classes=NUM_CLASSES,
            input_channels=num_channels,
            input_samples=num_samples,
            hidden_size=64,
            num_layers=1
        ).to(device)

        class_weights = Counter(y_tr)
        total = sum(class_weights.values())
        weight_tensor = torch.tensor([total / class_weights[i] for i in range(NUM_CLASSES)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        best_fold_f1 = 0
        patience_counter = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_trues.extend(y_batch.cpu().numpy())

            f1 = f1_score(val_trues, val_preds, average='macro', zero_division=0)
            print(f"Epoch {epoch} — Validation F1: {f1:.4f}")

            if f1 > best_fold_f1:
                best_fold_f1 = f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Test Evaluation
        model.eval()
        y_preds, y_trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                y_preds.extend(preds.cpu().numpy())
                y_trues.extend(y_batch.numpy())

        acc = accuracy_score(y_trues, y_preds)
        prec = precision_score(y_trues, y_preds, average='macro', zero_division=0)
        rec = recall_score(y_trues, y_preds, average='macro', zero_division=0)
        f1_final = f1_score(y_trues, y_preds, average='macro', zero_division=0)

        print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1_final:.4f}")

        all_metrics['accuracy'].append(acc)
        all_metrics['precision'].append(prec)
        all_metrics['recall'].append(rec)
        all_metrics['f1'].append(f1_final)

    print("\n=== Final Average Metrics (LOSO, Top 10 Features) ===")
    for metric in all_metrics:
        print(f"{metric.capitalize()}: {np.mean(all_metrics[metric]):.4f}")

    return all_metrics

# === MAIN ===
if __name__ == '__main__':
    print("Loading EEG data...")
    all_data = load_all_data()  # Use full features

    print("Normalizing data per subject...")
    all_data = normalize_subject_data(all_data)

    print("Starting LOSO evaluation...")
    all_metrics = train_and_evaluate(all_data)
