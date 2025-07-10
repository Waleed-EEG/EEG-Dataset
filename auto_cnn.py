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
import pywt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from scipy import signal
from scipy.signal import hilbert
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")

from itertools import combinations

# === CONFIGURATION ===
NUM_SUBJECTS = 46
NUM_CLASSES = 3
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 0.005
SEED = 42
VAL_SPLIT = 0.1
PATIENCE = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def save_features(X, y, subject_id, save_dir='cached_features'):
    """
    Save the features and labels to disk to avoid re-extraction.
    """
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"subject_{subject_id}_X.npy"), X)
    np.save(os.path.join(save_dir, f"subject_{subject_id}_y.npy"), y)
    print(f"Features for subject {subject_id} saved.")

def load_cached_features(subject_id, save_dir='cached_features'):
    """
    Load cached features from disk if available.
    """
    X_path = os.path.join(save_dir, f"subject_{subject_id}_X.npy")
    y_path = os.path.join(save_dir, f"subject_{subject_id}_y.npy")
    if os.path.exists(X_path) and os.path.exists(y_path):
        X = np.load(X_path)
        y = np.load(y_path)
        print(f"Loaded cached features for subject {subject_id}")
        return X, y
    return None, None

# Sampling parameters
SAMPLING_RATE = 256
WINDOW_SIZE_SEC = 2  # 2-second window
STEP_SIZE_SEC = 1    # 1-second overlap
RAW_COLUMNS = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
EXPECTED_COLUMNS = [
    'Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10',
    'Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10',
    'Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10',
    'Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10',
    'Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10'
]
channel_pairs = list(combinations(RAW_COLUMNS, 2))  # 6 unique pairs

from scipy.signal import coherence

def compute_plv(sig1, sig2):
    analytic1 = hilbert(sig1)
    analytic2 = hilbert(sig2)
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    phase_diff = phase1 - phase2
    plv = np.abs(np.sum(np.exp(1j * phase_diff)) / len(phase_diff))
    return plv

def compute_band_coherence(sig1, sig2, fs=256, band=(8, 13)):
    f, Cxy = coherence(sig1, sig2, fs=fs, nperseg=fs)
    band_indices = np.logical_and(f >= band[0], f <= band[1])
    return np.mean(Cxy[band_indices])

bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}



def compute_instantaneous_phase(signal):
    """
    Compute the instantaneous phase of a signal using the Hilbert transform.
    """
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))  # Unwrap to ensure continuity
    return instantaneous_phase

# === DeepConvNet ===
class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed

class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B, 1, F]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 32, F]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 64, F]
        x = self.pool(x).squeeze(-1)         # [B, 64]
        x = self.dropout(x)
        return self.fc(x)        
    

def compute_phase_difference(sig1, sig2):
    # Compute phase difference using Hilbert transform
    analytic1 = signal.hilbert(sig1)
    analytic2 = signal.hilbert(sig2)

    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)

    # Compute mean absolute phase difference
    phase_diff = np.abs(np.unwrap(phase1 - phase2))
    return np.mean(phase_diff)




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

def compute_dwt_features(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []

    for coeff in coeffs:
        # Energy
        energy = np.sum(np.square(coeff))
        # Entropy
        prob = np.square(coeff) / (np.sum(np.square(coeff)) + 1e-6)
        ent = -np.sum(prob * np.log2(prob + 1e-6))
        # Mean and STD
        features.extend([energy, ent, np.mean(coeff), np.std(coeff)])

    return features  # total features: 4 * (level + 1)


def compute_engineered_features(df):
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    channels = ['TP9', 'AF7', 'AF8', 'TP10']

    features = []

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

    # Ensure the features are flattened properly and have consistent size
    features_flattened = np.array(features).flatten()

    # Debugging check: Print the length of the features
    print(f"Feature vector length: {len(features_flattened)}")

    return features_flattened  # Ensure it's a 1D array

# === Extract Features from a Window of Data ===
def extract_window_features(df_window, sampling_rate=256):
    features = []
    # === 1b. Phase Differences Between Channel Pairs ===
    ch_pairs = [('RAW_TP9', 'RAW_AF7'), ('RAW_AF7', 'RAW_AF8'),
                ('RAW_AF8', 'RAW_TP10'), ('RAW_TP9', 'RAW_TP10')]
    
    # === 1. Power Band Features ===
    band_power_values = df_window[EXPECTED_COLUMNS].mean().values
    features.extend(band_power_values.tolist())  # 20 features

    # === 2. Entropy (1 per raw channel) ===
    for ch in RAW_COLUMNS:
        signal = df_window[ch].values
        features.append(compute_shannon_entropy(signal))  # 4 features

    # === 3. Phase Difference (4 pairs) ===
    for ch1, ch2 in ch_pairs:
        sig1 = df_window[ch1].values
        sig2 = df_window[ch2].values
        features.append(compute_phase_difference(sig1, sig2))  # 4 features

        # === 6. DWT Features (per raw channel) ===
    for ch in RAW_COLUMNS:
        signal = df_window[ch].values
        dwt_feats = compute_dwt_features(signal, wavelet='db4', level=4)
        features.extend(dwt_feats)  # 20 features per channel => 4 x 20 = 80

        # === 4. Phase Locking Value (PLV) ===
    for ch1, ch2 in channel_pairs:
        sig1 = df_window[ch1].values
        sig2 = df_window[ch2].values
        plv_value = compute_plv(sig1, sig2)
        features.append(plv_value)  # 6 features

    # === 5. Coherence Features (Per Band, Per Pair) ===
    for ch1, ch2 in channel_pairs:
        sig1 = df_window[ch1].values
        sig2 = df_window[ch2].values
        for band_range in bands.values():
            coh = compute_band_coherence(sig1, sig2, fs=sampling_rate, band=band_range)
            features.append(coh)  # 6 pairs × 5 bands = 30 features
        

    return np.array(features)  # Total: 20 + 4 + 4 + 3 = **31 features**



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

def normalize_subject_data(all_data):
    # Normalize data for each subject
    normalized_data = []
    for X, y in all_data:
        # Normalize X (the feature matrix) for each subject
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        normalized_data.append((X_normalized, y))
    return normalized_data



# === TRAINING + EVALUATION ===
def train_and_evaluate(X_train, y_train, X_test, y_test, input_dim, num_classes=3, device='cpu'):

    # === Print class distribution before ADASYN ===
    print("Before ADASYN:", Counter(y_train))

    # Save original training data before ADASYN
    X_train_original = X_train.copy()
    y_train_original = y_train.copy()

    nan_mask = ~np.isnan(X_train_original).any(axis=1)
    X_train_original = X_train_original[nan_mask]
    y_train_original = y_train_original[nan_mask]

    print(f"Dropped {(~nan_mask).sum()} rows with NaNs before ADASYN.")

    # Apply ADASYN (upsample minority classes adaptively)
    adasyn = ADASYN(sampling_strategy='not majority', random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_original, y_train_original)

    # Identify synthetic samples: those beyond the original length
    num_original = X_train_original.shape[0]
    X_synthetic = X_train_resampled[num_original:]
    y_synthetic = y_train_resampled[num_original:]

    # === Apply feature-wise Gaussian noise to synthetic samples (labels 1 & 2 only) ===
    def add_featurewise_gaussian_noise(X, reference_std, mean=0.0, noise_scale=0.05):
        noise = np.random.normal(loc=mean, scale=reference_std * noise_scale, size=X.shape)
        return X + noise

    # Compute std for each feature from original training data
    feature_std = X_train_original.std(axis=0)

    # Apply noise only to synthetic class 1 and 2
    mask = (y_synthetic == 1) | (y_synthetic == 2)
    X_synthetic[mask] = add_featurewise_gaussian_noise(X_synthetic[mask], reference_std=feature_std)

    # Combine the original and new synthetic data
    X_train = np.vstack((X_train_original, X_synthetic))
    y_train = np.hstack((y_train_original, y_synthetic))

    # Final distribution check
    print("After SMOTE-ENN + Gaussian noise:", Counter(y_train))


    # Create Datasets & Loaders
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # === Train Autoencoder ===
    autoencoder = FeatureAutoencoder(input_dim=input_dim).to(device)
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    ae_criterion = nn.MSELoss()

    autoencoder.train()
    for epoch in range(10):  # You can increase epochs
        total_loss = 0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            _, reconstructed = autoencoder(X_batch)
            loss = ae_criterion(reconstructed, X_batch)
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item()
        print(f"[Autoencoder] Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # === Extract Encoded Features ===
    def encode_features(loader):
        autoencoder.eval()
        all_feats, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                encoded, _ = autoencoder(X_batch)
                all_feats.append(encoded.cpu())
                all_labels.append(y_batch)
        return torch.cat(all_feats), torch.cat(all_labels)

    X_train_encoded, y_train_encoded = encode_features(train_loader)
    X_test_encoded, y_test_encoded = encode_features(test_loader)

    # === Train LSTM Classifier ===
    # Reshape to [batch, seq_len, features] where seq_len=1
    X_train_encoded = X_train_encoded.unsqueeze(1)  # Shape: (B, 1, F)
    X_test_encoded = X_test_encoded.unsqueeze(1)

    train_loader_cls = DataLoader(EEGDataset(X_train_encoded, y_train_encoded), batch_size=256, shuffle=True)
    test_loader_cls = DataLoader(EEGDataset(X_test_encoded, y_test_encoded), batch_size=256, shuffle=False)

    classifier = CNNClassifier(input_dim=X_train_encoded.shape[2], num_classes=num_classes).to(device)
    cls_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    cls_criterion = nn.CrossEntropyLoss()

    classifier.train()
    for epoch in range(10):  # Adjust as needed
        total_loss = 0
        for X_batch, y_batch in train_loader_cls:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = classifier(X_batch)
            loss = cls_criterion(outputs, y_batch)
            cls_optimizer.zero_grad()
            loss.backward()
            cls_optimizer.step()
            total_loss += loss.item()
        print(f"[Classifier] Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # === Evaluation ===
    classifier.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader_cls:
            X_batch = X_batch.to(device)
            outputs = classifier(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\nTest Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        # === Per-class metrics ===
    print("\nPer-Class Performance:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    return acc, prec, rec, f1

# === MAIN ===
if __name__ == '__main__':


    # Assuming that after calling `load_all_data`, the data will be structured as below
    all_data = load_all_data()

    print("Normalizing and structuring data...")
    all_data = normalize_subject_data(all_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    num_subjects = len(all_data)

    for test_idx in range(num_subjects):
        print(f"\n=== LOSO Fold {test_idx + 1}/{num_subjects} ===")
        
        # Prepare training and test data for LOOS cross-validation
        train_data = [all_data[i] for i in range(num_subjects) if i != test_idx]
        test_data = all_data[test_idx]

        X_train = np.vstack([d[0] for d in train_data])
        y_train = np.hstack([d[1] for d in train_data])
        X_test, y_test = test_data

        input_dim = X_train.shape[1]

        acc, prec, rec, f1 = train_and_evaluate(X_train, y_train, X_test, y_test, input_dim, device=device)

        # Collect metrics
        all_metrics['accuracy'].append(acc)
        all_metrics['precision'].append(prec)
        all_metrics['recall'].append(rec)
        all_metrics['f1'].append(f1)

    # Print final results
    print("\n=== Final LOSO Evaluation Results ===")
    for metric in all_metrics:
        print(f"{metric.capitalize()}: {np.mean(all_metrics[metric]):.4f}")
