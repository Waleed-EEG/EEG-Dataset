import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Function to aggregate features
def aggregate_features(df, window_size=300):
    aggregated_df = pd.DataFrame()
    for column in df.columns:
        if column != 'Label':
            aggregated_df[f'{column}_mean'] = df[column].rolling(window=window_size).mean().dropna()
            aggregated_df[f'{column}_std'] = df[column].rolling(window=window_size).std().dropna()
            aggregated_df[f'{column}_max'] = df[column].rolling(window=window_size).max().dropna()
            aggregated_df[f'{column}_min'] = df[column].rolling(window=window_size).min().dropna()
    aggregated_df['Label'] = df['Label'].iloc[window_size-1:].reset_index(drop=True)
    return aggregated_df

# Load datasets
file_paths = [f'Participant_{i}.csv' for i in range(1, 47)]
dfs = [pd.read_csv(fp) for fp in file_paths]
df = pd.concat(dfs, ignore_index=True)

# Convert timestamp to datetime and drop it
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
df.drop('TimeStamp', axis=1, inplace=True)

# Convert labels to numeric codes
df['Label'] = df['Label'].astype('category').cat.codes

# Apply aggregation
df_aggregated = aggregate_features(df)
df_aggregated.dropna(inplace=True)  # Drop rows with NaN values

# Split data into features and labels
X = df_aggregated.drop('Label', axis=1).values
y = df_aggregated['Label'].values

# Define a CNN model
class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * (X.shape[1] // 4), 256)  # Adjust based on pooling
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a function for cross-validation
def cross_validate(model_class, X, y, num_classes, num_folds=10, num_epochs=3, batch_size=32, patience=2):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_accuracies = []
    fold_reports = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{num_folds}")
        
        # Split the data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model, criterion, and optimizer
        model = model_class(1, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_loss = float('inf')
        epochs_since_improvement = 0

        # Training with early stopping
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            # Validation
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}")

            # Check for improvement
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_since_improvement = 0
                torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load the best model for this fold
        model.load_state_dict(torch.load(f'best_model_fold_{fold+1}.pth'))

        # Evaluate on the validation set
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True)
        fold_accuracies.append(accuracy)
        fold_reports.append(report)

        print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(all_labels, all_preds))

    # Summarize results
    mean_accuracy = np.mean(fold_accuracies)
    print(f"\nMean Accuracy over {num_folds} folds: {mean_accuracy:.4f}")
    return fold_accuracies, fold_reports

# Call the cross-validation function
num_classes = len(np.unique(y))
fold_accuracies, fold_reports = cross_validate(CNN, X, y, num_classes=num_classes)
