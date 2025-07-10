import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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
file_paths = [f'Participant_{i}.csv' for i in range(1, 47)]  # Update the file paths if needed
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

# Prepare the data
X = df_aggregated.drop('Label', axis=1).values
y = df_aggregated['Label'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the device to run the model on (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define an LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the output of the last time step
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Define a function for training and evaluating the model
def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    # Convert data to PyTorch tensors and add a sequence length dimension
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = LSTM(input_size=X_train.shape[1], hidden_size=64, num_classes=len(np.unique(y)))
    model.to(device)  # Move model to the correct device (GPU or CPU)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping parameters
    patience = 2
    best_loss = float('inf')
    epochs_since_improvement = 0

    # Training with early stopping
    num_epochs = 5
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')

        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f'Early stopping after {epoch+1} epochs.')
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model on the validation set
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
    report = classification_report(all_labels, all_preds)
    return accuracy, report

# Set up KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
reports = []

# Perform 10-fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Training fold {fold + 1}")
    
    # Split data into train and validation sets
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train and evaluate the model on the current fold
    accuracy, report = train_and_evaluate_model(X_train, y_train, X_val, y_val)
    
    accuracies.append(accuracy)
    reports.append(report)

# Calculate and print the average accuracy and classification report across all folds
average_accuracy = np.mean(accuracies)
print(f"Average accuracy across 10 folds: {average_accuracy:.4f}")

# Optionally print a summary of the classification reports
for fold, report in enumerate(reports):
    print(f"Classification report for fold {fold + 1}:\n{report}")
