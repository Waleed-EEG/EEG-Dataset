import pandas as pd
import numpy as np
from sklearn.svm import SVC
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
df_aggregated.dropna(inplace=True)

# Split data into features and labels
X = df_aggregated.drop('Label', axis=1).values
y = df_aggregated['Label'].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the SVM model
svm_model = SVC(kernel='linear')

# Perform 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold = 1
all_y_true = []
all_y_pred = []

for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    print(f"\n--- Fold {fold} ---")
    print(classification_report(y_test, y_pred))

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    fold += 1

# Overall classification report
print("\n=== Overall Classification Report (All Folds Combined) ===")
print(classification_report(all_y_true, all_y_pred))
