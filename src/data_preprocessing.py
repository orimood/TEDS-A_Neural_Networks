import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_teds_data(
    csv_file_path: str,
    initial_columns: list,
    final_features: list,
    target: str = 'SUB1',
    missing_val: int = -9,
    min_target_freq: float = 0.09,
    test_size: float = 0.2,
    seed: int = 42
) -> tuple:
    # Load selected columns from the dataset
    df = pd.read_csv(csv_file_path, usecols=initial_columns)

    # Remove rows with missing_val (-9) in any column
    df = df[~(df == missing_val).any(axis=1)].reset_index(drop=True)

    # Remove rare SUB1 categories (less than frequency threshold)
    freq = df[target].value_counts(normalize=True)
    df = df[~df[target].isin(freq[freq < min_target_freq].index)].reset_index(drop=True)

    # Separate features and target, convert features to string before encoding
    X = df[final_features].astype(str)
    y = df[target]

    # Encode features column-wise using LabelEncoder
    encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
    X_enc = np.column_stack([encoders[col].transform(X[col]) for col in X.columns])

    # Encode target column
    target_encoder = LabelEncoder().fit(y)
    y_enc = target_encoder.transform(y)

    # Split into train and test sets with stratified sampling on target
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=test_size, random_state=seed, stratify=y_enc
    )

    # Return processed data and encoders
    return X_train, X_test, y_train, y_test, encoders, target_encoder, final_features

if __name__ == '__main__':
    # Define columns to load and final features (excluding target)
    COLS = [
        'AGE', 'GENDER', 'RACE', 'ETHNIC', 'EDUC', 'EMPLOY',
        'LIVARAG', 'PRIMINC', 'STFIPS', 'REGION', 'DIVISION',
        'HLTHINS', 'SUB1'
    ]
    FEATURES = COLS[:-1]  # all except SUB1

    # Run preprocessing
    X_train, X_test, y_train, y_test, encs, y_enc, feat_names = preprocess_teds_data(
        'tedsa_puf_2020.csv',
        initial_columns=COLS,
        final_features=FEATURES
    )

    # Print summary
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Target classes: {y_enc.classes_}")
