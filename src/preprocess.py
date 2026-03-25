import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def handle_missing(df):
    # Missing values remove (simple approach)
    df = df.dropna()
    return df

def encode_data(df):
    # Convert categorical → numeric
    df = pd.get_dummies(df)
    return df

def split_features_target(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler