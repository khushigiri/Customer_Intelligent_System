import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load dataset safely"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def clean_data(df):
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.lower()

    # Rename target
    if 'churn_label' in df.columns:
        df.rename(columns={'churn_label': 'churn'}, inplace=True)

    # Convert target
    if 'churn' in df.columns:
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

    # Convert numeric columns
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

    # Drop leakage columns
    drop_cols = ['churn_value', 'churn_score', 'cltv', 'churn_reason']
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Drop ID
    if 'customerid' in df.columns:
        df.drop('customerid', axis=1, inplace=True)

    df.dropna(inplace=True)

    return df

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns"""
    df = df.copy()

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def scale_features(X):
    """Scale features"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def preprocess_data(path: str):
    """
    Full preprocessing pipeline
    Returns:
    df -> cleaned dataframe
    X -> features
    y -> target
    X_scaled -> scaled features
    scaler -> fitted scaler
    """
    df = load_data(path)
    df = clean_data(df)
    df = encode_data(df)

    if 'churn' not in df.columns:
        raise Exception("Target column 'churn' not found!")

    X = df.drop('churn', axis=1)
    y = df['churn']

    X_scaled, scaler = scale_features(X)

    return df, X, y, X_scaled, scaler