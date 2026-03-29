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

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Rename target column
    if 'Churn Label' in df.columns:
        df.rename(columns={'Churn Label': 'Churn'}, inplace=True)

    # Convert target to numeric (Yes/No → 1/0)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Convert TotalCharges if exists
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop unnecessary columns (VERY IMPORTANT 🔥)
    drop_cols = ['Churn Value', 'Churn Score', 'CLTV', 'Churn Reason']
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Drop missing values
    df.dropna(inplace=True)

    # Drop ID if exists
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

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

    if 'Churn' not in df.columns:
        raise Exception("Target column 'Churn' not found!")

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_scaled, scaler = scale_features(X)

    return df, X, y, X_scaled, scaler