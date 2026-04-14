import os
import joblib

from src.preprocessing import preprocess_data
from src.model import train_kmeans, train_and_compare_models

import warnings
warnings.filterwarnings("ignore")

# Paths
DATA_PATH = os.path.join("data", "telco.csv")
MODEL_DIR = "models"


def main():
    print("Starting training pipeline...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    df, X, y, X_scaled, scaler = preprocess_data(DATA_PATH)
    print("Data preprocessing completed")

    # KMeans Clustering
    kmeans, clusters = train_kmeans(X_scaled)
    df['Cluster'] = clusters
    print("KMeans clustering completed")

    model, model_name, results = train_and_compare_models(X_scaled, y)

    print(f"Best Model Selected: {model_name}")
    print("Model Comparison Results:")
    for name, score in results.items():
        print(f"   {name}: {score:.4f}")

    joblib.dump(model, os.path.join(MODEL_DIR, "churn_model.pkl"))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("Models saved successfully in /models")

    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()