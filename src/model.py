from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==============================
# 🔹 KMeans (Customer Segmentation)
# ==============================

def train_kmeans(X_scaled, n_clusters=3):
    """
    Train KMeans model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    return kmeans, clusters


def assign_clusters(kmeans, X_scaled):
    """
    Assign clusters using trained KMeans
    """
    return kmeans.predict(X_scaled)


# ==============================
# 🔹 Churn Prediction Models
# ==============================

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc


# ==============================
# 🔥 Model Comparison (IMPORTANT)
# ==============================

def train_and_compare_models(X_scaled, y):
    """
    Train multiple models and return best one
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    results = {}

    # Logistic Regression
    lr = train_logistic_regression(X_train, y_train)
    lr_acc = evaluate_model(lr, X_test, y_test)
    results['Logistic Regression'] = lr_acc

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    rf_acc = evaluate_model(rf, X_test, y_test)
    results['Random Forest'] = rf_acc

    # Select best model
    best_model = rf if rf_acc >= lr_acc else lr
    best_model_name = 'Random Forest' if rf_acc >= lr_acc else 'Logistic Regression'

    return best_model, best_model_name, results


# ==============================
# 🔹 Prediction Functions
# ==============================

def predict_churn(model, X_scaled):
    """
    Predict churn + probability
    """
    predictions = model.predict(X_scaled)

    # Some models (like RF) also support predict_proba
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[:, 1]
    else:
        probabilities = predictions

    return predictions, probabilities