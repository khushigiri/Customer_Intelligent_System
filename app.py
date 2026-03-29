import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

from src.preprocessing import preprocess_data
from src.model import assign_clusters, predict_churn
from src.insights import generate_insights

# ==============================
# 🔹 Page Config
# ==============================
st.set_page_config(page_title="Customer Intelligence System", layout="wide")

st.title("📊 Customer Intelligence Dashboard")

# ==============================
# 🔹 Paths
# ==============================
DATA_PATH = os.path.join("data", "telco.csv")
MODEL_PATH = "models"

# ==============================
# 🔹 Load Models (cached)
# ==============================
@st.cache_resource
def load_models():
    model = joblib.load(os.path.join(MODEL_PATH, "churn_model.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_PATH, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
    return model, kmeans, scaler

model, kmeans, scaler = load_models()

# ==============================
# 🔹 Load Data
# ==============================
df, X, y, X_scaled, _ = preprocess_data(DATA_PATH)

# ==============================
# 🔹 Apply ML Pipeline
# ==============================
df['Cluster'] = assign_clusters(kmeans, X_scaled)

preds, probs = predict_churn(model, X_scaled)
df['Churn_Predicted'] = preds
df['Churn_Probability'] = probs

insights = generate_insights(df)

# ==============================
# 🔥 KPI CARDS
# ==============================
col1, col2, col3 = st.columns(3)

col1.metric("👥 Total Customers", len(df))
col2.metric("📉 Avg Churn Probability", f"{df['Churn_Probability'].mean():.2f}")
col3.metric("⚠️ High Risk Customers", (df['Churn_Probability'] > 0.5).sum())

st.divider()

# ==============================
# 📊 MODEL COMPARISON (STATIC DISPLAY)
# ==============================
st.subheader("⚙️ Model Comparison")

# You can update these if you want dynamic later
model_results = {
    "Logistic Regression": 0.79,
    "Random Forest": 0.84
}

fig_model = px.bar(
    x=list(model_results.keys()),
    y=list(model_results.values()),
    labels={'x': 'Model', 'y': 'Accuracy'},
    title="Model Performance Comparison"
)

st.plotly_chart(fig_model, use_container_width=True)

best_model = max(model_results, key=model_results.get)
st.success(f"🏆 Best Model Selected: {best_model}")

# ==============================
# 📈 CLUSTER VISUALIZATION
# ==============================
st.subheader("📈 Customer Segmentation")

x_col = "tenure" if "tenure" in df.columns else df.columns[0]
y_col = "MonthlyCharges" if "MonthlyCharges" in df.columns else df.columns[1]

fig_cluster = px.scatter(
    df,
    x=x_col,
    y=y_col,
    color="Cluster",
    title="Customer Clusters"
)

st.plotly_chart(fig_cluster, use_container_width=True)

# ==============================
# 📉 CHURN DISTRIBUTION
# ==============================
st.subheader("📉 Churn Probability Distribution")

fig_churn = px.histogram(df, x="Churn_Probability", nbins=30)
st.plotly_chart(fig_churn, use_container_width=True)

# ==============================
# 🥧 CHURN PIE CHART
# ==============================
st.subheader("🥧 Churn Prediction Breakdown")

fig_pie = px.pie(df, names="Churn_Predicted", title="Churn vs No Churn")
st.plotly_chart(fig_pie, use_container_width=True)

# ==============================
# 📊 FEATURE IMPORTANCE
# ==============================
st.subheader("📊 Feature Importance")

if hasattr(model, "coef_"):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.coef_[0]
    }).sort_values(by="Importance", ascending=False)
else:
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

fig_imp = px.bar(
    importance.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Influential Features"
)

st.plotly_chart(fig_imp, use_container_width=True)

# ==============================
# 💡 SMART INSIGHTS (USP)
# ==============================
st.subheader("💡 Business Insights")

for ins in insights:
    st.success(ins)

# ==============================
# 🔮 USER INPUT (INTERACTIVE)
# ==============================
st.subheader("🔮 Predict for New Customer")

tenure = st.slider("Tenure", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0, 150, 70)

if st.button("Predict"):
    input_data = [[tenure, monthly] + [0]*(X.shape[1]-2)]
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Prediction: {'Churn' if pred==1 else 'No Churn'}")
    st.write(f"Probability: {prob:.2f}")

# ==============================
# 🔍 DATA PREVIEW
# ==============================
with st.expander("🔍 View Dataset"):
    st.dataframe(df.head(50))