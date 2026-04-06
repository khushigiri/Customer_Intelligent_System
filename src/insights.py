def generate_insights(df):
    insights = []

    # ==============================
    # 🔹 Cluster-wise churn analysis
    # ==============================
    cluster_stats = df.groupby('Cluster')['Churn_Probability'].mean()

    for cluster, prob in cluster_stats.items():
        if prob > 0.6:
            insights.append(
                f"Cluster {cluster}: Very high churn risk → Offer discounts or retention campaigns"
            )
        elif prob > 0.4:
            insights.append(
                f"Cluster {cluster}: Moderate churn risk → Engage with personalized offers"
            )
        else:
            insights.append(
                f"Cluster {cluster}: Low churn risk → Maintain engagement"
            )

    # ==============================
    # 🔥 Advanced Insight (SAFE)
    # ==============================
    if 'monthly_charges' in df.columns:
        high_value = df[
            (df['monthly_charges'] > df['monthly_charges'].mean()) &
            (df['Churn_Probability'] > 0.5)
        ]

        if len(high_value) > 0:
            insights.append(
                "High-value customers are at risk → prioritize retention strategies"
            )

    return insights