import os
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier

try:

    os.makedirs("reports", exist_ok=True)

    # Loading data
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    features = joblib.load("models/features.pkl")

    # Training RF
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    # Feature importance
    importance = rf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 20 Important Features:\n")
    print(importance_df.head(20))

    # Saving feature importance
    importance_df.to_csv(
        "reports/feature_importance.csv",
        index=False
    )

    # Plotting
    importance_df.head(20).plot(
        x="Feature",
        y="Importance",
        kind="bar",
        figsize=(14, 6)
    )

    plt.title("Top 20 Important Features")
    plt.tight_layout()
    plt.savefig("reports/top20_feature_importance.png", dpi=200)
    plt.close()

except Exception as e:

    print(f"\nError in feature_selection.py: {e}")
    traceback.print_exc()
    sys.exit(1)
