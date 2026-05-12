import os
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier


try:

    # Create reports folder

    os.makedirs("reports", exist_ok=True)

    # Load training data

    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")

    # Load feature names

    features = joblib.load("models/features.pkl")

    # Create Random Forest model

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Train Random Forest model

    rf.fit(X_train, y_train)

    # Get feature importance

    importance = rf.feature_importances_

    # Create DataFrame

    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": importance
        }
    )

    # Sort features by importance

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Print top 20 features

    print("\nTop 20 Important Features:\n")

    top_20_features = importance_df.head(20)

    print(top_20_features)

    # Save feature importance CSV

    importance_df.to_csv(
        "reports/feature_importance.csv",
        index=False
    )

    # Plot top 20 features

    top_20_features.plot(
        x="Feature",
        y="Importance",
        kind="bar",
        figsize=(14, 6)
    )

    plt.title("Top 20 Important Features")

    plt.tight_layout()

    plt.savefig(
        "reports/top20_feature_importance.png",
        dpi=200
    )

    plt.close()


except Exception as e:

    print(f"\nError in feature_selection.py: {e}")

    traceback.print_exc()

    sys.exit(1)