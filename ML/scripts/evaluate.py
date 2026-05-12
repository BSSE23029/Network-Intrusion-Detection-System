import os
import sys
import traceback
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)


try:

    # Create reports folder

    os.makedirs("reports", exist_ok=True)

    # Load test data

    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Load feature names

    features = joblib.load("models/features.pkl")

    # Load trained model

    model = joblib.load("models/best_model.pkl")

    # Convert test data into DataFrame

    X_test_df = pd.DataFrame(
        X_test,
        columns=features
    )

    # Make predictions

    pred = model.predict(X_test_df)

    # Print test metrics

    print("\n--- Test set metrics ---")

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Confusion Matrix

    cm = confusion_matrix(y_test, pred)

    print("\nConfusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.tight_layout()

    plt.savefig(
        "reports/confusion_matrix_best_model.png",
        dpi=200
    )

    plt.close()

    # ROC AUC

    if hasattr(model, "predict_proba"):

        # Get probability of Attack class
        probs = model.predict_proba(X_test_df)[:, 1]

        score = roc_auc_score(y_test, probs)

        print("\nROC-AUC:", score)

        RocCurveDisplay.from_predictions(
            y_test,
            probs
        )

        plt.tight_layout()

        plt.savefig(
            "reports/roc_curve_best_model.png",
            dpi=200
        )

        plt.close()


except Exception as e:

    print(f"\nError in evaluate.py: {e}")

    traceback.print_exc()

    sys.exit(1)