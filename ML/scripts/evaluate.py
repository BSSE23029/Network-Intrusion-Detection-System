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

    os.makedirs("reports", exist_ok=True)

    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    features = joblib.load("models/features.pkl")

    model = joblib.load("models/best_model.pkl")

    X_test_df = pd.DataFrame(X_test, columns=features)

    pred = model.predict(X_test_df)

    print("\n--- Test set metrics ---")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)

    print("\nConfusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix_best_model.png", dpi=200)
    plt.close()

    # ROC AUC
    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(X_test_df)[:, 1]

        score = roc_auc_score(y_test, probs)

        print("\nROC-AUC:", score)

        RocCurveDisplay.from_predictions(y_test, probs)
        plt.tight_layout()
        plt.savefig("reports/roc_curve_best_model.png", dpi=200)
        plt.close()

except Exception as e:
    print(f"\nError in evaluate.py: {e}")
    traceback.print_exc()
    sys.exit(1)
