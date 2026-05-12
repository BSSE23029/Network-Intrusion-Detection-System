import os
import sys
import traceback
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest

# Trains an Isolation Forest on normal-only scaled training rows so it learns
# the statistical fingerprint of legitimate traffic. At inference the backend
# uses predict(): -1 = OOD/anomalous, +1 = looks like training data. Crafted
# or otherwise-unseen rows trip the -1 branch even if XGB is confidently
# Normal, which is the case the user keeps hitting in test_20.csv.

try:

    os.makedirs("models", exist_ok=True)

    X_train_original = np.load("data/processed/X_train_original.npy")
    y_train_original = np.load("data/processed/y_train_original.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    print(f"Full training rows: {X_train_original.shape[0]}")

    X_normal = X_train_original[y_train_original == 0]
    print(f"Normal-only rows used for OOD training: {X_normal.shape[0]}")

    # contamination=0.10 is intentionally aggressive: anything sitting in the
    # bottom 10% of "normal-likeness" gets flagged. This catches realistic-
    # looking crafted rows that hand-typed values often produce (they tend to
    # cluster near the boundary of the normal-traffic feature region). The
    # tradeoff is ~10% of legitimate-but-unusual normal traffic being marked
    # Uncertain in the UI, which is acceptable for an analyst-facing tool.
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_normal)

    train_pred = iso.predict(X_normal)
    train_ood_rate = (train_pred == -1).mean()
    print(f"Train OOD rate on the fit set: {train_ood_rate * 100:.3f}%")

    test_pred = iso.predict(X_test)
    test_normal_ood = (test_pred[y_test == 0] == -1).mean() if (y_test == 0).any() else 0.0
    test_attack_ood = (test_pred[y_test == 1] == -1).mean() if (y_test == 1).any() else 0.0
    print(
        f"Test set OOD rate: normal={test_normal_ood * 100:.2f}%, "
        f"attack={test_attack_ood * 100:.2f}%"
    )
    print(
        "  (high attack-OOD rate is fine; we only override to Uncertain when "
        "XGB also predicts Normal.)"
    )

    out_path = "models/ood_detector.pkl"
    joblib.dump(iso, out_path)
    print(f"\nOOD detector saved to {out_path}")

except Exception as e:
    print(f"\nError in train_ood.py: {e}")
    traceback.print_exc()
    sys.exit(1)
