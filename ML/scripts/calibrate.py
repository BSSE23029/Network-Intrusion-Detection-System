import os
import sys
import traceback
import numpy as np
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss


# Try to import FrozenEstimator
try:
    from sklearn.frozen import FrozenEstimator

except ImportError:
    FrozenEstimator = None


try:

    # File paths

    base_path = "models/best_model.pkl"
    out_path = "models/calibrated_model.pkl"

    # Check model file exists

    if not os.path.exists(base_path):

        raise FileNotFoundError(
            f"Missing {base_path}. Run train.py first."
        )

    # Load saved model

    base = joblib.load(base_path)

    # Load saved data

    X_orig = np.load("data/processed/X_train_original.npy")
    y_orig = np.load("data/processed/y_train_original.npy")

    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    print(f"Base model: {type(base).__name__}")
    print(f"Calibration set (non-SMOTE) shape: {X_orig.shape}")

    # Check predict_proba support

    if not hasattr(base, "predict_proba"):

        print(
            "WARNING: base model does not support predict_proba "
            "(e.g. Isolation Forest / One-Class SVM). "
            "Calibration requires probabilistic output. "
            "Re-train so that a supervised model wins, then re-run."
        )

        sys.exit(2)

    # Test model before calibration

    raw_probs = base.predict_proba(X_test)[:, 1]

    raw_brier = brier_score_loss(y_test, raw_probs)

    raw_ll = log_loss(
        y_test,
        raw_probs,
        labels=[0, 1]
    )

    # Create calibration model

    if FrozenEstimator is not None:

        # For sklearn version >= 1.6
        cal = CalibratedClassifierCV(
            estimator=FrozenEstimator(base),
            method="isotonic",
            cv=5,
        )

    else:

        # For sklearn version < 1.6
        cal = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv="prefit",
        )

    # Train calibration layer

    cal.fit(X_orig, y_orig)

    # Test model after calibration

    cal_probs = cal.predict_proba(X_test)[:, 1]

    cal_brier = brier_score_loss(y_test, cal_probs)

    cal_ll = log_loss(
        y_test,
        cal_probs,
        labels=[0, 1]
    )

    # Print comparison

    print("\nBefore calibration:")
    print(f"  Brier score: {raw_brier:.6f}")
    print(f"  Log loss:    {raw_ll:.6f}")

    print("After  calibration:")
    print(f"  Brier score: {cal_brier:.6f}")
    print(f"  Log loss:    {cal_ll:.6f}")

    # Count uncertain rows

    middle_band = 0

    for prob in cal_probs:

        if prob >= 0.4 and prob <= 0.6:
            middle_band = middle_band + 1

    total_rows = len(cal_probs)

    percentage = 100 * middle_band / total_rows

    print(
        f"\nRows in 0.4-0.6 uncertainty band on test set: "
        f"{middle_band} / {total_rows} "
        f"({percentage:.3f}%)"
    )

    # Save calibrated model

    joblib.dump(cal, out_path)

    print(f"\nCalibrated model saved to {out_path}")


except Exception as e:

    print(f"\nError in calibrate.py: {e}")

    traceback.print_exc()

    sys.exit(1)