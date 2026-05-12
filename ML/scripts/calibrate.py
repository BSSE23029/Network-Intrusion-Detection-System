import os
import sys
import traceback
import numpy as np
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

# FrozenEstimator (sklearn >= 1.6) replaces the legacy cv="prefit" idiom for
# wrapping an already-fitted base model in CalibratedClassifierCV.
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

# Fits an isotonic probability calibrator on top of the already-trained
# best_model.pkl. Calibration uses the REAL-distribution (non-SMOTE)
# training data so output probabilities reflect true class frequencies,
# which is what the "Uncertain" abstention band relies on.

try:

    base_path = "models/best_model.pkl"
    out_path = "models/calibrated_model.pkl"

    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Missing {base_path}. Run train.py first."
        )

    base = joblib.load(base_path)

    X_orig = np.load("data/processed/X_train_original.npy")
    y_orig = np.load("data/processed/y_train_original.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    print(f"Base model: {type(base).__name__}")
    print(f"Calibration set (non-SMOTE) shape: {X_orig.shape}")

    if not hasattr(base, "predict_proba"):
        print(
            "WARNING: base model does not support predict_proba "
            "(e.g. Isolation Forest / One-Class SVM). "
            "Calibration requires probabilistic output. "
            "Re-train so that a supervised model wins, then re-run."
        )
        sys.exit(2)

    raw_probs = base.predict_proba(X_test)[:, 1]
    raw_brier = brier_score_loss(y_test, raw_probs)
    raw_ll = log_loss(y_test, raw_probs, labels=[0, 1])

    if FrozenEstimator is not None:
        # sklearn >= 1.6: freeze the base estimator so CalibratedClassifierCV
        # only fits the calibration layer (no re-training).
        cal = CalibratedClassifierCV(
            estimator=FrozenEstimator(base),
            method="isotonic",
            cv=5,
        )
    else:
        # sklearn < 1.6 fallback (legacy API).
        cal = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv="prefit",
        )
    cal.fit(X_orig, y_orig)

    cal_probs = cal.predict_proba(X_test)[:, 1]
    cal_brier = brier_score_loss(y_test, cal_probs)
    cal_ll = log_loss(y_test, cal_probs, labels=[0, 1])

    print("\nBefore calibration:")
    print(f"  Brier score: {raw_brier:.6f}")
    print(f"  Log loss:    {raw_ll:.6f}")
    print("After  calibration:")
    print(f"  Brier score: {cal_brier:.6f}")
    print(f"  Log loss:    {cal_ll:.6f}")

    middle_band = ((cal_probs >= 0.4) & (cal_probs <= 0.6)).sum()
    print(
        f"\nRows in 0.4-0.6 uncertainty band on test set: "
        f"{middle_band} / {len(cal_probs)} "
        f"({100 * middle_band / len(cal_probs):.3f}%)"
    )

    joblib.dump(cal, out_path)
    print(f"\nCalibrated model saved to {out_path}")

except Exception as e:
    print(f"\nError in calibrate.py: {e}")
    traceback.print_exc()
    sys.exit(1)
