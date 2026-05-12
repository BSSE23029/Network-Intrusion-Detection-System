import os
import sys
import traceback
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest


try:

    # Create models folder

    os.makedirs("models", exist_ok=True)

    # Load saved data

    X_train_original = np.load("data/processed/X_train_original.npy")
    y_train_original = np.load("data/processed/y_train_original.npy")

    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    print(f"Full training rows: {X_train_original.shape[0]}")

    # Keep only normal traffic rows

    X_normal = X_train_original[y_train_original == 0]

    print(
        f"Normal-only rows used for OOD training: {X_normal.shape[0]}"
    )

    # Create Isolation Forest model

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )

    # Train OOD model

    iso.fit(X_normal)

    # Check OOD rate on training data

    train_pred = iso.predict(X_normal)

    train_ood_count = 0

    for value in train_pred:

        if value == -1:
            train_ood_count = train_ood_count + 1

    train_total = len(train_pred)

    train_ood_rate = train_ood_count / train_total

    print(
        f"Train OOD rate on the fit set: "
        f"{train_ood_rate * 100:.3f}%"
    )

    # Check OOD rate on test data

    test_pred = iso.predict(X_test)

    # Normal test rows
    normal_test_pred = test_pred[y_test == 0]

    if len(normal_test_pred) > 0:

        normal_ood_count = 0

        for value in normal_test_pred:

            if value == -1:
                normal_ood_count = normal_ood_count + 1

        test_normal_ood = normal_ood_count / len(normal_test_pred)

    else:

        test_normal_ood = 0.0

    # Attack test rows
    attack_test_pred = test_pred[y_test == 1]

    if len(attack_test_pred) > 0:

        attack_ood_count = 0

        for value in attack_test_pred:

            if value == -1:
                attack_ood_count = attack_ood_count + 1

        test_attack_ood = attack_ood_count / len(attack_test_pred)

    else:

        test_attack_ood = 0.0

    print(
        f"Test set OOD rate: normal={test_normal_ood * 100:.2f}%, "
        f"attack={test_attack_ood * 100:.2f}%"
    )

    print(
        "  (high attack-OOD rate is fine; we only override to Uncertain when "
        "XGB also predicts Normal.)"
    )

    # Save OOD detector

    out_path = "models/ood_detector.pkl"

    joblib.dump(
        iso,
        out_path
    )

    print(f"\nOOD detector saved to {out_path}")


except Exception as e:

    print(f"\nError in train_ood.py: {e}")

    traceback.print_exc()

    sys.exit(1)