import sys
import traceback
import numpy as np
import pandas as pd
import joblib

try:

    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")

    # testing
    sample_input = np.zeros(len(features)).reshape(1, -1)

    if sample_input.shape[1] != len(features):
        raise ValueError(
            f"Expected {len(features)} features, got {sample_input.shape[1]}"
        )

    sample_scaled = scaler.transform(sample_input)

    sample_df = pd.DataFrame(sample_scaled, columns=features)

    prediction = model.predict(sample_df)[0]

    print("\nPrediction Result:")
    print("Attack" if prediction == 1 else "Normal")

    if hasattr(model, "predict_proba"):
        conf = model.predict_proba(sample_df)[0, 1]
        print("Confidence (attack probability):", round(float(conf), 4))

except Exception as e:
    print(f"\nError in predict.py: {e}")
    traceback.print_exc()
    sys.exit(1)
