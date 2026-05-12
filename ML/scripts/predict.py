import sys
import traceback
import numpy as np
import pandas as pd
import joblib


try:

    # Load saved files

    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")

    # Create sample input

    # Create one test row with all feature values as 0
    sample_input = np.zeros(len(features))

    # Convert it into 2D shape
    sample_input = sample_input.reshape(1, -1)

    # Check feature count

    expected_features = len(features)
    actual_features = sample_input.shape[1]

    if actual_features != expected_features:

        raise ValueError(
            f"Expected {expected_features} features, got {actual_features}"
        )

    # Scale sample input

    sample_scaled = scaler.transform(sample_input)

    # Convert to DataFrame

    sample_df = pd.DataFrame(
        sample_scaled,
        columns=features
    )

    # Make prediction

    prediction = model.predict(sample_df)[0]

    # Print prediction result

    print("\nPrediction Result:")

    if prediction == 1:
        print("Attack")
    else:
        print("Normal")

    # Print confidence if available

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(sample_df)

        attack_probability = probabilities[0, 1]

        confidence = round(float(attack_probability), 4)

        print("Confidence (attack probability):", confidence)


except Exception as e:

    print(f"\nError in predict.py: {e}")

    traceback.print_exc()

    sys.exit(1)