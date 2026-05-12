import os
import sys
import traceback
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", None)

# Working set size (faster training; not the full CICIDS CSV)
SAMPLE_SIZE = 200_000

try:

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Load
    df = pd.read_csv("data/cicids2017_cleaned.csv")
    print("Dataset Loaded Successfully")
    print(df.head())
    print("\nShape:", df.shape)

    # Cleaning (full data first so rare attack rows survive cleaning before we sample)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print("\nAfter cleaning:", df.shape)

    # Stratified sample: keep multiclass "Attack Type" proportions (helps rare attacks)
    n = len(df)
    if n > SAMPLE_SIZE:
        try:
            df, _ = train_test_split(
                df,
                train_size=SAMPLE_SIZE,
                stratify=df["Attack Type"],
                random_state=42,
            )
            print(
                f"\nStratified sample: using {SAMPLE_SIZE} rows by Attack Type "
                "(remainder discarded for speed)."
            )
        except ValueError as err:
            # e.g. some Attack Type has too few rows for sklearn stratify
            print(
                f"\nStratified sample failed ({err}); "
                "fallback: random sample without stratify."
            )
            df = df.sample(n=SAMPLE_SIZE, random_state=42)
    elif n == SAMPLE_SIZE:
        print(f"\nRow count equals SAMPLE_SIZE ({SAMPLE_SIZE}); using all rows.")
    else:
        print(
            f"\nWarning: only {n} rows after cleaning (less than {SAMPLE_SIZE}). "
            "Using all available rows."
        )

    print("\nAttack Type counts in working set:\n")
    print(df["Attack Type"].value_counts())

    # Target Encoding
    print("\nAttack Distribution:\n")
    print(df["Attack Type"].value_counts())
    df["target"] = df["Attack Type"].apply(
        lambda x: 0 if x == "Normal Traffic" else 1
    )

    # Dropping original target
    df.drop("Attack Type", axis=1, inplace=True)

    # Features
    X = df.drop("target", axis=1)
    y = df["target"]

    # Drop common leaky / ID columns if present (helps realistic NIDS scores)
    drop_if_exists = [
        "Flow ID", "Timestamp", "Src IP", "Dst IP",
        "Source IP", "Destination IP"
    ]
    to_drop = [c for c in drop_if_exists if c in X.columns]
    if to_drop:
        print("\nDropping columns (reduce leakage):", to_drop)
        X = X.drop(columns=to_drop)

    # Keeping only numeric columns
    X = X.select_dtypes(include=[np.number])
    feature_columns = X.columns.tolist()
    print("\nFinal Features Count:", len(feature_columns))
    print("\nFeature names (audit for leakage):\n", feature_columns)
    print("\nAttack Percentage:", y.mean())

    # Train/test split on the working set only.
    # Supervised models later fit on the FULL train portion (after SMOTE), not on test.
    # Test rows are held out for honest metrics in train.py / evaluate.py.
    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print("\nTrain Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Keep original scaled train for unsupervised models (no SMOTE)
    X_train_original = X_train_scaled.copy()
    y_train_original = y_train.to_numpy().copy()

    # SMOTE only on supervised training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print("\nAfter SMOTE:")
    print(pd.Series(y_train_smote).value_counts())

    # Saving scaler and features
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_columns, "models/features.pkl")

    # Supervised: SMOTE resampled (same names as before for tuning / feature_selection)
    np.save("data/processed/X_train.npy", X_train_smote)
    np.save("data/processed/y_train.npy", y_train_smote)
    # Unsupervised: scaled only, real class balance
    np.save("data/processed/X_train_original.npy", X_train_original)
    np.save("data/processed/y_train_original.npy", y_train_original)

    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_test.npy", y_test)
    print("\nPreprocessing Complete")

except Exception as e:

    print(f"\nError in preprocess.py: {e}")
    traceback.print_exc()
    sys.exit(1)
