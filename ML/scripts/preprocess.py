import os
import sys
import traceback
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# Show all columns when printing DataFrame
pd.set_option("display.max_columns", None)


# Number of rows we want to use
SAMPLE_SIZE = 200_000


try:

    # Create required folders

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Load dataset

    df = pd.read_csv("data/cicids2017_cleaned.csv")

    print("Dataset Loaded Successfully")
    print(df.head())
    print("\nShape:", df.shape)

    # Clean dataset

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.drop_duplicates(inplace=True)

    df.dropna(inplace=True)

    print("\nAfter cleaning:", df.shape)

    # Take sample from dataset

    total_rows = len(df)

    if total_rows > SAMPLE_SIZE:

        try:

            # Take 200,000 rows while keeping Attack Type ratio same
            df, unused_df = train_test_split(
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

            # If stratified sampling fails,
            # use normal random sampling

            print(
                f"\nStratified sample failed ({err}); "
                "fallback: random sample without stratify."
            )

            df = df.sample(
                n=SAMPLE_SIZE,
                random_state=42
            )

    elif total_rows == SAMPLE_SIZE:

        print(
            f"\nRow count equals SAMPLE_SIZE ({SAMPLE_SIZE}); using all rows."
        )

    else:

        print(
            f"\nWarning: only {total_rows} rows after cleaning "
            f"(less than {SAMPLE_SIZE}). Using all available rows."
        )

    # Show attack type counts

    print("\nAttack Type counts in working set:\n")
    print(df["Attack Type"].value_counts())

    print("\nAttack Distribution:\n")
    print(df["Attack Type"].value_counts())

    # Create target column

    # Normal Traffic = 0
    # Attack traffic = 1

    df["target"] = df["Attack Type"].apply(
        lambda x: 0 if x == "Normal Traffic" else 1
    )

    # Remove original target column

    df.drop(
        "Attack Type",
        axis=1,
        inplace=True
    )

    # Separate features and target

    X = df.drop(
        "target",
        axis=1
    )

    y = df["target"]

    # Remove leaky / ID columns

    drop_if_exists = [
        "Flow ID",
        "Timestamp",
        "Src IP",
        "Dst IP",
        "Source IP",
        "Destination IP"
    ]

    to_drop = []

    for column in drop_if_exists:

        if column in X.columns:

            to_drop.append(column)

    if to_drop:

        print("\nDropping columns (reduce leakage):", to_drop)

        X = X.drop(
            columns=to_drop
        )

    # Keep only numeric columns

    X = X.select_dtypes(
        include=[np.number]
    )

    feature_columns = X.columns.tolist()

    print("\nFinal Features Count:", len(feature_columns))
    print("\nFeature names (audit for leakage):\n", feature_columns)
    print("\nAttack Percentage:", y.mean())

    # Split data into train and test

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    # Scale features

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Keep original train data

    X_train_original = X_train_scaled.copy()

    y_train_original = y_train.to_numpy().copy()

    # Apply SMOTE on training data

    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_scaled,
        y_train
    )

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_smote).value_counts())

    # Save scaler and feature names

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    joblib.dump(
        feature_columns,
        "models/features.pkl"
    )

    # Save supervised training data

    np.save(
        "data/processed/X_train.npy",
        X_train_smote
    )

    np.save(
        "data/processed/y_train.npy",
        y_train_smote
    )

    # Save original training data

    np.save(
        "data/processed/X_train_original.npy",
        X_train_original
    )

    np.save(
        "data/processed/y_train_original.npy",
        y_train_original
    )

    # Save test data

    np.save(
        "data/processed/X_test.npy",
        X_test_scaled
    )

    np.save(
        "data/processed/y_test.npy",
        y_test
    )

    print("\nPreprocessing Complete")


except Exception as e:

    print(f"\nError in preprocess.py: {e}")

    traceback.print_exc()

    sys.exit(1)