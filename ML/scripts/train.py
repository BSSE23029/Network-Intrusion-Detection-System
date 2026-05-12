import os
import sys
import traceback
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Supervised Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Unsupervised Models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


try:

    # Load training and test data

    # Supervised training data (SMOTE applied)
    X_train_sup = np.load("data/processed/X_train.npy")
    y_train_sup = np.load("data/processed/y_train.npy")

    # Original training data (no SMOTE)
    X_train_unsup = np.load("data/processed/X_train_original.npy")
    y_train_original = np.load("data/processed/y_train_original.npy")

    # Test data
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Load scaler and features

    scaler = joblib.load("models/scaler.pkl")

    features = joblib.load("models/features.pkl")

    # Convert arrays into DataFrames

    X_train_sup_df = pd.DataFrame(
        X_train_sup,
        columns=features
    )

    X_train_unsup_df = pd.DataFrame(
        X_train_unsup,
        columns=features
    )

    X_test_df = pd.DataFrame(
        X_test,
        columns=features
    )

    print(
        "\nSupervised training rows (full train, SMOTE):",
        len(X_train_sup_df)
    )

    print(
        "Test rows (held out):",
        len(X_test_df)
    )

    # Default Random Forest parameters

    rf_defaults = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced"
    }

    # Default XGBoost parameters

    xgb_defaults = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6,
        "eval_metric": "logloss",
        "random_state": 42
    }

    # Load tuned RF parameters if available

    rf_params = dict(rf_defaults)

    if os.path.exists("models/best_rf_params.pkl"):

        loaded = joblib.load(
            "models/best_rf_params.pkl"
        )

        rf_params.update(loaded)

        rf_params.setdefault("random_state", 42)
        rf_params.setdefault("n_jobs", -1)
        rf_params.setdefault("class_weight", "balanced")

        print(
            "\nLoaded best RF params from models/best_rf_params.pkl"
        )

    # Load tuned XGB parameters if available

    xgb_params = dict(xgb_defaults)

    if os.path.exists("models/best_xgb_params.pkl"):

        loaded = joblib.load(
            "models/best_xgb_params.pkl"
        )

        xgb_params.update(loaded)

        xgb_params.setdefault("eval_metric", "logloss")

        xgb_params.setdefault("random_state", 42)

        print(
            "Loaded best XGB params from models/best_xgb_params.pkl"
        )

    # Handle class imbalance for XGBoost

    n_neg = int((y_train_original == 0).sum())

    n_pos = int((y_train_original == 1).sum())

    if n_pos > 0:

        xgb_params["scale_pos_weight"] = n_neg / n_pos

    # Store model results

    results = []

    # Random Forest

    rf = RandomForestClassifier(**rf_params)

    rf.fit(X_train_sup_df, y_train_sup)

    rf_pred = rf.predict(X_test_df)

    rf_accuracy = accuracy_score(y_test, rf_pred)

    rf_precision = precision_score(
        y_test,
        rf_pred,
        zero_division=0
    )

    rf_recall = recall_score(
        y_test,
        rf_pred,
        zero_division=0
    )

    rf_f1 = f1_score(
        y_test,
        rf_pred,
        zero_division=0
    )

    results.append({
        "Model": "Random Forest",
        "Accuracy": rf_accuracy,
        "Precision": rf_precision,
        "Recall": rf_recall,
        "F1 Score": rf_f1
    })

    print("\nRandom Forest")

    print(
        classification_report(
            y_test,
            rf_pred,
            zero_division=0
        )
    )

    # XGBoost

    xgb = XGBClassifier(**xgb_params)

    xgb.fit(X_train_sup_df, y_train_sup)

    xgb_pred = xgb.predict(X_test_df)

    xgb_accuracy = accuracy_score(y_test, xgb_pred)

    xgb_precision = precision_score(
        y_test,
        xgb_pred,
        zero_division=0
    )

    xgb_recall = recall_score(
        y_test,
        xgb_pred,
        zero_division=0
    )

    xgb_f1 = f1_score(
        y_test,
        xgb_pred,
        zero_division=0
    )

    results.append({
        "Model": "XGBoost",
        "Accuracy": xgb_accuracy,
        "Precision": xgb_precision,
        "Recall": xgb_recall,
        "F1 Score": xgb_f1
    })

    print("\nXGBoost")

    print(
        classification_report(
            y_test,
            xgb_pred,
            zero_division=0
        )
    )

    # Logistic Regression

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    lr.fit(X_train_sup_df, y_train_sup)

    lr_pred = lr.predict(X_test_df)

    lr_accuracy = accuracy_score(y_test, lr_pred)

    lr_precision = precision_score(
        y_test,
        lr_pred,
        zero_division=0
    )

    lr_recall = recall_score(
        y_test,
        lr_pred,
        zero_division=0
    )

    lr_f1 = f1_score(
        y_test,
        lr_pred,
        zero_division=0
    )

    results.append({
        "Model": "Logistic Regression",
        "Accuracy": lr_accuracy,
        "Precision": lr_precision,
        "Recall": lr_recall,
        "F1 Score": lr_f1
    })

    print("\nLogistic Regression")

    print(
        classification_report(
            y_test,
            lr_pred,
            zero_division=0
        )
    )

    # Isolation Forest

    attack_rate = float(np.mean(y_train_original))

    contamination = min(
        max(attack_rate, 0.01),
        0.5
    )

    iso = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    # Train only on normal traffic
    X_normal = X_train_unsup_df[
        y_train_original == 0
    ]

    iso.fit(X_normal)

    iso_pred = iso.predict(X_test_df)

    # Convert:
    # -1 = Attack
    #  1 = Normal

    iso_pred = np.where(
        iso_pred == -1,
        1,
        0
    )

    iso_accuracy = accuracy_score(y_test, iso_pred)

    iso_precision = precision_score(
        y_test,
        iso_pred,
        zero_division=0
    )

    iso_recall = recall_score(
        y_test,
        iso_pred,
        zero_division=0
    )

    iso_f1 = f1_score(
        y_test,
        iso_pred,
        zero_division=0
    )

    results.append({
        "Model": "Isolation Forest",
        "Accuracy": iso_accuracy,
        "Precision": iso_precision,
        "Recall": iso_recall,
        "F1 Score": iso_f1
    })

    print("\nIsolation Forest")

    print(
        classification_report(
            y_test,
            iso_pred,
            zero_division=0
        )
    )

    # One-Class SVM

    ocsvm = OneClassSVM(
        nu=min(contamination, 0.99),
        kernel="rbf",
        gamma="scale"
    )

    ocsvm.fit(X_normal)

    svm_pred = ocsvm.predict(X_test_df)

    svm_pred = np.where(
        svm_pred == -1,
        1,
        0
    )

    svm_accuracy = accuracy_score(y_test, svm_pred)

    svm_precision = precision_score(
        y_test,
        svm_pred,
        zero_division=0
    )

    svm_recall = recall_score(
        y_test,
        svm_pred,
        zero_division=0
    )

    svm_f1 = f1_score(
        y_test,
        svm_pred,
        zero_division=0
    )

    results.append({
        "Model": "One-Class SVM",
        "Accuracy": svm_accuracy,
        "Precision": svm_precision,
        "Recall": svm_recall,
        "F1 Score": svm_f1
    })

    print("\nOne-Class SVM")

    print(
        classification_report(
            y_test,
            svm_pred,
            zero_division=0
        )
    )

    # Compare all models

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    print("\nModel Comparison\n")

    print(results_df)

    # Best model

    print("\nBest Model:")

    best_model = results_df.iloc[0]

    print(best_model)

    # Save comparison results

    os.makedirs("data", exist_ok=True)

    results_df.to_csv(
        "data/model_comparison.csv",
        index=False
    )

    print("\nTraining complete. Results saved!")

    # Save best model

    model_map = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "Logistic Regression": lr,
        "Isolation Forest": iso,
        "One-Class SVM": ocsvm
    }

    best_model_object = model_map[
        best_model["Model"]
    ]

    joblib.dump(
        best_model_object,
        "models/best_model.pkl"
    )

    print("\nBest model saved successfully")


except Exception as e:

    print("\nERROR in train.py:")

    print(e)

    traceback.print_exc()

    sys.exit(1)