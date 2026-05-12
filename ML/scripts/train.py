import os
import sys
import traceback
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier  # Supervised Models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest  # Unsupervised Models
from sklearn.svm import OneClassSVM

try:
    # Optuna tuned on 30% of this train set; here we fit on 100% of train (SMOTE), not test.
    # X_test is only for evaluation — never merged into training.
    # Load data (supervised = SMOTE; test = never SMOTE)
    X_train_sup = np.load("data/processed/X_train.npy")
    X_train_unsup = np.load("data/processed/X_train_original.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train_sup = np.load("data/processed/y_train.npy")
    y_train_original = np.load("data/processed/y_train_original.npy")
    y_test = np.load("data/processed/y_test.npy")

    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")

    # Supervised models use SMOTE-balanced train
    X_train_sup_df = pd.DataFrame(X_train_sup, columns=features)
    X_test_df = pd.DataFrame(X_test, columns=features)
    # Unsupervised: real distribution, scaled (no SMOTE)
    X_train_unsup_df = pd.DataFrame(X_train_unsup, columns=features)

    print("\nSupervised training rows (full train, SMOTE):", len(X_train_sup_df))
    print("Test rows (held out):", len(X_test_df))

    # Tuned params if they exist (from hyperparameter_tuning.py)
    rf_defaults = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced"
    }
    xgb_defaults = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6,
        "eval_metric": "logloss",
        "random_state": 42
    }

    rf_params = dict(rf_defaults)
    if os.path.exists("models/best_rf_params.pkl"):
        loaded = joblib.load("models/best_rf_params.pkl")
        rf_params.update(loaded)
        rf_params.setdefault("random_state", 42)
        rf_params.setdefault("n_jobs", -1)
        rf_params.setdefault("class_weight", "balanced")
        print("\nLoaded best RF params from models/best_rf_params.pkl")

    xgb_params = dict(xgb_defaults)
    if os.path.exists("models/best_xgb_params.pkl"):
        loaded = joblib.load("models/best_xgb_params.pkl")
        xgb_params.update(loaded)
        xgb_params.setdefault("eval_metric", "logloss")
        xgb_params.setdefault("random_state", 42)
        print("Loaded best XGB params from models/best_xgb_params.pkl")

    # XGBoost: class imbalance backup (original train ratio, not SMOTE)
    n_neg = int((y_train_original == 0).sum())
    n_pos = int((y_train_original == 1).sum())
    if n_pos > 0:
        xgb_params["scale_pos_weight"] = n_neg / n_pos

    results = []

    # Random Forest (supervised)
    rf = RandomForestClassifier(**rf_params)

    rf.fit(X_train_sup_df, y_train_sup)
    rf_pred = rf.predict(X_test_df)

    results.append({
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, rf_pred),
        "Precision": precision_score(y_test, rf_pred, zero_division=0),
        "Recall": recall_score(y_test, rf_pred, zero_division=0),
        "F1 Score": f1_score(y_test, rf_pred, zero_division=0)
    })

    print("\nRandom Forest")
    print(classification_report(y_test, rf_pred, zero_division=0))

    # XGBoost (supervised)
    xgb = XGBClassifier(**xgb_params)

    xgb.fit(X_train_sup_df, y_train_sup)
    xgb_pred = xgb.predict(X_test_df)

    results.append({
        "Model": "XGBoost",
        "Accuracy": accuracy_score(y_test, xgb_pred),
        "Precision": precision_score(y_test, xgb_pred, zero_division=0),
        "Recall": recall_score(y_test, xgb_pred, zero_division=0),
        "F1 Score": f1_score(y_test, xgb_pred, zero_division=0)
    })

    print("\nXGBoost")
    print(classification_report(y_test, xgb_pred, zero_division=0))

    # Logistic Regression (supervised)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")

    lr.fit(X_train_sup_df, y_train_sup)
    lr_pred = lr.predict(X_test_df)

    results.append({
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, lr_pred),
        "Precision": precision_score(y_test, lr_pred, zero_division=0),
        "Recall": recall_score(y_test, lr_pred, zero_division=0),
        "F1 Score": f1_score(y_test, lr_pred, zero_division=0)
    })

    print("\nLogistic Regression")
    print(classification_report(y_test, lr_pred, zero_division=0))

    # Isolation Forest: train only on normal traffic, original scaled data (no SMOTE)
    attack_rate = float(np.mean(y_train_original))
    contamination = min(max(attack_rate, 0.01), 0.5)
    iso = IsolationForest(contamination=contamination, random_state=42)

    X_normal = X_train_unsup_df[y_train_original == 0]
    iso.fit(X_normal)
    iso_pred = iso.predict(X_test_df)

    iso_pred = np.where(iso_pred == -1, 1, 0)

    results.append({
        "Model": "Isolation Forest",
        "Accuracy": accuracy_score(y_test, iso_pred),
        "Precision": precision_score(y_test, iso_pred, zero_division=0),
        "Recall": recall_score(y_test, iso_pred, zero_division=0),
        "F1 Score": f1_score(y_test, iso_pred, zero_division=0)
    })

    print("\nIsolation Forest")
    print(classification_report(y_test, iso_pred, zero_division=0))

    # One-Class SVM: same as IF — normal-only train, no SMOTE
    ocsvm = OneClassSVM(nu=min(contamination, 0.99), kernel="rbf", gamma="scale")

    ocsvm.fit(X_normal)
    svm_pred = ocsvm.predict(X_test_df)

    svm_pred = np.where(svm_pred == -1, 1, 0)

    results.append({
        "Model": "One-Class SVM",
        "Accuracy": accuracy_score(y_test, svm_pred),
        "Precision": precision_score(y_test, svm_pred, zero_division=0),
        "Recall": recall_score(y_test, svm_pred, zero_division=0),
        "F1 Score": f1_score(y_test, svm_pred, zero_division=0)
    })

    print("\nOne-Class SVM")
    print(classification_report(y_test, svm_pred, zero_division=0))

    # Model Comparison
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1 Score", ascending=False)

    print("\nModel Comparison\n")
    print(results_df)

    print("\nBest Model:")
    best_model = results_df.iloc[0]
    print(best_model)

    os.makedirs("data", exist_ok=True)
    # SAVE RESULTS
    results_df.to_csv("data/model_comparison.csv", index=False)

    print("\nTraining complete. Results saved!")

    # SAVE BEST MODEL SAFELY
    model_map = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "Logistic Regression": lr,
        "Isolation Forest": iso,
        "One-Class SVM": ocsvm
    }

    best_model_object = model_map[best_model["Model"]]

    joblib.dump(best_model_object, "models/best_model.pkl")

    print("\nBest model saved successfully")

except Exception as e:
    print("\nERROR in train.py:")
    print(e)
    traceback.print_exc()
    sys.exit(1)
