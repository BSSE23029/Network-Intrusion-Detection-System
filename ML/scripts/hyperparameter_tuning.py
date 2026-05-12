import os
import sys
import traceback
import numpy as np
import optuna
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

try:

    os.makedirs("models", exist_ok=True)

    # Loading data
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")

    # Use 30% of training data for Optuna speed (final train uses full train in train.py)
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train,
        train_size=0.3,
        random_state=42,
        stratify=y_train
    )
    print(f"Tuning on {len(X_sample)} samples (30% of {len(X_train)} training rows)\n")

    # Stratified K-Fold with 3 splits for faster tuning
    skf = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    # Random Forest Optuna
    def rf_objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced"
        }
        model = RandomForestClassifier(**params)

        scores = cross_val_score(
            model,
            X_sample,
            y_sample,
            cv=skf,
            scoring="f1",
            n_jobs=-1
        )
        print(f"  RF trial F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores.mean()

    # XGBoost Optuna
    def xgb_objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eval_metric": "logloss",
            "random_state": 42
        }
        model = XGBClassifier(**params)

        scores = cross_val_score(
            model,
            X_sample,
            y_sample,
            cv=skf,
            scoring="f1",
            n_jobs=-1
        )
        print(f"  XGB trial F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores.mean()

    # RF tuning
    print("\nOptimizing Random Forest...\n")

    rf_study = optuna.create_study(direction="maximize")

    rf_study.optimize(
        rf_objective,
        n_trials=10
    )

    print("\nBest RF Params:")
    print(rf_study.best_params)

    print("\nBest RF F1:")
    print(rf_study.best_value)

    joblib.dump(
        rf_study.best_params,
        "models/best_rf_params.pkl"
    )

    # XGBoost tuning
    print("\nOptimizing XGBoost...\n")

    xgb_study = optuna.create_study(direction="maximize")

    xgb_study.optimize(
        xgb_objective,
        n_trials=10
    )

    print("\nBest XGB Params:")
    print(xgb_study.best_params)

    print("\nBest XGB F1:")
    print(xgb_study.best_value)

    joblib.dump(
        xgb_study.best_params,
        "models/best_xgb_params.pkl"
    )

    print("\nHyperparameter Tuning Complete")

except Exception as e:

    print(f"\nError in hyperparameter_tuning.py: {e}")
    traceback.print_exc()
    sys.exit(1)
