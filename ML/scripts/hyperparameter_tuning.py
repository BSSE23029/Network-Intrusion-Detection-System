import os
import sys
import traceback
import numpy as np
import optuna
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split
)

from xgboost import XGBClassifier


try:

    # Create models folder

    os.makedirs("models", exist_ok=True)

    # Load training data

    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")

    # Take 30% sample for faster tuning

    X_sample, X_unused, y_sample, y_unused = train_test_split(
        X_train,
        y_train,
        train_size=0.3,
        random_state=42,
        stratify=y_train
    )

    print(
        f"Tuning on {len(X_sample)} samples "
        f"(30% of {len(X_train)} training rows)\n"
    )

    # Create Stratified K-Fold

    skf = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    # Random Forest objective function

    def rf_objective(trial):

        # Optuna will try different values here
        n_estimators = trial.suggest_int(
            "n_estimators",
            100,
            300
        )

        max_depth = trial.suggest_int(
            "max_depth",
            5,
            30
        )

        min_samples_split = trial.suggest_int(
            "min_samples_split",
            2,
            10
        )

        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf",
            1,
            5
        )

        # Put all parameters in dictionary
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced"
        }

        # Create Random Forest model
        model = RandomForestClassifier(**params)

        # Test model using cross validation
        scores = cross_val_score(
            model,
            X_sample,
            y_sample,
            cv=skf,
            scoring="f1",
            n_jobs=-1
        )

        # Calculate mean and std F1 score
        mean_score = scores.mean()
        std_score = scores.std()

        print(
            f"  RF trial F1: {mean_score:.4f} "
            f"(+/- {std_score:.4f})"
        )

        return mean_score

    # XGBoost objective function

    def xgb_objective(trial):

        # Optuna will try different values here
        n_estimators = trial.suggest_int(
            "n_estimators",
            100,
            300
        )

        max_depth = trial.suggest_int(
            "max_depth",
            3,
            10
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            0.01,
            0.3
        )

        subsample = trial.suggest_float(
            "subsample",
            0.5,
            1.0
        )

        colsample_bytree = trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0
        )

        # Put all parameters in dictionary
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "eval_metric": "logloss",
            "random_state": 42
        }

        # Create XGBoost model
        model = XGBClassifier(**params)

        # Test model using cross validation
        scores = cross_val_score(
            model,
            X_sample,
            y_sample,
            cv=skf,
            scoring="f1",
            n_jobs=-1
        )

        # Calculate mean and std F1 score
        mean_score = scores.mean()
        std_score = scores.std()

        print(
            f"  XGB trial F1: {mean_score:.4f} "
            f"(+/- {std_score:.4f})"
        )

        return mean_score

    # Random Forest tuning

    print("\nOptimizing Random Forest...\n")

    rf_study = optuna.create_study(
        direction="maximize"
    )

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

    xgb_study = optuna.create_study(
        direction="maximize"
    )

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