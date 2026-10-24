"""Leakage-safe LightGBM training for the project."""
from __future__ import annotations

from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


LEARNING_RATE = 0.03
MAX_TREES = 2_000
EARLY_STOPPING_ROUNDS = 100


def make_model(seed: int, number_of_trees: int = MAX_TREES) -> LGBMClassifier:
    """Use LightGBM defaults except for the documented learning rate and seed."""
    return LGBMClassifier(
        objective="binary",
        learning_rate=LEARNING_RATE,
        n_estimators=number_of_trees,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _prepare_categories(
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Learn category levels from training rows and align validation to them."""
    categorical = training.select_dtypes(include=["object", "string"]).columns.tolist()
    fit = training.copy()
    valid = validation.copy()

    for column in categorical:
        fit_values = fit[column].astype("string").fillna("__MISSING__").astype(str)
        valid_values = valid[column].astype("string").fillna("__MISSING__").astype(str)
        known = set(fit_values.unique())
        categories = ["__MISSING__", "__UNKNOWN__"]
        categories.extend(sorted(known - set(categories)))
        valid_values = valid_values.where(valid_values.isin(known), "__UNKNOWN__")
        fit[column] = pd.Categorical(fit_values, categories=categories)
        valid[column] = pd.Categorical(valid_values, categories=categories)

    return fit, valid, categorical


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def cross_validate(
    frame: pd.DataFrame,
    labels: np.ndarray,
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[dict, list[int]]:
    """Evaluate a branch with one fixed set of folds and fold-only categories."""
    feature_names = [
        column for column in frame.columns if column not in {"SK_ID_CURR", "TARGET"}
    ]
    predictions = np.full(len(frame), np.nan, dtype=np.float32)
    fold_results = []
    best_iterations = []

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fit_raw = frame.iloc[fit_rows][feature_names]
        valid_raw = frame.iloc[valid_rows][feature_names]
        fit, valid, categorical = _prepare_categories(fit_raw, valid_raw)

        model = make_model(seed + fold_number)
        model.fit(
            fit,
            labels[fit_rows],
            eval_set=[(valid, labels[valid_rows])],
            eval_metric="auc",
            categorical_feature=categorical,
            callbacks=[
                lgb.early_stopping(
                    EARLY_STOPPING_ROUNDS,
                    first_metric_only=True,
                    verbose=False,
                )
            ],
        )
        probability = model.predict_proba(
            valid, num_iteration=model.best_iteration_
        )[:, 1]
        predictions[valid_rows] = probability.astype("float32")
        result = _metrics(labels[valid_rows], probability)
        result["fold"] = fold_number
        result["best_iteration"] = int(model.best_iteration_ or MAX_TREES)
        fold_results.append(result)
        best_iterations.append(result["best_iteration"])

    if np.isnan(predictions).any():
        raise RuntimeError("Some applicants did not receive an out-of-fold prediction.")

    pooled = _metrics(labels, predictions)
    summary = {
        "fold_mean_roc_auc": float(np.mean([row["roc_auc"] for row in fold_results])),
        "fold_sd_roc_auc": float(np.std(
            [row["roc_auc"] for row in fold_results], ddof=1
        )),
        "pooled_roc_auc": pooled["roc_auc"],
        "fold_mean_average_precision": float(
            np.mean([row["average_precision"] for row in fold_results])
        ),
        "fold_sd_average_precision": float(np.std(
            [row["average_precision"] for row in fold_results], ddof=1
        )),
        "pooled_average_precision": pooled["average_precision"],
        "median_best_iteration": int(np.median(best_iterations)),
        "folds": fold_results,
    }
    return summary, best_iterations


def _fit_categories(
    training: pd.DataFrame,
    other: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, list[str]]]:
    """Fit category levels on training data and apply the same levels elsewhere."""
    categorical = training.select_dtypes(include=["object", "string"]).columns.tolist()
    fit = training.copy()
    aligned = other.copy()
    schema = {}

    for column in categorical:
        values = fit[column].astype("string").fillna("__MISSING__").astype(str)
        known = set(values.unique())
        categories = ["__MISSING__", "__UNKNOWN__"]
        categories.extend(sorted(known - set(categories)))
        other_values = aligned[column].astype("string").fillna("__MISSING__").astype(str)
        other_values = other_values.where(other_values.isin(known), "__UNKNOWN__")
        fit[column] = pd.Categorical(values, categories=categories)
        aligned[column] = pd.Categorical(other_values, categories=categories)
        schema[column] = categories

    return fit, aligned, categorical, schema


def fit_final_model(
    training_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    number_of_trees: int,
    seed: int,
) -> tuple[LGBMClassifier, pd.DataFrame, dict]:
    feature_names = [
        column for column in training_frame.columns
        if column not in {"SK_ID_CURR", "TARGET"}
    ]
    training_features, test_features, categorical, schema = _fit_categories(
        training_frame[feature_names], test_frame[feature_names]
    )
    labels = training_frame["TARGET"].to_numpy(dtype="int8")
    model = make_model(seed, number_of_trees)
    model.fit(
        training_features,
        labels,
        categorical_feature=categorical,
    )

    probabilities = model.predict_proba(test_features)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("The final model produced a missing or non-finite score.")
    submission = pd.DataFrame({
        "SK_ID_CURR": test_frame["SK_ID_CURR"].to_numpy(),
        "TARGET": probabilities,
    })
    importance = pd.DataFrame({
        "feature": feature_names,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).head(10).reset_index(drop=True)

    saved_model = {
        "model": model,
        "feature_columns": feature_names,
        "categorical_columns": categorical,
        "category_schema": schema,
        "id_column": "SK_ID_CURR",
        "target_column": "TARGET",
    }
    return model, submission, {"importance": importance, "saved_model": saved_model}

