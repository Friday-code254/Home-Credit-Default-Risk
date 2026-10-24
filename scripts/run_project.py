#!/usr/bin/env python3
"""Build Home Credit features, compare raw and cleaned models, and write one report."""
from __future__ import annotations

import ctypes
import gc
import logging
import shutil
import sys
import time
from ctypes import wintypes
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from home_credit.data_io import check_raw_directory, read_application
from home_credit.features import (
    TABLE_CONFIG,
    add_employment_sentinel_feature,
    aggregate_bureau_balance,
    aggregate_numeric_table,
    merge_history_features,
)
from home_credit.modeling import (
    EARLY_STOPPING_ROUNDS,
    LEARNING_RATE,
    MAX_TREES,
    cross_validate,
    fit_final_model,
)

SEED = 42
FOLDS = 5
CHUNK_SIZE = 300_000
LOGGER = logging.getLogger("home_credit")


def peak_working_set_gib() -> float:
    if sys.platform == "win32":
        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_info = psapi.GetProcessMemoryInfo
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(MemoryCounters),
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not get_info(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        peak_bytes = counters.PeakWorkingSetSize
    else:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return peak_bytes / (1024 ** 3)


def build_aggregates(raw_dir: Path, cache_dir: Path) -> tuple[dict, dict]:
    """Create temporary aggregates and count the previous-application placeholders."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    cleaning_counts = {}
    for config in TABLE_CONFIG:
        filename = config["file"]
        LOGGER.info("Aggregating %s", filename)
        if filename == "previous_application.csv":
            raw_features, audit = aggregate_numeric_table(
                raw_dir, config, clean_sentinels=False, chunksize=CHUNK_SIZE
            )
            clean_features, _ = aggregate_numeric_table(
                raw_dir, config, clean_sentinels=True, chunksize=CHUNK_SIZE
            )
            raw_features.to_pickle(cache_dir / "previous_application_raw.pkl")
            clean_features.to_pickle(cache_dir / "previous_application_cleaned.pkl")

            remaining = {}
            for column in config["sentinels"]:
                feature = f"PREV_APP_{column}__MAX"
                remaining[column] = (
                    int(clean_features[feature].eq(365243).sum())
                    if feature in clean_features else 0
                )
            audit["cleaned_max_still_sentinel"] = remaining
            cleaning_counts["previous_application"] = audit
            del raw_features, clean_features
        else:
            features, _ = aggregate_numeric_table(
                raw_dir, config, clean_sentinels=False, chunksize=CHUNK_SIZE
            )
            features.to_pickle(cache_dir / f"{Path(filename).stem}.pkl")
            del features
        gc.collect()

    LOGGER.info("Aggregating bureau-balance history")
    bureau_balance, linkage = aggregate_bureau_balance(
        raw_dir, chunksize=CHUNK_SIZE
    )
    bureau_balance.to_pickle(cache_dir / "bureau_balance.pkl")
    del bureau_balance
    gc.collect()
    return cleaning_counts, linkage


def build_application_frame(
    raw_dir: Path,
    cache_dir: Path,
    label: str,
    clean_sentinels: bool,
) -> tuple[pd.DataFrame, int]:
    frame = read_application(raw_dir / f"application_{label}.csv")
    frame, sentinel_count = add_employment_sentinel_feature(
        frame, clean_sentinel=clean_sentinels
    )
    previous_variant = "cleaned" if clean_sentinels else "raw"
    frame = merge_history_features(frame, cache_dir, previous_variant)
    return frame, sentinel_count


def make_report(
    training_rows: int,
    test_rows: int,
    positive_rate: float,
    employment_counts: dict[str, int],
    cleaning_counts: dict,
    linkage: dict,
    raw_metrics: dict,
    cleaned_metrics: dict,
    importance: pd.DataFrame,
    chosen_trees: int,
    python_version: str,
    runtime_minutes: float,
    peak_memory_gib: float,
) -> str:
    previous = cleaning_counts["previous_application"]
    total_sentinels = sum(previous["sentinel_counts"].values())
    field_rows = [
        f"| {column} | {count:,} | "
        f"{previous['cleaned_max_still_sentinel'][column]:,} |"
        for column, count in previous["sentinel_counts"].items()
    ]
    fields_table = "\n".join(field_rows)

    metric_rows = []
    for label, key in [("ROC-AUC", "roc_auc"), ("Average precision", "average_precision")]:
        metric_rows.append(
            f"| {label} | "
            f"{raw_metrics[f'fold_mean_{key}']:.6f} ± {raw_metrics[f'fold_sd_{key}']:.6f} | "
            f"{cleaned_metrics[f'fold_mean_{key}']:.6f} ± "
            f"{cleaned_metrics[f'fold_sd_{key}']:.6f} |"
        )
    metrics_table = "\n".join(metric_rows)

    fold_rows = []
    cleaned_wins = 0
    for raw_fold, cleaned_fold in zip(raw_metrics["folds"], cleaned_metrics["folds"]):
        difference = cleaned_fold["roc_auc"] - raw_fold["roc_auc"]
        cleaned_wins += int(difference > 0)
        fold_rows.append(
            f"| {raw_fold['fold']} | {raw_fold['roc_auc']:.6f} | "
            f"{cleaned_fold['roc_auc']:.6f} | {difference:+.6f} |"
        )
    fold_table = "\n".join(fold_rows)

    importance_rows = [
        f"| {row.feature} | {row.gain:,.1f} |"
        for row in importance.itertuples(index=False)
    ]
    importance_table = "\n".join(importance_rows)
    unmatched = linkage["unmatched_rows"]
    total_balance = linkage["source_rows"]
    unmatched_rate = unmatched / total_balance if total_balance else 0.0

    return f"""# Home Credit Default Risk: Project Report

## Cleaning results

This project focuses on three data problems with a clear treatment.

### Encoded missing dates

Home Credit uses 365243 as a placeholder in DAYS_EMPLOYED and five previous-application day-offset fields. It is not a real elapsed-day value. Both branches keep the same sentinel indicators and applicant-level counts. They also share the same history tables and bureau-balance linkage rules. The only difference is that the cleaned branch changes 365243 to missing before aggregation; the raw branch keeps it numeric.

This follows a known Home Credit data convention used in participant feature engineering ([example](https://github.com/pklauke/Kaggle-HomeCreditDefaultRisk/blob/master/feature_engineering.py)). The source CSV files are not modified.

| Application field | Training rows with placeholder | Test rows with placeholder |
|---|---:|---:|
| DAYS_EMPLOYED | {employment_counts["train"]:,} | {employment_counts["test"]:,} |

Previous-application placeholders found: **{total_sentinels:,}** across **{previous["rows_with_any_sentinel"]:,}** rows out of {previous["source_rows"]:,}.

| Previous-application field | Placeholder values found | Cleaned aggregate MAX still equal to 365243 |
|---|---:|---:|
{fields_table}

### Unmatched bureau-balance rows

Bureau-balance rows are joined through the bureau loan ID. Rows without a matching bureau record cannot be assigned to an applicant, so they are excluded from applicant features.

- Rows read: {total_balance:,}
- Unmatched rows excluded: {unmatched:,} ({unmatched_rate:.2%})

The count is reported here; no separate quarantine file is written.

## Raw versus cleaned model

Both branches use the same {FOLDS} stratified folds (split seed {SEED}). Category levels are learned from each training fold only. Early stopping uses each fold's validation data, and that same fold produces the reported score, so scores are slightly optimistic. The raw and cleaned runs are affected equally. The ± values below are standard deviations across folds.

| Metric | Raw (mean ± standard deviation) | Cleaned (mean ± standard deviation) |
|---|---:|---:|
{metrics_table}

| Fold | Raw ROC-AUC | Cleaned ROC-AUC | Difference |
|---:|---:|---:|---:|
{fold_table}

Cleaned scored higher on {cleaned_wins} of {FOLDS} folds. This is a descriptive comparison from one seed and five folds, not evidence of a reliable performance gain. The main result is the corrected treatment of known placeholder values.

## Model setup

LightGBM binary classifier; {FOLDS} stratified folds with split seed {SEED}; fold model seeds {SEED + 1}–{SEED + FOLDS}; learning rate {LEARNING_RATE}; maximum {MAX_TREES:,} trees with {EARLY_STOPPING_ROUNDS}-round early stopping. The final cleaned model uses {chosen_trees:,} trees, the median best iteration across cleaned folds. LightGBM tree parameters otherwise use library defaults; all CPU threads are enabled and verbosity is suppressed.

- Python used for this run: {python_version}
- Runtime: {runtime_minutes:.1f} minutes
- Peak process working set: {peak_memory_gib:.1f} GiB
- Training applicants: {training_rows:,}; positive target rate: {positive_rate:.2%}
- Test applicants scored: {test_rows:,}

## Final model feature importance

Gain importance shows how much each feature reduced loss in this fitted model. It describes this model; it does not show that a feature causes default.

| Feature | Gain |
|---|---:|
{importance_table}

## Data terms and limitations

The shareable project copy excludes the Kaggle CSVs. Obtain the source files from Kaggle and follow the competition rules. Cleaning counts and excluded-row counts apply to the source files used for this run. Metrics use one seed and five folds, with early stopping on each validation fold.

## Reproduce the run

1. Download the Home Credit Default Risk CSV files from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place them in data/raw/.
2. Install packages with: pip install -r requirements.txt
3. Run: python scripts/run_project.py

The script rebuilds temporary aggregates on each run and removes them afterward. The model is saved in artifacts/, predictions in data/processed/, and this report in reports/project_report.md.
"""


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    raw_dir = ROOT / "data" / "raw"
    cache_dir = ROOT / "data" / "interim" / "aggregates"
    report_dir = ROOT / "reports"
    artifact_dir = ROOT / "artifacts"
    processed_dir = ROOT / "data" / "processed"

    check_raw_directory(raw_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    try:
        cleaning_counts, linkage = build_aggregates(raw_dir, cache_dir)

        raw_train, _ = build_application_frame(
            raw_dir, cache_dir, "train", clean_sentinels=False
        )
        labels = raw_train["TARGET"].to_numpy(dtype="int8")
        ids = raw_train["SK_ID_CURR"].to_numpy()
        fold_plan = list(
            StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
            .split(np.zeros(len(labels)), labels)
        )

        LOGGER.info("Training raw-data comparison")
        raw_metrics, _ = cross_validate(raw_train, labels, fold_plan, SEED)
        del raw_train
        gc.collect()

        cleaned_train, cleaned_employment_train = build_application_frame(
            raw_dir, cache_dir, "train", clean_sentinels=True
        )
        cleaned_test, cleaned_employment_test = build_application_frame(
            raw_dir, cache_dir, "test", clean_sentinels=True
        )
        if not np.array_equal(ids, cleaned_train["SK_ID_CURR"].to_numpy()):
            raise ValueError("Raw and cleaned training rows are not in the same order.")
        if not np.array_equal(labels, cleaned_train["TARGET"].to_numpy(dtype="int8")):
            raise ValueError("The cleaned branch changed training labels.")

        LOGGER.info("Training cleaned-data comparison")
        cleaned_metrics, best_iterations = cross_validate(
            cleaned_train, labels, fold_plan, SEED
        )
        chosen_trees = int(np.median(best_iterations))

        LOGGER.info("Fitting the final cleaned model")
        _, submission, final_details = fit_final_model(
            cleaned_train,
            cleaned_test,
            number_of_trees=chosen_trees,
            seed=SEED,
        )
        submission.to_csv(processed_dir / "submission.csv", index=False)
        joblib.dump(
            final_details["saved_model"],
            artifact_dir / "home_credit_default_model.joblib",
            compress=3,
        )

        report = make_report(
            training_rows=len(cleaned_train),
            test_rows=len(cleaned_test),
            positive_rate=float(labels.mean()),
            employment_counts={
                "train": cleaned_employment_train,
                "test": cleaned_employment_test,
            },
            cleaning_counts=cleaning_counts,
            linkage=linkage,
            raw_metrics=raw_metrics,
            cleaned_metrics=cleaned_metrics,
            importance=final_details["importance"],
            chosen_trees=chosen_trees,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            runtime_minutes=(time.perf_counter() - started_at) / 60,
            peak_memory_gib=peak_working_set_gib(),
        )
        (report_dir / "project_report.md").write_text(report, encoding="utf-8")

        LOGGER.info(
            "Finished: %s test predictions; report at %s",
            len(submission),
            report_dir / "project_report.md",
        )
        del submission, final_details, cleaned_train, cleaned_test
        gc.collect()
    finally:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
