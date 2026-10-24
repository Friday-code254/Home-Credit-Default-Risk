"""Feature aggregation for the Home Credit history tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TABLE_CONFIG = [
    {
        "file": "bureau.csv", "key": "SK_ID_CURR", "prefix": "BUREAU_",
        "numeric": [
            "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT",
            "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE",
            "DAYS_CREDIT_UPDATE", "AMT_ANNUITY",
        ],
        "categorical": {
            "CREDIT_ACTIVE": ["ACTIVE", "CLOSED", "SOLD", "BAD DEBT"],
            "CREDIT_TYPE": ["CONSUMER CREDIT", "CREDIT CARD", "CAR LOAN", "MORTGAGE", "MICROLOAN"],
        },
    },
    {
        "file": "previous_application.csv", "key": "SK_ID_CURR", "prefix": "PREV_APP_",
        "numeric": [
            "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE",
            "HOUR_APPR_PROCESS_START", "NFLAG_LAST_APPL_IN_DAY", "RATE_DOWN_PAYMENT",
            "RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED", "DAYS_DECISION",
            "SELLERPLACE_AREA", "CNT_PAYMENT", "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
            "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION",
        ],
        "categorical": {
            "NAME_CONTRACT_STATUS": ["APPROVED", "REFUSED", "CANCELED", "UNUSED OFFER"],
            "NAME_CONTRACT_TYPE": ["CASH LOANS", "CONSUMER LOANS", "REVOLVING LOANS", "XNA"],
        },
        "sentinels": {
            "DAYS_FIRST_DRAWING": 365243,
            "DAYS_FIRST_DUE": 365243,
            "DAYS_LAST_DUE_1ST_VERSION": 365243,
            "DAYS_LAST_DUE": 365243,
            "DAYS_TERMINATION": 365243,
        },
    },
    {
        "file": "POS_CASH_balance.csv", "key": "SK_ID_CURR", "prefix": "POS_",
        "numeric": ["MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"],
        "categorical": {
            "NAME_CONTRACT_STATUS": [
                "ACTIVE", "COMPLETED", "SIGNED", "RETURNED TO THE STORE", "DEMAND", "XNA",
            ],
        },
    },
    {
        "file": "credit_card_balance.csv", "key": "SK_ID_CURR", "prefix": "CC_",
        "numeric": [
            "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT",
            "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT", "AMT_DRAWINGS_POS_CURRENT",
            "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT",
            "AMT_RECEIVABLE_PRINCIPAL", "AMT_RECIVABLE", "AMT_TOTAL_RECEIVABLE",
            "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT", "CNT_DRAWINGS_OTHER_CURRENT",
            "CNT_DRAWINGS_POS_CURRENT", "CNT_INSTALMENT_MATURE_CUM", "SK_DPD", "SK_DPD_DEF",
        ],
        "categorical": {
            "NAME_CONTRACT_STATUS": [
                "ACTIVE", "COMPLETED", "SIGNED", "DEMAND", "SENT PROPOSAL", "REFUSED", "XNA",
            ],
        },
    },
    {
        "file": "installments_payments.csv", "key": "SK_ID_CURR", "prefix": "INST_",
        "numeric": [
            "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
        ],
        "categorical": {},
    },
]

def _combine_partials(current: pd.DataFrame | None, part: pd.DataFrame, key: str) -> pd.DataFrame:
    if part.empty:
        return current if current is not None else part
    if current is None or current.empty:
        return part

    columns = sorted((set(current.columns) | set(part.columns)) - {key})
    joined = pd.concat([current.set_index(key), part.set_index(key)], axis=0, sort=False)
    operations = {
        column: ("min" if column.endswith("__MIN")
                 else "max" if column.endswith("__MAX")
                 else "sum")
        for column in columns
    }
    return joined.groupby(level=0, sort=False).agg(operations).reset_index()


def _aggregate_chunk(
    chunk: pd.DataFrame,
    config: dict,
    clean_sentinels: bool,
) -> pd.DataFrame:
    key = config["key"]
    prefix = config["prefix"]
    chunk = chunk.loc[chunk[key].notna()].copy()
    if chunk.empty:
        return pd.DataFrame(columns=[key])

    for column, sentinel in config.get("sentinels", {}).items():
        if column not in chunk:
            continue
        values = pd.to_numeric(chunk[column], errors="raise")
        sentinel_rows = values.eq(sentinel)
        chunk[f"_SENTINEL_{column}"] = sentinel_rows.astype("int32")
        if clean_sentinels:
            values = values.mask(sentinel_rows)
        chunk[column] = values

    numeric = [column for column in config["numeric"] if column in chunk]
    for column in numeric:
        chunk[column] = pd.to_numeric(chunk[column], errors="raise")

    if numeric:
        grouped = chunk.groupby(key, sort=False)[numeric].agg(["sum", "count", "min", "max"])
        grouped.columns = [
            f"{prefix}{column}__{operation.upper()}"
            for column, operation in grouped.columns
        ]
        result = grouped
    else:
        result = chunk.groupby(key, sort=False).size().to_frame("_unused").drop(columns="_unused")

    result[f"{prefix}RECORD_COUNT"] = chunk.groupby(key, sort=False).size()
    for column in config.get("sentinels", {}):
        marker = f"_SENTINEL_{column}"
        if marker in chunk:
            result[f"{prefix}{column}__SENTINEL_COUNT"] = (
                chunk.groupby(key, sort=False)[marker].sum()
            )

    for column, values in config["categorical"].items():
        if column not in chunk:
            continue
        labels = chunk[column].astype("string")
        for value in values:
            slug = value.replace(" ", "_").replace("/", "_")
            feature = f"{prefix}{column}__CAT_{slug}"
            result[feature] = (
                labels.eq(value).fillna(False).astype("int32")
                .groupby(chunk[key], sort=False).sum()
            )
    return result.reset_index()


def aggregate_numeric_table(
    raw_dir: str | Path,
    config: dict,
    clean_sentinels: bool,
    chunksize: int = 300_000,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate one history table and count any configured placeholders."""
    path = Path(raw_dir) / config["file"]
    if not path.exists():
        raise FileNotFoundError(f"Required history table is missing: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    key = config["key"]
    if key not in header:
        raise ValueError(f"{path.name} does not contain the applicant key {key}")

    numeric = [column for column in config["numeric"] if column in header]
    categorical = [column for column in config["categorical"] if column in header]
    sentinel_columns = [column for column in config.get("sentinels", {}) if column in header]
    usecols = list(dict.fromkeys([key, *numeric, *categorical]))
    partials: pd.DataFrame | None = None
    field_counts = {column: 0 for column in sentinel_columns}
    source_rows = 0
    rows_with_sentinel = 0

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        source_rows += len(chunk)
        if sentinel_columns:
            masks = {}
            for column in sentinel_columns:
                values = pd.to_numeric(chunk[column], errors="raise")
                masks[column] = values.eq(config["sentinels"][column])
                field_counts[column] += int(masks[column].sum())
            rows_with_sentinel += int(pd.DataFrame(masks).any(axis=1).sum())

        part = _aggregate_chunk(chunk, config, clean_sentinels)
        partials = _combine_partials(partials, part, key)

    if partials is None:
        partials = pd.DataFrame(columns=[key])

    sum_columns = [
        column for column in partials.columns
        if column.startswith(config["prefix"]) and column.endswith("__SUM")
    ]
    for sum_column in sum_columns:
        root = sum_column[:-5]
        count_column = f"{root}__COUNT"
        if count_column in partials:
            count = partials[count_column].replace(0, np.nan)
            partials[f"{root}__MEAN"] = partials[sum_column] / count


    audit = {
        "table": path.name,
        "source_rows": source_rows,
        "rows_with_any_sentinel": rows_with_sentinel,
        "sentinel_counts": field_counts,
    }
    return partials, audit


def aggregate_bureau_balance(
    raw_dir: str | Path,
    chunksize: int = 300_000,
) -> tuple[pd.DataFrame, dict]:
    """Join monthly records through SK_ID_BUREAU; count and exclude unlinked rows."""
    raw_dir = Path(raw_dir)
    bureau_path = raw_dir / "bureau.csv"
    balance_path = raw_dir / "bureau_balance.csv"

    mapping_frame = pd.read_csv(bureau_path, usecols=["SK_ID_BUREAU", "SK_ID_CURR"])
    mapping_frame = mapping_frame.dropna(subset=["SK_ID_BUREAU", "SK_ID_CURR"])
    conflicting = (
        mapping_frame.groupby("SK_ID_BUREAU", sort=False)["SK_ID_CURR"].nunique().gt(1)
    )
    if conflicting.any():
        raise ValueError("A bureau ID maps to more than one applicant; the join is ambiguous.")
    mapping = (
        mapping_frame.drop_duplicates("SK_ID_BUREAU")
        .set_index("SK_ID_BUREAU")["SK_ID_CURR"]
    )

    combined: pd.DataFrame | None = None
    total_rows = 0
    unmatched_rows = 0

    for chunk in pd.read_csv(
        balance_path,
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
        chunksize=chunksize,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk["SK_ID_CURR"] = chunk["SK_ID_BUREAU"].map(mapping)
        unmatched_rows += int(chunk["SK_ID_CURR"].isna().sum())
        chunk = chunk.loc[chunk["SK_ID_CURR"].notna()].copy()
        if chunk.empty:
            continue

        chunk["MONTHS_BALANCE"] = pd.to_numeric(
            chunk["MONTHS_BALANCE"], errors="raise"
        )
        grouped = chunk.groupby("SK_ID_CURR", sort=False).agg(
            BB_RECORD_COUNT=("SK_ID_BUREAU", "size"),
            BB_MONTHS_BALANCE__MIN=("MONTHS_BALANCE", "min"),
            BB_MONTHS_BALANCE__MAX=("MONTHS_BALANCE", "max"),
        )
        status = chunk["STATUS"].astype("string")
        for value in ["0", "1", "2", "3", "4", "5", "C", "X"]:
            grouped[f"BB_STATUS_{value}_COUNT"] = (
                status.eq(value).fillna(False).astype("int32")
                .groupby(chunk["SK_ID_CURR"], sort=False).sum()
            )
        grouped["BB_DELINQUENT_STATUS_COUNT"] = (
            status.isin(["1", "2", "3", "4", "5"]).astype("int32")
            .groupby(chunk["SK_ID_CURR"], sort=False).sum()
        )
        combined = _combine_partials(combined, grouped.reset_index(), "SK_ID_CURR")

    if combined is None:
        combined = pd.DataFrame(columns=["SK_ID_CURR"])
    audit = {"source_rows": total_rows, "unmatched_rows": unmatched_rows}
    return combined, audit


def add_employment_sentinel_feature(
    frame: pd.DataFrame,
    clean_sentinel: bool,
) -> tuple[pd.DataFrame, int]:
    """Add a sentinel indicator to both branches and optionally recode the value."""
    values = pd.to_numeric(frame["DAYS_EMPLOYED"], errors="raise")
    mask = values.eq(365243)
    frame["DAYS_EMPLOYED_WAS_SENTINEL"] = mask.astype("int8")
    frame["DAYS_EMPLOYED"] = values
    if clean_sentinel:
        frame.loc[mask, "DAYS_EMPLOYED"] = np.nan
    return frame, int(mask.sum())


def merge_history_features(
    frame: pd.DataFrame,
    cache_dir: str | Path,
    previous_app_variant: str,
) -> pd.DataFrame:
    """Join the already-aggregated tables to each applicant."""
    result = frame.copy()
    cache_dir = Path(cache_dir)
    for table_config in TABLE_CONFIG:
        filename = table_config["file"]
        suffix = f"_{previous_app_variant}" if filename == "previous_application.csv" else ""
        feature_path = cache_dir / f"{Path(filename).stem}{suffix}.pkl"
        history = pd.read_pickle(feature_path)
        result = result.merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
        del history
    bureau_balance = pd.read_pickle(cache_dir / "bureau_balance.pkl")
    result = result.merge(
        bureau_balance, on="SK_ID_CURR", how="left", validate="one_to_one"
    )

    for column in result.select_dtypes(include=[np.number]).columns:
        if column not in {"SK_ID_CURR", "TARGET"}:
            result[column] = result[column].astype("float32")
    result["SK_ID_CURR"] = pd.to_numeric(result["SK_ID_CURR"], errors="raise").astype("int32")
    if "TARGET" in result:
        result["TARGET"] = pd.to_numeric(result["TARGET"], errors="raise").astype("int8")
    return result.sort_values("SK_ID_CURR", kind="mergesort").reset_index(drop=True)


