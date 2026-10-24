"""Read the Home Credit source tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
]


def check_raw_directory(raw_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    missing = [name for name in REQUIRED_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Home Credit CSV files in {raw_dir}: {', '.join(missing)}"
        )


def read_application(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)
