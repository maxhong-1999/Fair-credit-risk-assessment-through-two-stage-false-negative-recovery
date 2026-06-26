from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


ID_COL = "KCB_DEID1_ENCRYPT"
ACCOUNT_COL = "KCB_DEID2_ENCRYPT"
CUSTOMER_ID_COL = "KCB_DEID_ENCRYPT"
MONTH_COL = "BS_YR_MON"
TARGET_COL = "DLQ_ANY_YN"
POLICY_LOAN_COL = "LN_AMT_YN"

POLICY_LOAN_BALANCE_COLS = ["LN_AMT_SUN", "LN_AMT_NHP", "LN_AMT_ETC"]
DELINQUENCY_COLS = ["FST_DLQ_RCKN_DT", "DLQ_AMT"]

REFERENCE_PAIRS = [(201512, 201612), (201612, 201712), (201712, 201812)]
REFERENCE_PAIR_TO_PERIOD = {
    (201512, 201612): 1516,
    (201612, 201712): 1617,
    (201712, 201812): 1718,
}

SENSITIVE_OR_DESCRIPTIVE_COLS = [
    "GENDER",
    "AGE_CD",
    "JOB_CD",
    "HOM_CD",
    "COM_CD",
    "HOM_MOVE_YN",
    "COM_MOVE_YN",
    "JOB_MOVE_YN",
    "YR_INCOM_MST_AMT",
    "INCOM_AVG",
    "INCOM_LAST",
    "INCOM_SLOPE",
    "LST_SCORE",
    "SCORE_AVG",
    "SCORE_LAST",
    "SCORE_SLOPE",
    "PERIOD",
]

FINAL_G_FEATURES = [
    "BIS_AREA_FLAG_01",
    "BIS_AREA_FLAG_03",
    "BIS_AREA_FLAG_05",
    "BIS_AREA_LAST",
    "BIS_AREA_MAX_RATIO",
    "FND_PURP_FLAG_01",
    "FND_PURP_MAX_RATIO",
    "LN_BAL_MEAN",
    "LN_BAL_SLOPE",
    "LN_CONT_LN_LST_RATIO",
    "LN_MOST_RATIO",
    "LN_SUN_AVG",
    "LN_TERM_LNG_RATIO",
    "LN_TERM_SRT_RATIO",
    "MRTY_BS_COUNT",
    "OPN_BS_COUNT",
    "TX_TP_FLAG_01",
    "TX_TP_FLAG_03",
    "TX_TP_FLAG_04",
    "TX_TP_LAST",
]

FINAL_P_FEATURES = [
    "BIS_AREA_FLAG_03",
    "BIS_AREA_FLAG_05",
    "BIS_AREA_LAST",
    "LN_BAL_SLOPE",
    "LN_NHP_AVG",
    "LN_SUN_SLOPE",
]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_columns(df: pd.DataFrame, columns: Iterable[str], frame_name: str = "dataframe") -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{frame_name} is missing required columns: {missing}")


def ensure_parent(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser(), **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent(path)
    df.to_csv(Path(path).expanduser(), index=False)


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def policy_loan_exposure(df: pd.DataFrame) -> pd.Series:
    require_columns(df, POLICY_LOAN_BALANCE_COLS)
    return df[POLICY_LOAN_BALANCE_COLS].fillna(0).gt(0).any(axis=1)


def account_delinquency_indicator(df: pd.DataFrame) -> pd.Series:
    require_columns(df, DELINQUENCY_COLS)
    return df["FST_DLQ_RCKN_DT"].notna() | df["DLQ_AMT"].notna()


def pair_to_period(pair: tuple[int, int]) -> int:
    return REFERENCE_PAIR_TO_PERIOD[pair]


def infer_period_code(months: Iterable[int]) -> int | None:
    month_set = set(int(m) for m in months)
    for pair, code in REFERENCE_PAIR_TO_PERIOD.items():
        if set(pair).issubset(month_set):
            return code
    return None


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def safe_feature_columns(df: pd.DataFrame, feature_cols: Iterable[str]) -> list[str]:
    return [col for col in feature_cols if col in df.columns]


def read_model_config(config: Mapping | None = None) -> Mapping:
    return {} if config is None else config
