from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

import numpy as np
import pandas as pd

from common import (
    ACCOUNT_COL,
    ID_COL,
    MONTH_COL,
    POLICY_LOAN_COL,
    TARGET_COL,
    account_delinquency_indicator,
    as_numeric,
    load_config,
    policy_loan_exposure,
    read_csv,
    require_columns,
    write_csv,
)


BASE_PERSON_COLS = ["GENDER", "AGE_CD", "JOB_CD", "HOM_CD", "COM_CD"]


def ym_to_months(ym: float | int) -> float:
    if pd.isna(ym):
        return np.nan
    ym = int(ym)
    return (ym // 100) * 12 + (ym % 100)


def month_diff(start_ym: float | int, end_ym: float | int) -> float:
    if pd.isna(start_ym) or pd.isna(end_ym):
        return np.nan
    return ym_to_months(end_ym) - ym_to_months(start_ym)


def first_last_slope(group: pd.DataFrame, col: str, agg: str = "mean") -> tuple[float, float, float]:
    monthly = group.groupby(MONTH_COL)[col]
    if agg == "sum":
        series = monthly.sum(min_count=1).sort_index()
    else:
        series = monthly.mean().sort_index()
    series = as_numeric(series).dropna()
    if series.empty:
        return np.nan, np.nan, np.nan
    avg = float(series.mean())
    last = float(series.iloc[-1])
    slope = float(series.iloc[-1] - series.iloc[0]) if len(series) >= 2 else np.nan
    return avg, last, slope


def mode_or_nan(series: pd.Series):
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def most_recent_value(group: pd.DataFrame, col: str):
    ordered = group.sort_values(MONTH_COL)
    values = ordered[col].dropna()
    return values.iloc[-1] if len(values) else np.nan


def categorical_summary(group: pd.DataFrame, col: str, weight_col: str, prefix: str, flags: list[int]) -> dict:
    values = group[col].dropna()
    out = {
        f"{prefix}_MOST": mode_or_nan(values),
        f"{prefix}_MOST_RATIO": values.value_counts(normalize=True).iloc[0] if len(values) else np.nan,
        f"{prefix}_NUM": values.nunique(),
        f"{prefix}_LAST": most_recent_value(group, col),
    }
    for value in flags:
        out[f"{prefix}_FLAG_{value:02d}"] = int((values == value).any())

    weighted = group[[col, weight_col]].dropna()
    if weighted.empty:
        out[f"{prefix}_MAX"] = np.nan
        out[f"{prefix}_MAX_RATIO"] = np.nan
    else:
        sums = weighted.groupby(col)[weight_col].sum()
        total = sums.sum()
        out[f"{prefix}_MAX"] = sums.idxmax()
        out[f"{prefix}_MAX_RATIO"] = float(sums.max() / total) if total > 0 else np.nan
    return out


def account_open_maturity_counts(group: pd.DataFrame) -> tuple[int, int]:
    recent_year = group[MONTH_COL].max() // 100
    open_year = as_numeric(group["OPN_BS_YR_MON"]) // 100
    maturity_year = as_numeric(group["MRTY_BS_YR_MON"]) // 100
    return int((open_year == recent_year).sum()), int((maturity_year == recent_year).sum())


def balance_ratios(group: pd.DataFrame) -> dict:
    latest = (
        group.sort_values([ACCOUNT_COL, MONTH_COL])
        .groupby(ACCOUNT_COL)
        .tail(1)[[ACCOUNT_COL, "LN_BAL_TOT", "LN_CONT_TOT"]]
    )
    balance_sum = as_numeric(latest["LN_BAL_TOT"]).sum()
    contract_sum = as_numeric(latest["LN_CONT_TOT"]).sum()
    account_count = len(latest)
    return {
        "LN_CONT_LN_LST_RATIO": float(balance_sum / contract_sum) if contract_sum > 0 else np.nan,
        "LN_FIN_RATIO": float((as_numeric(latest["LN_BAL_TOT"]) == 0).sum() / account_count) if account_count else np.nan,
        "LN_MOST_RATIO": float(as_numeric(latest["LN_BAL_TOT"]).max() / balance_sum) if balance_sum > 0 else 0.0,
    }


def loan_term_features(group: pd.DataFrame) -> dict:
    terms = [
        month_diff(row.OPN_BS_YR_MON, row.MRTY_BS_YR_MON)
        for row in group[["OPN_BS_YR_MON", "MRTY_BS_YR_MON"]].itertuples(index=False)
    ]
    terms = pd.Series(terms, dtype="float64").dropna()
    if terms.empty:
        return {"LN_TERM_MAX": np.nan, "LN_TERM_SRT_RATIO": np.nan, "LN_TERM_LNG_RATIO": np.nan}
    return {
        "LN_TERM_MAX": float(terms.max()),
        "LN_TERM_SRT_RATIO": float((terms <= 13).mean()),
        "LN_TERM_LNG_RATIO": float((terms >= 62).mean()),
    }


def build_borrower_features(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, [ID_COL, ACCOUNT_COL, MONTH_COL, "LN_CONT_TOT", "LN_BAL_TOT"])
    rows = []

    for borrower_id, group in df.groupby(ID_COL, sort=False):
        group = group.sort_values(MONTH_COL)
        first = group.iloc[0]
        row = {ID_COL: borrower_id}
        for col in BASE_PERSON_COLS:
            row[col] = first.get(col, np.nan)

        row["JOB_MOVE_YN"] = int(group["JOB_CD"].dropna().nunique() > 1) if "JOB_CD" in group else np.nan
        row["HOM_MOVE_YN"] = int(group["HOM_CD"].dropna().nunique() > 1) if "HOM_CD" in group else np.nan
        row["COM_MOVE_YN"] = int(group["COM_CD"].dropna().nunique() > 1) if "COM_CD" in group else np.nan
        row["PERIOD"] = int(group["PERIOD"].iloc[0]) if "PERIOD" in group else np.nan

        for src, prefix, agg in [
            ("YR_INCOM_MST_AMT", "INCOM", "mean"),
            ("LST_SCORE", "SCORE", "mean"),
            ("LN_AMT_SUN", "LN_SUN", "sum"),
            ("LN_AMT_NHP", "LN_NHP", "sum"),
            ("LN_AMT_ETC", "LN_ETC", "sum"),
        ]:
            if src in group:
                avg, last, slope = first_last_slope(group, src, agg=agg)
                row[f"{prefix}_AVG"] = avg
                row[f"{prefix}_LAST"] = last
                row[f"{prefix}_SLOPE"] = slope

        if all(c in group for c in ["LN_AMT_SUN", "LN_AMT_NHP", "LN_AMT_ETC"]):
            latest_month = group[MONTH_COL].max()
            latest_rows = group[group[MONTH_COL] == latest_month]
            row[POLICY_LOAN_COL] = int(policy_loan_exposure(latest_rows).any())

        row.update(categorical_summary(group, "BIS_AREA", "LN_CONT_TOT", "BIS_AREA", [1, 2, 3, 4, 5, 6]))
        row.update(categorical_summary(group, "LN_GOODS_CD", "LN_CONT_TOT", "LN_GOODS", [1, 2, 3, 4]))
        row.update(categorical_summary(group, "TX_TP_CD", "LN_CONT_TOT", "TX_TP", [1, 2, 3, 4]))

        fnd = categorical_summary(group, "FND_PURP_CD", "LN_CONT_TOT", "FND_PURP", [1, 2, 3, 4])
        row.update({k: v for k, v in fnd.items() if k not in {"FND_PURP_MOST", "FND_PURP_MOST_RATIO", "FND_PURP_LAST"}})

        row["OPN_BS_COUNT"], row["MRTY_BS_COUNT"] = account_open_maturity_counts(group)
        row.update(loan_term_features(group))
        row.update(balance_ratios(group))

        first_balance = group[group[MONTH_COL] == group[MONTH_COL].min()]["LN_BAL_TOT"].sum()
        last_balance = group[group[MONTH_COL] == group[MONTH_COL].max()]["LN_BAL_TOT"].sum()
        row["LN_BAL_MEAN"] = float(as_numeric(group["LN_BAL_TOT"]).mean())
        row["LN_BAL_SLOPE"] = float(last_balance - first_balance)
        row["LN_COUNT"] = int(group[ACCOUNT_COL].nunique())
        row["HOM_IN_SMA_YN"] = int(str(row.get("HOM_CD", ""))[:2] in {"11", "41", "28"})

        policy_repay_sum = group[["LN_AMT_SUN", "LN_AMT_NHP", "LN_AMT_ETC"]].fillna(0).sum(axis=1).sum()
        recent_income = row.get("INCOM_LAST", np.nan)
        row["DSR"] = float(policy_repay_sum / recent_income) if pd.notna(recent_income) and recent_income > 0 else np.nan

        row[TARGET_COL] = int(account_delinquency_indicator(group).any())
        rows.append(row)

    return pd.DataFrame(rows)


def run(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    cohort = read_csv(config["cohort_account_csv"])
    features = build_borrower_features(cohort)
    write_csv(features, config["borrower_features_csv"])
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
