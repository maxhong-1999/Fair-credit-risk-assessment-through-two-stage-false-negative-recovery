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
    REFERENCE_PAIR_TO_PERIOD,
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
    ordered = group.sort_values(MONTH_COL, kind="mergesort")
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


FINAL_ACCOUNT_PRODUCT_CODES = {1, 4}
DUPLICATE_ACCOUNT_CONTENT_COLS = [
    MONTH_COL,
    "BIS_AREA",
    "LN_BAL_TOT",
    "LN_GOODS_CD",
    "LN_CONT_TOT",
    "TX_TP_CD",
    "FND_PURP_CD",
    "OPN_BS_YR_MON",
    "MRTY_BS_YR_MON",
    "FST_DLQ_RCKN_DT",
    "DLQ_AMT",
]


def year_from_ym(value: float | int) -> float:
    if pd.isna(value):
        return np.nan
    value = int(value)
    return value // 100 if value >= 10000 else value


def period_to_pair(period: float | int) -> tuple[int, int]:
    reverse = {v: k for k, v in REFERENCE_PAIR_TO_PERIOD.items()}
    if pd.isna(period) or int(period) not in reverse:
        raise ValueError(f"Unknown PERIOD value: {period}")
    return reverse[int(period)]


def _valid_com_values(series: pd.Series) -> pd.Series:
    values = series.dropna()
    return values[~values.astype(str).isin(["", "NULL", "None", "nan", "NaN"])]


def first_last_row_values(group: pd.DataFrame, col: str) -> tuple[float, float, float]:
    ordered = group.sort_values(MONTH_COL, ascending=False, kind="mergesort")
    values = pd.to_numeric(ordered[col], errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    avg = float(values.mean())
    last = float(values[0])
    slope = float(values[0] - values[1]) if len(values) >= 2 else np.nan
    return avg, last, slope


def two_period_diff(group: pd.DataFrame, col: str, agg: str = "sum") -> float:
    periods = sorted(group[MONTH_COL].dropna().astype(int).unique())
    if len(periods) != 2:
        return np.nan
    first = group[group[MONTH_COL] == periods[0]][col]
    second = group[group[MONTH_COL] == periods[1]][col]
    if agg == "mean":
        return float(pd.to_numeric(second, errors="coerce").mean() - pd.to_numeric(first, errors="coerce").mean())
    return float(pd.to_numeric(second, errors="coerce").fillna(0).sum() - pd.to_numeric(first, errors="coerce").fillna(0).sum())


def apply_final_account_scope_filters(account_df: pd.DataFrame, borrower_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        account_df,
        [ID_COL, ACCOUNT_COL, MONTH_COL, "LN_GOODS_CD", "OPN_BS_YR_MON", "MRTY_BS_YR_MON"],
        "account_df",
    )
    require_columns(borrower_df, [ID_COL, "PERIOD"], "borrower_df")
    account_df = account_df.copy()
    account_df["_row_order"] = np.arange(len(account_df))
    borrower_periods = borrower_df[[ID_COL, "PERIOD"]].drop_duplicates(ID_COL).copy()
    pairs = borrower_periods["PERIOD"].map(period_to_pair)
    borrower_periods["_earlier_month"] = pairs.map(lambda pair: pair[0])
    borrower_periods["_later_month"] = pairs.map(lambda pair: pair[1])

    scoped = account_df.merge(borrower_periods, on=ID_COL, how="inner")
    scoped = scoped[scoped["LN_GOODS_CD"].isin(FINAL_ACCOUNT_PRODUCT_CODES)].copy()

    account_summary = (
        scoped.groupby([ID_COL, ACCOUNT_COL], sort=False)
        .agg(
            _months=(MONTH_COL, lambda values: set(int(value) for value in values.dropna())),
            _earlier_month=("_earlier_month", "first"),
            _later_month=("_later_month", "first"),
            _open_ym=("OPN_BS_YR_MON", "first"),
            _maturity_ym=("MRTY_BS_YR_MON", "first"),
        )
        .reset_index()
    )
    account_summary["_open_year"] = account_summary["_open_ym"].map(year_from_ym)
    account_summary["_maturity_year"] = account_summary["_maturity_ym"].map(year_from_ym)
    account_summary["_later_year"] = account_summary["_later_month"] // 100
    account_summary["_has_earlier"] = [
        earlier in months for earlier, months in zip(account_summary["_earlier_month"], account_summary["_months"])
    ]
    account_summary["_has_later"] = [
        later in months for later, months in zip(account_summary["_later_month"], account_summary["_months"])
    ]
    later_only_bad_open = (
        (~account_summary["_has_earlier"])
        & account_summary["_has_later"]
        & (account_summary["_open_year"] != account_summary["_later_year"])
    )
    earlier_only_bad_maturity = (
        account_summary["_has_earlier"]
        & (~account_summary["_has_later"])
        & (account_summary["_maturity_year"] > account_summary["_later_year"])
    )
    valid_accounts = account_summary.loc[
        ~(later_only_bad_open | earlier_only_bad_maturity), [ID_COL, ACCOUNT_COL]
    ]
    scoped = scoped.merge(valid_accounts.assign(_valid_account=1), on=[ID_COL, ACCOUNT_COL], how="inner")
    scoped = scoped.sort_values("_row_order", kind="mergesort")
    return scoped.drop(columns=["_earlier_month", "_later_month", "_valid_account", "_row_order"])


def duplicate_content_borrower_ids(account_df: pd.DataFrame) -> set:
    require_columns(account_df, [ID_COL, ACCOUNT_COL] + DUPLICATE_ACCOUNT_CONTENT_COLS, "account_df")
    duplicated = account_df.duplicated([ID_COL] + DUPLICATE_ACCOUNT_CONTENT_COLS, keep=False)
    return set(account_df.loc[duplicated, ID_COL])


def build_borrower_features_from_notebook_formulas(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, [ID_COL, ACCOUNT_COL, MONTH_COL, "LN_CONT_TOT", "LN_BAL_TOT"], "df")
    rows = []
    for borrower_id, group in df.groupby(ID_COL, sort=False):
        first = group.iloc[0]
        row = {ID_COL: borrower_id}
        for col in BASE_PERSON_COLS:
            row[col] = first.get(col, np.nan)

        row["JOB_MOVE_YN"] = int(group["JOB_CD"].dropna().nunique() > 1) if "JOB_CD" in group else np.nan
        row["HOM_MOVE_YN"] = int(group["HOM_CD"].dropna().nunique() > 1) if "HOM_CD" in group else np.nan
        row["COM_MOVE_YN"] = int(_valid_com_values(group["COM_CD"]).nunique() > 1) if "COM_CD" in group else np.nan
        row["PERIOD"] = int(group["PERIOD"].iloc[0]) if "PERIOD" in group else np.nan

        for src, prefix in [("YR_INCOM_MST_AMT", "INCOM"), ("LST_SCORE", "SCORE")]:
            if src in group:
                avg, last, slope = first_last_row_values(group, src)
                row[f"{prefix}_AVG"] = avg
                row[f"{prefix}_LAST"] = last
                row[f"{prefix}_SLOPE"] = slope
        for src, prefix in [("LN_AMT_SUN", "LN_SUN"), ("LN_AMT_NHP", "LN_NHP"), ("LN_AMT_ETC", "LN_ETC")]:
            if src in group:
                avg, last, _ = first_last_row_values(group, src)
                row[f"{prefix}_AVG"] = avg
                row[f"{prefix}_LAST"] = last
                row[f"{prefix}_SLOPE"] = two_period_diff(group, src, agg="sum")
        if "YR_INCOM_MST_AMT" in group:
            row["INCOM_SLOPE"] = two_period_diff(group, "YR_INCOM_MST_AMT", agg="sum")
        if "LN_CONT_TOT" in group:
            contract_values = pd.to_numeric(group["LN_CONT_TOT"], errors="coerce").dropna()
            row["LN_CONT_MEAN"] = float(contract_values.mean()) if len(contract_values) else np.nan
            row["LN_CONT_MAX"] = float(contract_values.max()) if len(contract_values) else np.nan

        if all(c in group for c in ["LN_AMT_SUN", "LN_AMT_NHP", "LN_AMT_ETC"]):
            ordered = group.sort_values(MONTH_COL, kind="mergesort")
            latest_month = ordered[MONTH_COL].iloc[-1]
            latest_rows = ordered[ordered[MONTH_COL] == latest_month]
            row[POLICY_LOAN_COL] = int(policy_loan_exposure(latest_rows).any())

        row.update(categorical_summary(group, "BIS_AREA", "LN_CONT_TOT", "BIS_AREA", [1, 2, 3, 4, 5, 6]))
        goods = categorical_summary(group, "LN_GOODS_CD", "LN_CONT_TOT", "LN_GOODS", [1, 2, 3, 4])
        row.update({k: v for k, v in goods.items() if not k.startswith("LN_GOODS_FLAG_") and k != "LN_GOODS_LAST"})
        row.update(categorical_summary(group, "TX_TP_CD", "LN_CONT_TOT", "TX_TP", [1, 2, 3, 4]))
        fnd = categorical_summary(group, "FND_PURP_CD", "LN_CONT_TOT", "FND_PURP", [1, 2, 3, 4])
        row.update({k: v for k, v in fnd.items() if k not in {"FND_PURP_MOST", "FND_PURP_MOST_RATIO", "FND_PURP_LAST"}})

        recent_year_by_account = group.groupby(ACCOUNT_COL)[MONTH_COL].max().astype(int) // 100
        open_year = as_numeric(group["OPN_BS_YR_MON"]).astype("Int64") // 100
        maturity_year = as_numeric(group["MRTY_BS_YR_MON"]).astype("Int64") // 100
        recent_year = group[ACCOUNT_COL].map(recent_year_by_account)
        row["OPN_BS_COUNT"] = int((open_year == recent_year).sum())
        row["MRTY_BS_COUNT"] = int((maturity_year == recent_year).sum())
        row.update(loan_term_features(group))
        row.update(balance_ratios(group))

        periods = sorted(group[MONTH_COL].dropna().astype(int).unique())
        if len(periods) == 2:
            first_balance = as_numeric(group.loc[group[MONTH_COL] == periods[0], "LN_BAL_TOT"]).sum()
            last_balance = as_numeric(group.loc[group[MONTH_COL] == periods[1], "LN_BAL_TOT"]).sum()
            row["LN_BAL_SLOPE"] = float(last_balance - first_balance)
        else:
            row["LN_BAL_SLOPE"] = np.nan
        row["LN_BAL_MEAN"] = float(as_numeric(group["LN_BAL_TOT"]).mean())
        row["LN_COUNT"] = int(group[ACCOUNT_COL].nunique())
        row["HOM_IN_SMA_YN"] = int(str(row.get("HOM_CD", ""))[:2] in {"11", "41", "28"})

        repay_sum = group[["LN_AMT_SUN", "LN_AMT_NHP", "LN_AMT_ETC"]].fillna(0).sum(axis=1).sum()
        recent_income = as_numeric(group.sort_values([ID_COL, MONTH_COL], kind="mergesort")["YR_INCOM_MST_AMT"]).dropna()
        recent_income_value = recent_income.iloc[-1] if len(recent_income) else np.nan
        row["DSR"] = float(repay_sum / recent_income_value) if pd.notna(recent_income_value) and recent_income_value > 0 else np.nan
        row[TARGET_COL] = int(account_delinquency_indicator(group).any())
        rows.append(row)
    return pd.DataFrame(rows)


def apply_final_predictor_validity_filtering(
    borrower_df: pd.DataFrame,
    account_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    require_columns(borrower_df, [ID_COL, "SPLIT_LN_FLAG", "PERIOD"], "borrower_df")
    account_scope = apply_final_account_scope_filters(account_df, borrower_df)
    duplicate_ids = duplicate_content_borrower_ids(account_scope)
    keep_mask = ~borrower_df[ID_COL].isin(duplicate_ids)
    final_df = borrower_df.loc[keep_mask].copy()

    retained_split_ids = set(final_df.loc[final_df["SPLIT_LN_FLAG"].eq(1), ID_COL])
    if retained_split_ids:
        recalculated = build_borrower_features_from_notebook_formulas(
            account_scope[account_scope[ID_COL].isin(retained_split_ids)].copy()
        )
        preserve_from_target_eligible = {
            ID_COL,
            "SPLIT_LN_FLAG",
            "PERIOD",
            POLICY_LOAN_COL,
            "SCORE_LAST",
            "SCORE_SLOPE",
            "LN_SUN_LAST",
            "LN_NHP_LAST",
            "LN_ETC_LAST",
            "TX_TP_FIRST",
        }
        common_update_cols = [
            col for col in recalculated.columns if col in final_df.columns and col not in preserve_from_target_eligible
        ]
        final_df = final_df.set_index(ID_COL)
        recalculated = recalculated.set_index(ID_COL)
        final_df.loc[recalculated.index, common_update_cols] = recalculated[common_update_cols]
        final_df = final_df.reset_index()

    final_df = final_df.drop(columns=["SPLIT_LN_FLAG"], errors="ignore")
    summary = {
        "input_borrowers": int(len(borrower_df)),
        "split_flagged_borrowers": int(borrower_df["SPLIT_LN_FLAG"].eq(1).sum()),
        "duplicate_content_removed_borrowers": int(len(duplicate_ids)),
        "retained_split_flagged_borrowers": int(len(retained_split_ids)),
        "output_borrowers": int(len(final_df)),
    }
    return final_df, summary


def run(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    cohort = read_csv(config["cohort_account_csv"])

    target_eligible_path = config.get("target_period_eligible_borrowers_csv")
    if target_eligible_path:
        features = read_csv(target_eligible_path)
    else:
        features = build_borrower_features(cohort)
        write_csv(features, config["borrower_features_csv"])

    if "final_borrower_features_csv" in config and "SPLIT_LN_FLAG" in features.columns:
        final_features, summary = apply_final_predictor_validity_filtering(features, cohort)
        write_csv(final_features, config["final_borrower_features_csv"])
        summary_path = config.get("final_predictor_filtering_summary_csv")
        if summary_path:
            write_csv(pd.DataFrame([summary]), summary_path)
        return final_features

    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
