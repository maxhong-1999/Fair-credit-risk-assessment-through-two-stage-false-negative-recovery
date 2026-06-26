from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

import pandas as pd

from common import (
    CUSTOMER_ID_COL,
    ID_COL,
    MONTH_COL,
    POLICY_LOAN_BALANCE_COLS,
    POLICY_LOAN_COL,
    account_delinquency_indicator,
    load_config,
    policy_loan_exposure,
    require_columns,
    read_csv,
    write_csv,
)


CUSTOMER_COLS = [
    CUSTOMER_ID_COL,
    MONTH_COL,
    "GENDER",
    "AGE_CD",
    "JOB_CD",
    "HOM_CD",
    "COM_CD",
    "YR_INCOM_MST_AMT",
    "LST_SCORE",
    "LN_AMT_SUN",
    "LN_AMT_NHP",
    "LN_AMT_ETC",
]


def add_policy_loan_indicator(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[POLICY_LOAN_COL] = policy_loan_exposure(out).map({True: "Y", False: "N"})
    out[POLICY_LOAN_BALANCE_COLS] = out[POLICY_LOAN_BALANCE_COLS].fillna(0)
    return out


def add_account_delinquency_indicator(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DLQ_YN"] = account_delinquency_indicator(out).map({True: "Y", False: "N"})
    return out


def merge_customer_account(customer: pd.DataFrame, account: pd.DataFrame) -> pd.DataFrame:
    require_columns(customer, CUSTOMER_COLS, "customer")
    require_columns(account, [ID_COL, MONTH_COL], "account")

    customer = customer[CUSTOMER_COLS].copy()
    account = account.copy()

    customer["_match_key"] = customer[CUSTOMER_ID_COL].astype(str) + "_" + customer[MONTH_COL].astype(str)
    account["_match_key"] = account[ID_COL].astype(str) + "_" + account[MONTH_COL].astype(str)

    matched_keys = set(customer["_match_key"]) & set(account["_match_key"])
    account = account[account["_match_key"].isin(matched_keys)].copy()

    customer_payload = customer.drop(columns=[CUSTOMER_ID_COL, MONTH_COL])
    merged = account.merge(customer_payload, on="_match_key", how="left")
    merged = merged.drop(columns=["_match_key"])

    month_idx = list(merged.columns).index(MONTH_COL)
    customer_payload_cols = [c for c in CUSTOMER_COLS if c not in {CUSTOMER_ID_COL, MONTH_COL}]
    ordered_cols = (
        list(merged.columns[: month_idx + 1])
        + customer_payload_cols
        + [c for c in merged.columns if c not in set(merged.columns[: month_idx + 1]) | set(customer_payload_cols)]
    )
    return merged[ordered_cols]


def run(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    customer = read_csv(config["raw_customer_csv"])
    account = read_csv(config["raw_account_csv"])

    merged = merge_customer_account(customer, account)
    merged = add_policy_loan_indicator(merged)
    merged = add_account_delinquency_indicator(merged)
    merged["COM_CD"] = merged["COM_CD"].fillna("NULL")

    write_csv(merged, config["merged_csv"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
