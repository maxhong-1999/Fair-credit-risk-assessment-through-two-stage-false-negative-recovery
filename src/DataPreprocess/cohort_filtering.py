from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

import pandas as pd

from common import (
    ACCOUNT_COL,
    ID_COL,
    MONTH_COL,
    REFERENCE_PAIRS,
    load_config,
    pair_to_period,
    read_csv,
    require_columns,
    write_csv,
)


def has_reference_gap(months: list[int]) -> bool:
    if len(months) < 2:
        return False
    months = sorted(set(int(m) for m in months))
    diffs = [months[i + 1] - months[i] for i in range(len(months) - 1)]
    return any(diff > 100 for diff in diffs)


def remove_discontinuous_accounts(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, [ACCOUNT_COL, MONTH_COL])
    periods = df.groupby(ACCOUNT_COL)[MONTH_COL].apply(lambda s: sorted(s.dropna().unique()))
    gapped_accounts = periods[periods.apply(has_reference_gap)].index
    return df[~df[ACCOUNT_COL].isin(gapped_accounts)].reset_index(drop=True)


def remove_zero_contract_rows(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["LN_CONT_TOT"])
    return df[df["LN_CONT_TOT"] != 0].reset_index(drop=True)


def latest_eligible_pair_by_borrower(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    require_columns(df, [ID_COL, MONTH_COL])
    all_reference_months = sorted({m for pair in REFERENCE_PAIRS for m in pair})
    user_months = df.groupby(ID_COL)[MONTH_COL].unique().apply(lambda s: sorted(int(x) for x in s))

    keep_pairs: dict[str, tuple[int, int]] = {}
    for borrower_id, months in user_months.items():
        month_set = set(months)
        eligible = []
        for i in range(len(all_reference_months) - 1):
            pair = (all_reference_months[i], all_reference_months[i + 1])
            if pair in REFERENCE_PAIRS and set(pair).issubset(month_set):
                eligible.append(pair)
        if eligible:
            keep_pairs[borrower_id] = eligible[-1]
    return keep_pairs


def retain_latest_two_reference_points(df: pd.DataFrame) -> pd.DataFrame:
    keep_pairs = latest_eligible_pair_by_borrower(df)
    keep_rows = []
    for borrower_id, pair in keep_pairs.items():
        keep_rows.append((borrower_id, pair[0], pair_to_period(pair)))
        keep_rows.append((borrower_id, pair[1], pair_to_period(pair)))

    keep_df = pd.DataFrame(keep_rows, columns=[ID_COL, MONTH_COL, "PERIOD"])
    retained = df.merge(keep_df, on=[ID_COL, MONTH_COL], how="inner")
    retained = retained.sort_values([ID_COL, MONTH_COL, ACCOUNT_COL], ascending=[True, False, True])
    return retained.reset_index(drop=True)


def run(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    merged = read_csv(config["merged_csv"])

    filtered = remove_discontinuous_accounts(merged)
    filtered = remove_zero_contract_rows(filtered)
    retained = retain_latest_two_reference_points(filtered)

    write_csv(retained, config["cohort_account_csv"])
    return retained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
