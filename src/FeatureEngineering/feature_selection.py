from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (
    FINAL_G_FEATURES,
    FINAL_P_FEATURES,
    ID_COL,
    POLICY_LOAN_COL,
    SENSITIVE_OR_DESCRIPTIVE_COLS,
    TARGET_COL,
    load_config,
    read_csv,
    safe_feature_columns,
    write_csv,
)


@dataclass(frozen=True)
class FeatureSelectionConfig:
    correlation_abs_threshold: float = 0.70
    iv_min_threshold: float = 0.10
    threshold_in: float = 0.01
    threshold_out: float = 0.05


SELECTED_DATASET_OUTPUT_FILES = {
    "general_borrower_g_model": "06_general_borrower_dataset_g_model_features.csv",
    "all_borrowers_p_model_feature_mapping": "06_all_borrowers_aligned_to_p_model_features.csv",
    "policy_loan_borrower_g_model_diagnostic": "06_policy_loan_borrowers_aligned_to_g_model_features.csv",
    "policy_loan_borrower_p_model": "06_policy_loan_borrower_dataset_p_model_features.csv",
}


def drop_non_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in SENSITIVE_OR_DESCRIPTIVE_COLS if c in df.columns]
    return df.drop(columns=drop_cols)


def drop_correlated_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    id_col: str = ID_COL,
    threshold: float = 0.70,
) -> pd.DataFrame:
    feature_df = df.drop(columns=[c for c in [target_col, id_col] if c in df.columns]).copy()
    numeric = feature_df.apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]
    return df.drop(columns=to_drop)


def calculate_iv_table(df: pd.DataFrame, target_col: str = TARGET_COL, id_col: str = ID_COL) -> pd.DataFrame:
    try:
        import scorecardpy as sc
    except ImportError as exc:
        raise ImportError("scorecardpy is required for IV filtering") from exc

    features = [c for c in df.columns if c not in {target_col, id_col}]
    bins = sc.woebin(df[features + [target_col]], y=target_col)
    rows = []
    for variable, bin_df in bins.items():
        rows.append({"variable": variable, "info_value": float(bin_df["total_iv"].iloc[0])})
    return pd.DataFrame(rows)


def drop_low_iv_features(df: pd.DataFrame, iv_table: pd.DataFrame, min_iv: float = 0.10) -> pd.DataFrame:
    low_iv = iv_table.loc[iv_table["info_value"] < min_iv, "variable"].tolist()
    return df.drop(columns=[c for c in low_iv if c in df.columns])


def stepwise_pvalue(
    X: pd.DataFrame,
    y: pd.Series,
    threshold_in: float = 0.01,
    threshold_out: float = 0.05,
) -> list[str]:
    included: list[str] = []
    candidates = list(X.columns)

    while True:
        changed = False
        excluded = [c for c in candidates if c not in included]
        new_pvalues = pd.Series(index=excluded, dtype=float)

        for col in excluded:
            try:
                model = sm.Logit(y, sm.add_constant(X[included + [col]], has_constant="add")).fit(disp=0)
                new_pvalues[col] = model.pvalues[col]
            except Exception:
                continue

        if not new_pvalues.empty and new_pvalues.min() < threshold_in:
            included.append(str(new_pvalues.idxmin()))
            changed = True

        if included:
            try:
                model = sm.Logit(y, sm.add_constant(X[included], has_constant="add")).fit(disp=0)
                pvalues = model.pvalues.iloc[1:]
                if not pvalues.empty and pvalues.max() > threshold_out:
                    included.remove(str(pvalues.idxmax()))
                    changed = True
            except Exception:
                pass

        if not changed:
            break

    return included


def make_selected_dataset(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    keep_cols = [ID_COL] + selected_features + [TARGET_COL]
    return df[safe_feature_columns(df, keep_cols)].copy()


def build_final_datasets(df_person: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df_amt_person = df_person[df_person[POLICY_LOAN_COL] == 1].copy()
    return {
        "general_borrower_g_model": make_selected_dataset(df_person, FINAL_G_FEATURES),
        "all_borrowers_p_model_feature_mapping": make_selected_dataset(df_person, FINAL_P_FEATURES),
        "policy_loan_borrower_g_model_diagnostic": make_selected_dataset(df_amt_person, FINAL_G_FEATURES),
        "policy_loan_borrower_p_model": make_selected_dataset(df_amt_person, FINAL_P_FEATURES),
    }


def run(config_path: str) -> dict[str, pd.DataFrame]:
    config = load_config(config_path)
    df_person = read_csv(config["final_borrower_features_csv"])
    datasets = build_final_datasets(df_person)

    out_dir = config.get("selected_features_dir")
    if out_dir:
        for name, data in datasets.items():
            output_file = SELECTED_DATASET_OUTPUT_FILES.get(name, f"{name}.csv")
            write_csv(data, f"{out_dir}/{output_file}")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
