from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "private" / "processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "two_stage"

TARGET_COL = "DLQ_ANY_YN"
ID_COL = "KCB_DEID1_ENCRYPT"
RANDOM_STATE = 42
PAPER_STAGE2_THRESHOLD = 0.027991
ANALYSIS_MODE = "two_stage_general_policy_model_paper_mode"

STAGE2_INPUT_FILES = {
    "stage1_general_model_input": "06_general_borrower_dataset_g_model_features.csv",
    "stage2_policy_loan_model_input": "06_policy_loan_borrower_dataset_p_model_features.csv",
    "non_overlap_feature_mapping_input": "06_all_borrowers_aligned_to_p_model_features.csv",
}
LEGACY_INPUT_IDS = {
    "stage1_general_model_input": "C_C",
    "stage2_policy_loan_model_input": "T_T",
    "non_overlap_feature_mapping_input": "C_T",
}
EXPOSURE_CANDIDATES = (
    "EAD",
    "EAD_PROXY",
    "LN_BAL_TOT",
    "LN_BAL_TOT_SUM",
    "TOTAL_LN_BAL",
    "TOT_LN_BAL",
    "LN_BAL_SUM",
)

STAGE1_XGB_PARAMS = {
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "logloss",
}
# Same XGBoost settings used by the final notebook for the Stage 2 P-model
# trained on policy-loan borrowers (legacy internal input id: T_T).
STAGE2_XGB_PARAMS = {
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "logloss",
    "subsample": 0.9,
    "reg_lambda": 0.1,
    "reg_alpha": 1.0,
    "n_estimators": 50,
    "max_depth": 2,
    "learning_rate": 0.1,
    "colsample_bytree": 1.0,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --paths points to a YAML file.") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return loaded


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _resolve_path(raw: str, config_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [
        (config_path.parent / path).resolve(),
        (REPO_ROOT / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _find_path(config: dict[str, Any], config_path: Path, suffix: str, default: Path) -> Path:
    for raw in _iter_strings(config):
        if raw.endswith(suffix):
            return _resolve_path(raw, config_path)
    return default


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse=False)


def _binary_target(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.fillna(0).astype(int).to_numpy()
    mapped = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"Y": 1, "N": 0, "YES": 1, "NO": 0, "TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
    )
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique()[:10])
        raise ValueError(f"Cannot map target values to binary labels: {bad}")
    return mapped.astype(int).to_numpy()


def build_preprocessor_safe(df: pd.DataFrame, target_col: str, id_col: str):
    use_cols = [c for c in df.columns if c not in [target_col, id_col]]
    num_cols = [c for c in use_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in use_cols if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in num_cols if df[c].nunique(dropna=False) > 1]
    cat_cols = [c for c in cat_cols if df[c].nunique(dropna=False) > 1]
    num_tf = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", _one_hot_encoder()),
    ]) if cat_cols else "drop"
    return ColumnTransformer(
        [("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def thr_youden(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, score)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def thr_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, _, thresholds = roc_curve(y_true, score)
    return float(thresholds[int(np.argmin(np.abs(fpr - target_fpr)))])


class Calibrator:
    def __init__(self, method: str = "platt"):
        self.method = method
        self.model: Any | None = None

    def fit(self, score: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "identity":
            self.model = None
        elif self.method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(score, y)
            self.model = model
        else:
            model = LogisticRegression(solver="lbfgs", max_iter=1000)
            model.fit(score.reshape(-1, 1), y)
            self.model = model
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype=float)
        if self.method == "identity" or self.model is None:
            return score
        if isinstance(self.model, IsotonicRegression):
            return self.model.predict(score)
        return self.model.predict_proba(score.reshape(-1, 1))[:, 1]


def train_calibrator(score: np.ndarray, y: np.ndarray, method: str = "platt") -> Calibrator:
    return Calibrator(method=method).fit(score, y)


def build_row_stratified_folds(df: pd.DataFrame, target_col: str, n_splits: int, random_state: int) -> np.ndarray:
    y = _binary_target(df[target_col])
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    labels = np.empty(len(df), dtype=int)
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        labels[valid_idx] = fold
    return labels


def majority_fold_per_id(df: pd.DataFrame, id_col: str, fold_labels: np.ndarray) -> dict[Any, int]:
    tmp = df[[id_col]].copy()
    tmp["_fold"] = fold_labels
    out: dict[Any, int] = {}
    for value, sub in tmp.groupby(id_col, sort=False):
        out[value] = Counter(sub["_fold"].values).most_common(1)[0][0]
    return out


def labels_from_id_map(df: pd.DataFrame, id_col: str, id_to_fold: dict[Any, int], n_splits: int) -> np.ndarray:
    labels = np.full(len(df), -1, dtype=int)
    missing = []
    for idx, value in enumerate(df[id_col].values):
        if value in id_to_fold:
            labels[idx] = id_to_fold[value]
        else:
            missing.append(idx)
    for j, idx in enumerate(missing):
        labels[idx] = j % n_splits
    return labels


def _xgb_classifier(params: dict[str, Any]):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed.") from exc
    return xgb.XGBClassifier(**params)


def fit_xgb_oof_with_foldlabels(
    df: pd.DataFrame,
    fold_labels: np.ndarray,
    model_params: dict[str, Any],
    target_col: str = TARGET_COL,
    id_col: str = ID_COL,
    preprocessor_from_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[Pipeline], np.ndarray, float, float]:
    y = _binary_target(df[target_col])
    x = df.drop(columns=[c for c in [target_col, id_col] if c in df.columns])
    src_df = preprocessor_from_df if preprocessor_from_df is not None else df
    n_splits = int(fold_labels.max()) + 1
    oof = np.zeros(len(df), dtype=float)
    fold_models = []
    fold_metrics = []
    for fold in range(n_splits):
        train_idx = np.where(fold_labels != fold)[0]
        valid_idx = np.where(fold_labels == fold)[0]
        pre = build_preprocessor_safe(src_df, target_col, id_col)
        model = _xgb_classifier(model_params)
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(x.iloc[train_idx], y[train_idx])
        score = pipe.predict_proba(x.iloc[valid_idx])[:, 1]
        oof[valid_idx] = score
        fold_metrics.append([roc_auc_score(y[valid_idx], score), average_precision_score(y[valid_idx], score)])
        fold_models.append(pipe)
    return oof, fold_models, np.asarray(fold_metrics), float(roc_auc_score(y, oof)), float(average_precision_score(y, oof))


def predict_external_avg(sub_df: pd.DataFrame, id_col: str, target_col: str, fold_models: list[Pipeline]) -> pd.Series:
    drop_cols = [c for c in [id_col, target_col] if c in sub_df.columns]
    x_ext = sub_df.drop(columns=drop_cols, errors="ignore")
    preds = [pipe.predict_proba(x_ext)[:, 1] for pipe in fold_models]
    score = np.mean(np.vstack(preds), axis=0)
    return pd.Series(score, index=sub_df[id_col].values).groupby(level=0).mean()


def _exposure(df: pd.DataFrame) -> pd.Series:
    for col in EXPOSURE_CANDIDATES:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(np.ones(len(df)), index=df.index, name="EAD_PROXY")


def _approval_confusion(y_true: np.ndarray, approve: np.ndarray) -> dict[str, int]:
    pred_bad = 1 - approve.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_bad, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def run_two_stage_pipeline(
    df_general_model: pd.DataFrame,
    df_policy_model: pd.DataFrame,
    df_non_overlap_feature_mapping: pd.DataFrame,
    n_splits: int,
    seed: int,
    threshold_method: str,
    fpr_target: float,
    calibrate: str,
    stage2_threshold_override: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels_stage1_general = build_row_stratified_folds(df_general_model, TARGET_COL, n_splits, seed)
    id_to_fold = majority_fold_per_id(df_general_model, ID_COL, labels_stage1_general)
    labels_stage2_policy = labels_from_id_map(df_policy_model, ID_COL, id_to_fold, n_splits)

    p_stage1, stage1_models, stage1_fold_metrics, stage1_auc, stage1_pr_auc = fit_xgb_oof_with_foldlabels(
        df_general_model,
        labels_stage1_general,
        STAGE1_XGB_PARAMS,
        preprocessor_from_df=df_general_model,
    )
    y_stage1 = _binary_target(df_general_model[TARGET_COL])
    if threshold_method == "fpr":
        thr_stage1 = thr_at_fpr(y_stage1, p_stage1, target_fpr=fpr_target)
        thr_stage1_desc = f"FPR@{fpr_target:.2%}"
    else:
        thr_stage1 = thr_youden(y_stage1, p_stage1)
        thr_stage1_desc = "Youden"
    pred_bad_s1 = (p_stage1 >= thr_stage1).astype(int)
    approve_s1 = 1 - pred_bad_s1
    fn_mask = (y_stage1 == 1) & (approve_s1 == 1)
    fn_ids = df_general_model.loc[fn_mask, ID_COL].to_numpy()

    p_stage2_raw, stage2_models, stage2_fold_metrics, stage2_auc, stage2_pr_auc = fit_xgb_oof_with_foldlabels(
        df_policy_model,
        labels_stage2_policy,
        STAGE2_XGB_PARAMS,
        preprocessor_from_df=df_policy_model,
    )
    y_stage2 = _binary_target(df_policy_model[TARGET_COL])
    calibrator = train_calibrator(p_stage2_raw, y_stage2, method=calibrate)
    p_stage2_cal = calibrator.predict(p_stage2_raw)
    if stage2_threshold_override is not None:
        thr_stage2 = float(stage2_threshold_override)
        thr_stage2_desc = "paper_override"
    elif threshold_method == "fpr":
        thr_stage2 = thr_at_fpr(y_stage2, p_stage2_cal, target_fpr=fpr_target)
        thr_stage2_desc = f"FPR@{fpr_target:.2%}_{calibrate}"
    else:
        thr_stage2 = thr_youden(y_stage2, p_stage2_cal)
        thr_stage2_desc = f"Youden_{calibrate}"

    flags = df_general_model[[ID_COL, TARGET_COL]].copy()
    flags[TARGET_COL] = y_stage1
    flags["ANALYSIS_MODE"] = ANALYSIS_MODE
    flags["APPROVE_S1"] = approve_s1.astype(int)
    flags["REVIEW_FLAG"] = flags[ID_COL].isin(set(fn_ids)).astype(int)
    flags["APPROVE_S2_FINAL"] = flags["APPROVE_S1"].copy()
    flags["S1_SCORE"] = p_stage1
    flags["S1_THRESHOLD"] = float(thr_stage1)
    flags["S2_SCORE_CAL"] = np.nan
    flags["S2_THRESHOLD"] = float(thr_stage2)
    flags["REVIEW_SUBGROUP"] = "not_reviewed"
    flags["EAD_PROXY"] = _exposure(df_general_model).to_numpy()

    t_ids = set(df_policy_model[ID_COL].unique())
    overlap_ids = np.array([value for value in fn_ids if value in t_ids])
    nonoverlap_ids = np.array([value for value in fn_ids if value not in t_ids])

    overlap_dec_map: dict[Any, int] = {}
    overlap_score_map: dict[Any, float] = {}
    if len(overlap_ids) > 0:
        idx_overlap = df_policy_model[ID_COL].isin(overlap_ids).to_numpy()
        score_overlap = pd.Series(p_stage2_cal[idx_overlap], index=df_policy_model.loc[idx_overlap, ID_COL].to_numpy()).groupby(level=0).mean()
        overlap_dec_map = (score_overlap >= thr_stage2).map(lambda bad: 0 if bad else 1).to_dict()
        overlap_score_map = score_overlap.to_dict()

    non_dec_map: dict[Any, int] = {}
    non_score_map: dict[Any, float] = {}
    if len(nonoverlap_ids) > 0:
        sub_feature_mapped = df_non_overlap_feature_mapping[df_non_overlap_feature_mapping[ID_COL].isin(nonoverlap_ids)].copy()
        raw_non = predict_external_avg(sub_feature_mapped, ID_COL, TARGET_COL, stage2_models)
        cal_non = pd.Series(calibrator.predict(raw_non.to_numpy()), index=raw_non.index)
        non_dec_map = (cal_non >= thr_stage2).map(lambda bad: 0 if bad else 1).to_dict()
        non_score_map = cal_non.to_dict()

    if overlap_dec_map:
        mask = flags[ID_COL].isin(overlap_dec_map)
        flags.loc[mask, "APPROVE_S2_FINAL"] = flags.loc[mask, ID_COL].map(overlap_dec_map).astype(int).to_numpy()
        flags.loc[mask, "S2_SCORE_CAL"] = flags.loc[mask, ID_COL].map(overlap_score_map).to_numpy()
        flags.loc[mask, "REVIEW_SUBGROUP"] = "overlap_policy_loan_borrowers"

    if non_dec_map:
        mask = flags[ID_COL].isin(non_dec_map)
        flags.loc[mask, "APPROVE_S2_FINAL"] = flags.loc[mask, ID_COL].map(non_dec_map).astype(int).to_numpy()
        flags.loc[mask, "S2_SCORE_CAL"] = flags.loc[mask, ID_COL].map(non_score_map).to_numpy()
        flags.loc[mask, "REVIEW_SUBGROUP"] = "non_overlap_feature_mapped"

    flags["RECOVERED_FN"] = ((flags["REVIEW_FLAG"] == 1) & (flags["APPROVE_S2_FINAL"] == 0)).astype(int)

    s1_confusion = _approval_confusion(y_stage1, flags["APPROVE_S1"].to_numpy())
    s2_confusion = _approval_confusion(y_stage1, flags["APPROVE_S2_FINAL"].to_numpy())
    summary = {
        "analysis_mode": ANALYSIS_MODE,
        "n_borrowers": int(len(flags)),
        "n_splits": int(n_splits),
        "stage1_model": "G-model XGBoost",
        "stage2_model": "P-model XGBoost",
        "stage1_training_input": "entire borrower dataset (general borrower dataset)",
        "stage2_training_input": "policy loan borrower dataset",
        "non_overlap_reassessment_input": "borrower-level variables aligned to P-model feature definitions",
        "legacy_stage1_input_id": LEGACY_INPUT_IDS["stage1_general_model_input"],
        "legacy_stage2_input_id": LEGACY_INPUT_IDS["stage2_policy_loan_model_input"],
        "legacy_non_overlap_feature_mapping_id": LEGACY_INPUT_IDS["non_overlap_feature_mapping_input"],
        "stage1_params": STAGE1_XGB_PARAMS,
        "stage2_params": STAGE2_XGB_PARAMS,
        "stage1_auc": float(stage1_auc),
        "stage1_pr_auc": float(stage1_pr_auc),
        "stage2_auc": float(stage2_auc),
        "stage2_pr_auc": float(stage2_pr_auc),
        "stage1_threshold": float(thr_stage1),
        "stage1_threshold_desc": thr_stage1_desc,
        "stage2_threshold": float(thr_stage2),
        "stage2_threshold_desc": thr_stage2_desc,
        "calibration": calibrate,
        "stage1_fn_count": int(fn_mask.sum()),
        "overlap_fn_count": int(len(overlap_ids)),
        "non_overlap_fn_count": int(len(nonoverlap_ids)),
        "recovered_fn_count": int(flags["RECOVERED_FN"].sum()),
        "stage1_confusion": s1_confusion,
        "stage2_confusion": s2_confusion,
        "stage1_fold_metrics": stage1_fold_metrics.tolist(),
        "stage2_fold_metrics": stage2_fold_metrics.tolist(),
    }
    return flags, summary


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.paths).resolve()
    config = _load_yaml(config_path)
    selected_features_dir = config.get("selected_features_dir")
    input_paths = {}
    for key, suffix in STAGE2_INPUT_FILES.items():
        default_input = (
            _resolve_path(f"{selected_features_dir}/{suffix}", config_path)
            if selected_features_dir
            else DEFAULT_DATA_DIR / suffix
        )
        input_paths[key] = _find_path(config, config_path, suffix, default_input)
    missing = {key: str(path) for key, path in input_paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    df_general_model = pd.read_csv(input_paths["stage1_general_model_input"])
    df_policy_model = pd.read_csv(input_paths["stage2_policy_loan_model_input"])
    df_non_overlap_feature_mapping = pd.read_csv(input_paths["non_overlap_feature_mapping_input"])
    threshold_override = None if args.stage2_threshold_override == "none" else float(args.stage2_threshold_override)
    flags, summary = run_two_stage_pipeline(
        df_general_model,
        df_policy_model,
        df_non_overlap_feature_mapping,
        n_splits=args.n_splits,
        seed=args.seed,
        threshold_method=args.threshold_method,
        fpr_target=args.fpr_target,
        calibrate=args.calibrate,
        stage2_threshold_override=threshold_override,
    )

    configured_output_dir = config.get("model_outputs_dir")
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif configured_output_dir:
        output_dir = _resolve_path(str(configured_output_dir), config_path)
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    flags.to_csv(output_dir / "06_two_stage_decision_flags.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "06_two_stage_summary.csv", index=False)
    with (output_dir / "06_two_stage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final two-stage G-model/P-model evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--threshold-method", choices=["youden", "fpr"], default="youden")
    parser.add_argument("--fpr-target", type=float, default=0.01)
    parser.add_argument("--calibrate", choices=["platt", "isotonic", "identity"], default="platt")
    parser.add_argument("--stage2-threshold-override", default=str(PAPER_STAGE2_THRESHOLD))
    args = parser.parse_args()
    args.paths = args.config
    return args


if __name__ == "__main__":
    run(parse_args())
