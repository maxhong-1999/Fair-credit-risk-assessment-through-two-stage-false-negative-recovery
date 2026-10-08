"""Appendix C: fold-separated validation beyond retrospectively known FNs.

This is separate from the original FN-centred audit in two_stage_evaluation.py.
It reads three restricted, borrower-level feature files and prints only the
aggregate results for full and capacity-limited Stage 2 reassessment. No
borrower IDs or row-level predictions are written.

Example:
    python src/Experiments/additional_fold_validation.py \
        --general-csv /restricted/06_general_borrower_dataset_g_model_features.csv \
        --policy-csv /restricted/06_policy_loan_borrower_dataset_p_model_features.csv \
        --mapped-csv /restricted/06_all_borrowers_aligned_to_p_model_features.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


ID = "KCB_DEID1_ENCRYPT"
TARGET = "DLQ_ANY_YN"
SEED = 42
FOLDS = 5
CAPACITIES = (0.05, 0.10, 0.20, 1.0)

# Fixed feature specification used by the reported additional validation.
G_FEATURES = (
    "BIS_AREA_FLAG_01", "BIS_AREA_FLAG_03", "BIS_AREA_FLAG_05",
    "BIS_AREA_LAST", "BIS_AREA_MAX_RATIO", "FND_PURP_FLAG_01",
    "FND_PURP_MAX_RATIO", "LN_BAL_MEAN", "LN_BAL_SLOPE",
    "LN_CONT_LN_LST_RATIO", "LN_MOST_RATIO", "LN_SUN_AVG",
    "LN_TERM_LNG_RATIO", "LN_TERM_SRT_RATIO", "MRTY_BS_COUNT",
    "OPN_BS_COUNT", "TX_TP_FLAG_01", "TX_TP_FLAG_03",
    "TX_TP_FLAG_04", "TX_TP_LAST",
)
P_FEATURES = (
    "BIS_AREA_FLAG_03", "BIS_AREA_FLAG_05", "BIS_AREA_LAST",
    "LN_BAL_SLOPE", "LN_NHP_AVG", "LN_SUN_SLOPE",
)

# Vanilla XGBoost configuration for Stage 1, consistent with the primary audit.
STAGE1_PARAMS = dict(
    random_state=SEED,
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="logloss",
)
STAGE2_PARAMS = dict(
    colsample_bytree=1.0, learning_rate=0.1, max_depth=2,
    n_estimators=50, reg_alpha=1.0, reg_lambda=0.1, subsample=0.9,
    random_state=SEED, objective="binary:logistic", tree_method="hist",
    eval_metric="logloss",
)


def read_inputs(general_csv: Path, policy_csv: Path, mapped_csv: Path):
    general, policy, mapped = (
        pd.read_csv(path, dtype={ID: "string"})
        for path in (general_csv, policy_csv, mapped_csv)
    )
    for name, frame in (("general", general), ("policy", policy), ("mapped", mapped)):
        if ID not in frame or frame[ID].isna().any() or frame[ID].duplicated().any():
            raise ValueError(f"{name} needs one non-null row per borrower ID")
    ids = general[ID].astype(str).to_numpy()
    policy_ids = policy[ID].astype(str).to_numpy()
    mapped_ids = mapped[ID].astype(str).to_numpy()
    if not set(policy_ids).issubset(ids) or set(mapped_ids) != set(ids):
        raise ValueError("Policy IDs must be a subset; mapped IDs must match general IDs")
    for name, frame, features in (
        ("general", general, G_FEATURES),
        ("policy", policy, P_FEATURES),
        ("mapped", mapped, P_FEATURES),
    ):
        missing = set(features).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing features: {sorted(missing)}")
    if TARGET not in general or TARGET not in policy:
        raise ValueError(f"General and policy files need {TARGET}")
    y = pd.to_numeric(general[TARGET], errors="raise")
    policy_y = pd.to_numeric(policy[TARGET], errors="raise")
    if y.isna().any() or policy_y.isna().any():
        raise ValueError("Targets must not be missing")
    if not set(y.unique()).issubset({0, 1}) or set(y.unique()) != {0, 1}:
        raise ValueError("General target must contain both binary classes")
    if not set(policy_y.unique()).issubset({0, 1}):
        raise ValueError("Policy target must be binary")
    policy_positions = pd.Index(ids).get_indexer(policy_ids)
    if not np.array_equal(y.to_numpy()[policy_positions], policy_y.to_numpy()):
        raise ValueError("General and policy targets disagree for shared borrowers")

    x_g = general[list(G_FEATURES)].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    mapped_by_id = mapped.set_index(ID).loc[ids, list(P_FEATURES)]
    x_p = mapped_by_id.apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    x_p[policy_positions] = (
        policy[list(P_FEATURES)].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    )
    policy_mask = np.zeros(len(ids), dtype=bool)
    policy_mask[policy_positions] = True
    return y.to_numpy(np.int8), x_g, x_p, policy_mask


def model(params: dict, device: str) -> XGBClassifier:
    options = dict(params)
    options.setdefault("n_jobs", -1)
    options.setdefault("verbosity", 0)
    return XGBClassifier(**options, device=device)


def inner_oof(x: np.ndarray, y: np.ndarray, params: dict, device: str) -> np.ndarray:
    if min(np.bincount(y, minlength=2)) < FOLDS:
        raise ValueError("Each inner-training class needs at least five borrowers")
    scores = np.full(len(y), np.nan)
    for train, valid in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(x, y):
        fitted = model(params, device).fit(x[train], y[train])
        scores[valid] = fitted.predict_proba(x[valid])[:, 1]
    if np.isnan(scores).any():
        raise AssertionError("Incomplete inner out-of-fold scores")
    return scores


def youden(y: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax(tpr[finite] - fpr[finite])])


def fold_predictions(
    y: np.ndarray, x_g: np.ndarray, x_p: np.ndarray,
    policy_mask: np.ndarray, device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fold_id = np.full(len(y), -1, dtype=np.int8)
    s1_score = np.full(len(y), np.nan)
    s1_bad = np.full(len(y), -1, dtype=np.int8)
    s2_escalate = np.zeros(len(y), dtype=bool)
    split = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(split.split(x_g, y), start=1):
        fold_id[test] = fold
        s1_threshold = youden(y[train], inner_oof(x_g[train], y[train], STAGE1_PARAMS, device))
        s1 = model(STAGE1_PARAMS, device).fit(x_g[train], y[train])
        s1_score[test] = s1.predict_proba(x_g[test])[:, 1]
        s1_bad[test] = s1_score[test] >= s1_threshold

        policy_train = train[policy_mask[train]]
        raw_oof = inner_oof(x_p[policy_train], y[policy_train], STAGE2_PARAMS, device)
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED)
        calibrator.fit(raw_oof.reshape(-1, 1), y[policy_train])
        calibrated_oof = calibrator.predict_proba(raw_oof.reshape(-1, 1))[:, 1]
        s2_threshold = youden(y[policy_train], calibrated_oof)
        s2 = model(STAGE2_PARAMS, device).fit(x_p[policy_train], y[policy_train])
        negative = test[s1_bad[test] == 0]
        if len(negative):
            raw = s2.predict_proba(x_p[negative])[:, 1]
            calibrated = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
            s2_escalate[negative] = calibrated >= s2_threshold
    if (fold_id < 1).any() or (s1_bad < 0).any() or np.isnan(s1_score).any():
        raise AssertionError("Incomplete held-out predictions")
    if (s2_escalate & (s1_bad != 0)).any():
        raise AssertionError("Stage 2 escalated a Stage 1 positive")
    return fold_id, s1_score, s1_bad, s2_escalate


def metrics(y: np.ndarray, predicted_bad: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted_bad, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, recall=recall,
                precision=precision, specificity=specificity, f1=f1)


def review_masks(fold_id: np.ndarray, s1_score: np.ndarray, s1_bad: np.ndarray):
    """Select review capacity within each fold using Stage 1 scores only."""
    masks = {}
    for capacity in CAPACITIES:
        selected = np.zeros(len(s1_bad), dtype=bool)
        for fold in range(1, FOLDS + 1):
            candidates = np.flatnonzero((fold_id == fold) & (s1_bad == 0))
            order = np.lexsort((candidates, -s1_score[candidates]))
            count = len(candidates) if capacity == 1.0 else math.ceil(capacity * len(candidates))
            selected[candidates[order[:count]]] = True
        masks[capacity] = selected
    return masks


def summarize(y: np.ndarray, fold_id: np.ndarray, s1_score: np.ndarray,
              s1_bad: np.ndarray, s2_escalate: np.ndarray) -> dict:
    stage1 = metrics(y, s1_bad)
    negative = s1_bad == 0
    full = metrics(y, s1_bad | s2_escalate)
    capacities = []
    for capacity, selected in review_masks(fold_id, s1_score, s1_bad).items():
        escalated = selected & s2_escalate
        final = metrics(y, s1_bad | escalated)
        recovered = final["tp"] - stage1["tp"]
        added_fp = final["fp"] - stage1["fp"]
        capacities.append(dict(
            capacity=capacity, reassessed_borrowers=int(selected.sum()),
            recovered_fn=recovered,
            fn_recovery_rate_pct=100.0 * recovered / stage1["fn"] if stage1["fn"] else 0.0,
            additional_fp=added_fp,
            fp_per_recovered_fn=added_fp / recovered if recovered else None,
        ))
    if capacities[-1]["reassessed_borrowers"] != int(negative.sum()):
        raise AssertionError("Full review must cover all Stage 1 negatives")
    if full["fn"] != stage1["fn"] - capacities[-1]["recovered_fn"]:
        raise AssertionError("FN accounting does not reconcile")
    return dict(
        borrower_n=len(y), actual_positive_n=int(y.sum()),
        stage1_predicted_negative_n=int(negative.sum()),
        table_c1=dict(stage1=stage1, stage1_plus_stage2=full,
                      recovered_fn=full["tp"] - stage1["tp"],
                      additional_fp=full["fp"] - stage1["fp"]),
        table_c2=capacities,
        stage1_params=STAGE1_PARAMS, stage2_params=STAGE2_PARAMS,
        outer_folds=FOLDS, inner_folds=FOLDS, seed=SEED,
    )


def self_check() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)
    folds = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int8)
    scores = np.array([.4, .2, .3, .1, .5, .2, .1, .05, .1, .02])
    stage1 = np.zeros(10, dtype=np.int8)
    stage2 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    result = summarize(y, folds, scores, stage1, stage2)
    assert result["table_c1"]["recovered_fn"] == 2
    assert result["table_c1"]["additional_fp"] == 0
    assert result["table_c2"][-1]["reassessed_borrowers"] == 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general-csv", type=Path)
    parser.add_argument("--policy-csv", type=Path)
    parser.add_argument("--mapped-csv", type=Path)
    parser.add_argument("--device", default="cpu", help="XGBoost device: cpu or cuda")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check OK")
        return
    if any(path is None for path in (args.general_csv, args.policy_csv, args.mapped_csv)):
        parser.error("--general-csv, --policy-csv, and --mapped-csv are required")
    arrays = read_inputs(args.general_csv, args.policy_csv, args.mapped_csv)
    predictions = fold_predictions(*arrays, args.device)
    print(json.dumps(summarize(arrays[0], *predictions), indent=2))


if __name__ == "__main__":
    main()
