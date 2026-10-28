"""Current Appendix C: fixed vanilla/tuned G/P 2x2, aggregate outputs only.

The portable runner refits all 120 models. It does not require private caches.
Training is explicit and was not run during packaging. See README.md.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
ID = "KCB_DEID1_ENCRYPT"
TARGET = "DLQ_ANY_YN"
SEED = 42
FOLDS = 5
ARMS = ("Gv_Pv", "Gv_Pt", "Gt_Pv", "Gt_Pt")


def read_inputs(general_csv, policy_csv, mapped_csv, cfg):
    """Align by string ID; preserve general row order and frozen feature order."""
    general, policy, mapped = [pd.read_csv(p, dtype={ID: "string"})
                               for p in (general_csv, policy_csv, mapped_csv)]
    for name, frame, family in (("general", general, "G"), ("policy", policy, "P"),
                                ("mapped", mapped, "P")):
        if ID not in frame or frame[ID].isna().any() or frame[ID].duplicated().any():
            raise ValueError(f"{name}: one non-null row per unique borrower ID required")
        required = cfg[family + "_features"]
        if set(required).difference(frame.columns):
            raise ValueError(f"{name}: required features are missing; see protocol")
        values = frame[required]
        if not all(pd.api.types.is_numeric_dtype(t) for t in values.dtypes):
            raise ValueError(f"{name}: features must be numeric")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name}: frozen inputs require finite, nonmissing features")
        expected = cfg[family + "_raw_column_dtypes"]
        if any(str(values[k].dtype) != expected[k] for k in required):
            raise ValueError(f"{name}: feature dtypes differ from frozen protocol")
    for name, frame in (("general", general), ("policy", policy)):
        if TARGET not in frame or frame[TARGET].isna().any() or not set(frame[TARGET]).issubset({0, 1}):
            raise ValueError(f"{name}: nonmissing binary target required")
    if set(general[TARGET]) != {0, 1}:
        raise ValueError("general: both binary classes required")
    ids = general[ID].astype(str)
    if set(mapped[ID].astype(str)) != set(ids):
        raise ValueError("mapped IDs must equal general IDs")
    positions = pd.Index(ids).get_indexer(policy[ID].astype(str))
    if np.any(positions < 0):
        raise ValueError("policy IDs must be a subset of general IDs")
    y = general[TARGET].to_numpy(dtype=np.int8)
    if not np.array_equal(y[positions], policy[TARGET].to_numpy()):
        raise ValueError("general and policy targets disagree")
    xg = general[cfg["G_features"]].copy()
    xp = mapped.set_index(ID).loc[ids, cfg["P_features"]].reset_index(drop=True)
    for column in cfg["P_features"]:
        xp.loc[positions, column] = policy[column].to_numpy()
    mask = np.zeros(len(y), dtype=bool)
    mask[positions] = True
    return y, xg, xp, mask


def pipeline(params):
    clf = XGBClassifier(**params)
    if any(clf.get_params()[key] != value for key, value in params.items()):
        raise ValueError("Estimator parameters were not applied")
    return Pipeline([("imputer", SimpleImputer(strategy="mean")),
                     ("scaler", StandardScaler()), ("clf", clf)])


def youden(y, score):
    fpr, tpr, threshold = roc_curve(y, score, drop_intermediate=False)
    finite = np.isfinite(threshold)
    return float(threshold[finite][np.argmax((tpr - fpr)[finite])])


def native_decisions(g_scores, p_scores, thresholds):
    decisions = {g: score >= thresholds[g] for g, score in g_scores.items()}
    for arm in ARMS:
        g, p = arm.split("_")
        # No labels enter this gate. Keep every G positive, reassess all G negatives.
        decisions[arm] = decisions[g] | ((~decisions[g]) & (p_scores[p] >= thresholds[p]))
    return decisions


def matched_mask(folds, score, native):
    """Retrospective test-batch ranking; tie break by original row position."""
    chosen = np.zeros(len(native), dtype=bool)
    budgets = []
    for fold in sorted(np.unique(folds)):
        rows = np.flatnonzero(folds == fold)
        count = int(native[rows].sum())
        order = rows[np.lexsort((rows, -score[rows]))]
        chosen[order[:count]] = True
        budgets.append({"fold": int(fold), "two_stage_flags": count,
                        "standalone_flags": int(chosen[rows].sum())})
    return chosen, budgets


def metrics(y, decision):
    tp = int(np.sum((y == 1) & decision))
    fp = int(np.sum((y == 0) & decision))
    tn = int(np.sum((y == 0) & ~decision))
    fn = int(np.sum((y == 1) & ~decision))
    def ratio(a, b):
        return a / b if b else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, flagged=tp + fp,
                recall=ratio(tp, tp + fn), precision=ratio(tp, tp + fp),
                specificity=ratio(tn, tn + fp), f1=ratio(2 * tp, 2 * tp + fp + fn))


def summarize(y, folds, g_scores, p_scores, thresholds):
    decisions = native_decisions(g_scores, p_scores, thresholds)
    panel_a, panel_b, budgets, native_folds = [], [], [], []
    for name in ("Gv", "Gt", *ARMS):
        row = {"method": name, **metrics(y, decisions[name])}
        if name in ARMS:
            g = name.split("_")[0]
            base = metrics(y, decisions[g])
            row.update(recovered_fn_from_own_G=row["tp"] - base["tp"],
                       additional_fp_from_own_G=row["fp"] - base["fp"],
                       G_negative_reassessed_n=int((~decisions[g]).sum()))
        panel_a.append(row)
        for fold in sorted(np.unique(folds)):
            take = folds == fold
            native_folds.append({"panel": "A", "method": name, "fold": int(fold),
                                 **metrics(y[take], decisions[name][take])})
    for arm in ARMS:
        g = arm.split("_")[0]
        for comparator, score in (("tuned_G_primary", g_scores["Gt"]),
                                  ("same_first_stage_G_secondary", g_scores[g])):
            chosen, fold_budgets = matched_mask(folds, score, decisions[arm])
            budgets.extend({"arm": arm, "comparator": comparator, **row} for row in fold_budgets)
            for side, mask in (("two_stage", decisions[arm]), ("standalone", chosen)):
                panel_b.append({"arm": arm, "comparator": comparator, "method": side,
                                **metrics(y, mask)})
    r1 = [r for r in panel_b if r["arm"] == "Gv_Pt" and r["comparator"] == "tuned_G_primary"]
    return {"panel_A.csv": panel_a, "panel_B.csv": panel_b,
            "matched_fold_budgets.csv": budgets, "native_fold_results.csv": native_folds,
            "table_R1.csv": r1}


def fit_scores(y, xg, xp, policy, cfg):
    """Refit 4 families x 5 outer folds x (5 inner + 1 outer) = 120 fits."""
    g_scores = {k: np.full(len(y), np.nan) for k in ("Gv", "Gt")}
    p_scores = {k: np.full(len(y), np.nan) for k in ("Pv", "Pt")}
    thresholds = {k: np.full(len(y), np.nan) for k in cfg["params"]}
    folds = np.zeros(len(y), dtype=np.int8)
    threshold_rows = []
    outer = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    # Validate every split before fitting; a policy class must support inner 5-fold CV.
    splits = list(outer.split(xg, y))
    for train, _ in splits:
        for population in (train, train[policy[train]]):
            if min(np.bincount(y[population], minlength=2)) < FOLDS:
                raise ValueError("Each outer-training population needs five cases per class")
    for fold, (train, test) in enumerate(splits, 1):
        folds[test] = fold
        # P populations and scores are shared unchanged across both G arms.
        for name, params in cfg["params"].items():
            is_g = name.startswith("G")
            population = train if is_g else train[policy[train]]
            x = xg if is_g else xp
            inner_y = y[population]
            oof = np.full(len(population), np.nan)
            inner = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
            for itr, iva in inner.split(x.iloc[population], inner_y):
                pipe = pipeline(params)
                pipe.fit(x.iloc[population[itr]], inner_y[itr])
                if pipe[:-1].transform(x.iloc[population[iva[:1]]]).dtype != np.float64:
                    raise RuntimeError("Frozen preprocessing must yield float64")
                oof[iva] = pipe.predict_proba(x.iloc[population[iva]])[:, 1]
            if not np.isfinite(oof).all():
                raise RuntimeError("Incomplete inner OOF scores")
            if not is_g:
                cal = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED,
                                         C=1.0, class_weight=None)
                cal.fit(oof.reshape(-1, 1), inner_y)
                threshold_score = cal.predict_proba(oof.reshape(-1, 1))[:, 1]
            else:
                threshold_score = oof
            threshold = youden(inner_y, threshold_score)
            fitted = pipeline(params).fit(x.iloc[population], inner_y)
            score = fitted.predict_proba(x.iloc[test])[:, 1]
            if is_g:
                g_scores[name][test] = score
            else:
                p_scores[name][test] = cal.predict_proba(score.reshape(-1, 1))[:, 1]
            thresholds[name][test] = threshold
            threshold_rows.append({"fold": fold, "family": name, "threshold": threshold,
                                   "outer_training_n": len(population)})
        print(json.dumps({"completed_outer_fold": fold, "cumulative_XGB_fits": 24 * fold}), flush=True)
    for array in (*g_scores.values(), *p_scores.values(), *thresholds.values()):
        if not np.isfinite(array).all():
            raise RuntimeError("Incomplete held-out scores or thresholds")
    return folds, g_scores, p_scores, thresholds, threshold_rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_environment(cfg):
    versions = {key: metadata.version(key) for key in cfg["versions"]}
    if versions != cfg["versions"] or platform.python_version() != cfg["python_version"]:
        raise RuntimeError("Training requires the recorded Python and package versions; see README")
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("general-csv", "policy-csv", "mapped-csv", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-training", action="store_true", help="Explicitly allow 120 XGBoost fits")
    args = parser.parse_args()
    if not args.run_training:
        parser.error("--run-training is required; use test_appendix_c.py for checks without fitting")
    cfg = json.loads((ROOT / "locked_protocol.json").read_text(encoding="utf-8"))
    versions = check_environment(cfg)
    # Reserve a new directory; never overwrite frozen reference results or prior runs.
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    y, xg, xp, policy = read_inputs(args.general_csv, args.policy_csv, args.mapped_csv, cfg)
    args.output_dir.mkdir(parents=True)
    folds, gs, ps, thresholds, threshold_rows = fit_scores(y, xg, xp, policy, cfg)
    tables = summarize(y, folds, gs, ps, thresholds)
    tables["thresholds.csv"] = threshold_rows
    for name, rows in tables.items():
        write_csv(args.output_dir / name, rows)
    run = {"source_experiment": cfg["source_experiment"], "XGB_fits": 120,
           "python_version": platform.python_version(), "versions": versions,
           "N": len(y), "positive_n": int(y.sum()), "policy_n": int(policy.sum()),
           "outputs": list(tables), "limitations": cfg["limitations"]}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "XGB_fits": 120, "aggregate_tables": len(tables)}))


if __name__ == "__main__":
    main()
