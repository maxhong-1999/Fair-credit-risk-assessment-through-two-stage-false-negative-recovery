"""Verify user-supplied aggregate tables using only the Python standard library."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("Gv_Pv", "Gv_Pt", "Gt_Pv", "Gt_Pt")
COMPARATORS = ("tuned_G_primary", "same_first_stage_G_secondary")
COUNTS = ("tp", "fp", "tn", "fn", "flagged")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid aggregate count") from error
    require(number.is_finite() and number >= 0 and number == number.to_integral_value(),
            "Aggregate counts must be nonnegative integers")
    return int(number)


def read_rows(directory, name):
    with (directory / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def indexed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[key] for key in fields)
        require(key not in result, "Duplicate aggregate result key")
        result[key] = row
    return result


def check_metrics(row, n, positives):
    values = {key: count(row[key]) for key in COUNTS}
    tp, fp, tn, fn = (values[key] for key in ("tp", "fp", "tn", "fn"))
    require(tp + fn == positives and tp + fp + tn + fn == n, "Cohort totals disagree")
    require(tp + fp == values["flagged"], "Final flagged count disagrees")
    expected = {"recall": tp / (tp + fn) if tp + fn else 0.0,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "specificity": tn / (tn + fp) if tn + fp else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    for key, value in expected.items():
        require(math.isclose(float(row[key]), value, rel_tol=1e-12, abs_tol=1e-14),
                f"{key} disagrees with confusion counts")
    return values


def verify_results(directory, cfg=None, provenance=None):
    directory = Path(directory)
    methods = ("Gv", "Gt", *ARMS)
    a = indexed(read_rows(directory, "panel_A.csv"), ("method",))
    require(set(a) == {(name,) for name in methods}, "Panel A must contain six native methods")
    if cfg is None:
        base = a[("Gv",)]
        n = sum(count(base[key]) for key in ("tp", "fp", "tn", "fn"))
        positives = count(base["tp"]) + count(base["fn"])
    else:
        n, positives = cfg["N"], cfg["positive_n"]
    counts_a = {name: check_metrics(a[(name,)], n, positives) for name in methods}
    for arm in ARMS:
        baseline = counts_a[arm.split("_")[0]]
        values, row = counts_a[arm], a[(arm,)]
        recovered = count(row["recovered_fn_from_own_G"])
        added = count(row["additional_fp_from_own_G"])
        require(recovered == values["tp"] - baseline["tp"] == baseline["fn"] - values["fn"],
                "FN recovery does not reconcile")
        require(added == values["fp"] - baseline["fp"] == baseline["tn"] - values["tn"],
                "Additional FP count does not reconcile")
        require(count(row["G_negative_reassessed_n"]) == n - baseline["flagged"],
                "Full G-negative reassessment scope disagrees")
    folds = indexed(read_rows(directory, "native_fold_results.csv"), ("method", "fold"))
    require(set(folds) == {(method, str(fold)) for method in methods for fold in range(1, 6)},
            "Native fold results must contain all 30 method-fold pairs")
    for method in methods:
        for fold in range(1, 6):
            row = folds[(method, str(fold))]
            base = folds[("Gv", str(fold))]
            fold_n = sum(count(base[key]) for key in ("tp", "fp", "tn", "fn"))
            fold_pos = count(base["tp"]) + count(base["fn"])
            require(row["panel"] == "A", "Unexpected native panel")
            check_metrics(row, fold_n, fold_pos)
        for key in COUNTS:
            require(sum(count(folds[(method, str(fold))][key]) for fold in range(1, 6))
                    == counts_a[method][key], "Native fold counts do not sum to pooled result")
    b = indexed(read_rows(directory, "panel_B.csv"), ("arm", "comparator", "method"))
    expected_b = {(arm, comparator, method) for arm in ARMS for comparator in COMPARATORS
                  for method in ("two_stage", "standalone")}
    require(set(b) == expected_b, "Panel B must contain all 16 comparisons")
    for (arm, comparator, method), row in b.items():
        values = check_metrics(row, n, positives)
        require(values["flagged"] == counts_a[arm]["flagged"], "Panel B final flags disagree")
        if method == "two_stage":
            require(values == counts_a[arm], "Matched analysis changed native two-stage decisions")
    budgets = indexed(read_rows(directory, "matched_fold_budgets.csv"), ("arm", "comparator", "fold"))
    require(set(budgets) == {(arm, comparator, str(fold)) for arm in ARMS
                            for comparator in COMPARATORS for fold in range(1, 6)},
            "All 40 per-fold budget comparisons are required")
    for (arm, _, fold), row in budgets.items():
        require(count(row["two_stage_flags"]) == count(row["standalone_flags"])
                == count(folds[(arm, fold)]["flagged"]), "A per-fold budget does not match")
    r1 = indexed(read_rows(directory, "table_R1.csv"), ("arm", "comparator", "method"))
    require(set(r1) == {("Gv_Pt", "tuned_G_primary", side) for side in ("two_stage", "standalone")},
            "R1 must compare Gv_Pt with tuned G at matched final flags")
    for key, row in r1.items():
        require(row == b[key], "R1 differs from its Panel B source rows")
    if provenance is not None:
        hashes = provenance["aggregate_sources"] + [provenance["derived_R1"]]
        for record in hashes:
            require(hashlib.sha256((directory / record["file"]).read_bytes()).hexdigest()
                    == record["sha256"], "Aggregate source SHA-256 mismatch")
    two, single = (r1[("Gv_Pt", "tuned_G_primary", side)] for side in ("two_stage", "standalone"))
    return {"status": "PASS", "native_methods": len(a), "native_fold_rows": len(folds),
            "matched_comparison_rows": len(b), "matched_fold_rows": len(budgets),
            "source_hashes_checked": provenance is not None,
            "R1_final_flags": count(two["flagged"]),
            "R1_two_stage_minus_tuned_G_tp": count(two["tp"]) - count(single["tp"]),
            "scope": "Aggregate arithmetic, source hashes and fold budgets; no model replay"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, help="Optional local aggregate hash manifest")
    args = parser.parse_args()
    provenance = (json.loads(args.provenance.read_text(encoding="utf-8"))
                  if args.provenance is not None else None)
    print(json.dumps(verify_results(args.results_dir, provenance=provenance), indent=2))


if __name__ == "__main__":
    main()
