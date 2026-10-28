"""Synthetic aggregate-verifier tests; no empirical results or model fitting."""
import csv
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def metric(tp, fp, tn, fn):
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, flagged=tp + fp,
                recall=tp / (tp + fn) if tp + fn else 0.,
                precision=tp / (tp + fp) if tp + fp else 0.,
                specificity=tn / (tn + fp) if tn + fp else 0.,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_results(directory):
    # Five identical toy folds: four cases per fold, one actual positive.
    counts = {"Gv": (0, 0, 3, 1), "Gt": (0, 1, 2, 1),
              "Gv_Pv": (1, 2, 1, 0), "Gv_Pt": (1, 1, 2, 0),
              "Gt_Pv": (1, 2, 1, 0), "Gt_Pt": (0, 2, 1, 1)}
    pooled = {name: metric(*(5 * n for n in values)) for name, values in counts.items()}
    a, b, budgets, folds = [], [], [], []
    for name, values in counts.items():
        row = {"method": name, **pooled[name]}
        if "_" in name:
            base = pooled[name.split("_")[0]]
            row.update(recovered_fn_from_own_G=row["tp"] - base["tp"],
                       additional_fp_from_own_G=row["fp"] - base["fp"],
                       G_negative_reassessed_n=20 - base["flagged"])
        a.append(row)
        folds.extend({"panel": "A", "method": name, "fold": fold, **metric(*values)}
                     for fold in range(1, 6))
        if "_" not in name:
            continue
        for comparator in ("tuned_G_primary", "same_first_stage_G_secondary"):
            b.extend({"arm": name, "comparator": comparator, "method": side, **pooled[name]}
                     for side in ("two_stage", "standalone"))
            budgets.extend({"arm": name, "comparator": comparator, "fold": fold,
                            "two_stage_flags": values[0] + values[1],
                            "standalone_flags": values[0] + values[1]} for fold in range(1, 6))
    tables = {"panel_A.csv": a, "panel_B.csv": b, "native_fold_results.csv": folds,
              "matched_fold_budgets.csv": budgets,
              "table_R1.csv": [r for r in b if r["arm"] == "Gv_Pt" and r["comparator"] == "tuned_G_primary"]}
    for name, rows in tables.items():
        write_table(directory / name, rows)
    def record(name):
        return {"file": name, "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()}
    return {"aggregate_sources": [record(name) for name in tables if name != "table_R1.csv"],
            "derived_R1": record("table_R1.csv")}


class AggregateVerification(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("aggregate_verifier", ROOT / "verify_aggregates.py")
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_synthetic_arithmetic_and_hashes(self):
        with tempfile.TemporaryDirectory(prefix="aggregate-test-", dir=ROOT) as td:
            directory = Path(td)
            provenance = synthetic_results(directory)
            result = self.m.verify_results(directory, provenance=provenance)
            self.assertEqual(result["native_methods"], 6)
            self.assertEqual(result["matched_fold_rows"], 40)
            self.assertEqual(result["R1_final_flags"], 10)
            self.assertEqual(result["R1_two_stage_minus_tuned_G_tp"], 0)

    def test_changed_confusion_count_is_rejected_without_hash_check(self):
        with tempfile.TemporaryDirectory(prefix="aggregate-test-", dir=ROOT) as td:
            directory = Path(td)
            synthetic_results(directory)
            path = directory / "panel_A.csv"
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            rows[0]["tp"] = str(int(rows[0]["tp"]) + 1)
            write_table(path, rows)
            with self.assertRaises(ValueError):
                self.m.verify_results(directory)

    def test_fractional_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            self.m.count("1.5")


if __name__ == "__main__":
    unittest.main()
