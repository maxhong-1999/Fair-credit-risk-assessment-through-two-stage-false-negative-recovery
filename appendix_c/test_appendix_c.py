"""Synthetic protocol checks. No estimator fitting or restricted data access."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


class AppendixContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "run_2x2.py"
        cls.assert_source_exists = path.exists()
        if path.exists():
            spec = importlib.util.spec_from_file_location("current_appendix", path)
            cls.runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.runner)

    def setUp(self):
        self.assertTrue(self.assert_source_exists, "Current harmonized runner is missing")
        self.m = self.runner

    def test_native_gate_retains_positives_even_when_p_score_is_low(self):
        gs = {"Gv": np.array([.9, .1, .3, .7]), "Gt": np.array([.1, .9, .7, .3])}
        ps = {"Pv": np.array([.1, .9, .4, .8]), "Pt": np.array([.8, .4, .9, .1])}
        thresholds = {k: np.full(4, .5) for k in ("Gv", "Gt", "Pv", "Pt")}
        d = self.m.native_decisions(gs, ps, thresholds)
        np.testing.assert_array_equal(d["Gv_Pv"], [True, True, False, True])
        np.testing.assert_array_equal(d["Gt_Pt"], [True, True, True, False])
        for arm in ("Gv_Pv", "Gv_Pt", "Gt_Pv", "Gt_Pt"):
            g = arm.split("_")[0]
            self.assertTrue(d[arm][d[g]].all())

    def test_budget_is_matched_in_each_fold_and_ties_use_row_order(self):
        folds = np.array([1, 2, 1, 2, 1, 2])
        score = np.array([.5, .2, .5, .9, .1, .2])
        native = np.array([False, True, False, True, True, False])
        chosen, budgets = self.m.matched_mask(folds, score, native)
        np.testing.assert_array_equal(chosen, [True, True, False, True, False, False])
        self.assertEqual([(b["two_stage_flags"], b["standalone_flags"]) for b in budgets], [(1, 1), (2, 2)])

    def test_zero_and_full_fold_budgets(self):
        folds = np.array([1, 1, 2, 2])
        native = np.array([False, False, True, True])
        chosen, _ = self.m.matched_mask(folds, np.ones(4), native)
        np.testing.assert_array_equal(chosen, native)

    def test_finite_youden_threshold_and_greater_equal_decisions(self):
        self.assertEqual(self.m.youden(np.array([0, 0, 1, 1]), np.array([.1, .2, .2, .9])), .9)
        y = np.array([0, 1, 1, 0])
        self.assertEqual(self.m.metrics(y, np.array([False, True, False, True]))["flagged"], 2)

    def test_input_alignment_overwrites_policy_features_and_rejects_disagreement(self):
        cfg = json.loads((ROOT / "locked_protocol.json").read_text(encoding="utf-8"))
        ids = ["a", "b", "c", "d"]
        def frame(features, dtypes, order):
            result = pd.DataFrame({self.m.ID: order})
            for i, key in enumerate(features):
                result[key] = pd.Series([i + 1] * len(order), dtype=dtypes[key])
            return result
        general = frame(cfg["G_features"], cfg["G_raw_column_dtypes"], ids)
        general[self.m.TARGET] = [0, 1, 0, 1]
        mapped = frame(cfg["P_features"], cfg["P_raw_column_dtypes"], ids[::-1])
        policy = frame(cfg["P_features"], cfg["P_raw_column_dtypes"], ["d", "b"])
        policy[self.m.TARGET] = [1, 1]
        policy[cfg["P_features"][0]] = [55, 66]
        with tempfile.TemporaryDirectory(prefix="synthetic-", dir=ROOT) as td:
            paths = [Path(td) / name for name in ("general.csv", "policy.csv", "mapped.csv")]
            for p, f in zip(paths, (general, policy, mapped)):
                f.to_csv(p, index=False)
            y, xg, xp, mask = self.m.read_inputs(*paths, cfg)
            self.assertEqual(list(xg.columns), cfg["G_features"])
            self.assertEqual(list(xp.columns), cfg["P_features"])
            np.testing.assert_array_equal(mask, [False, True, False, True])
            self.assertEqual(xp.iloc[1, 0], 66)
            self.assertEqual(xp.iloc[3, 0], 55)
            policy.loc[0, self.m.TARGET] = 0
            policy.to_csv(paths[1], index=False)
            with self.assertRaisesRegex(ValueError, "targets disagree"):
                self.m.read_inputs(*paths, cfg)

    def test_pipeline_parameters_and_float64_preprocessing_without_fit(self):
        cfg = json.loads((ROOT / "locked_protocol.json").read_text(encoding="utf-8"))
        for family in ("Gv", "Gt", "Pv", "Pt"):
            pipe = self.m.pipeline(cfg["params"][family])
            self.assertEqual([name for name, _ in pipe.steps], ["imputer", "scaler", "clf"])
            for key, value in cfg["params"][family].items():
                self.assertEqual(pipe.named_steps["clf"].get_params()[key], value)
            numeric = pd.DataFrame({"integer": [1, 2, 3], "floating": [.5, .7, .9]})
            self.assertEqual(pipe[:-1].fit_transform(numeric).dtype, np.float64)

    def test_summary_cross_checks_all_arms_and_r1_without_training(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        folds = np.array([1, 2, 1, 2, 1, 2])
        gs = {"Gv": np.array([.9, .8, .7, .6, .5, .4]),
              "Gt": np.array([.1, .9, .2, .8, .3, .7])}
        ps = {"Pv": np.array([.6, .8, .1, .2, .4, .9]),
              "Pt": np.array([.1, .2, .3, .4, .8, .9])}
        thresholds = {k: np.full(6, .5) for k in ("Gv", "Gt", "Pv", "Pt")}
        result = self.m.summarize(y, folds, gs, ps, thresholds)
        self.assertEqual(len(result["panel_A.csv"]), 6)
        self.assertEqual(len(result["panel_B.csv"]), 16)
        self.assertEqual(len(result["matched_fold_budgets.csv"]), 16)
        r1 = result["table_R1.csv"]
        self.assertEqual(len(r1), 2)
        self.assertEqual(r1[0]["flagged"], r1[1]["flagged"])
        self.assertEqual({r["method"] for r in r1}, {"two_stage", "standalone"})


if __name__ == "__main__":
    unittest.main()
