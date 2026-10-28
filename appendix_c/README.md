# Current Appendix C: harmonized vanilla/tuned G/P code

This is a code-only release of the current Appendix C analysis. Empirical result tables and cohort counts are not bundled. It is separate from the original paper's retrospective FN-only audit in `src/Experiments/two_stage_evaluation.py`. The original audit, baseline, tuning, preprocessing and proxy-loss code remain in their existing locations.

The current analysis compares four fixed XGBoost combinations: `Gv_Pv`, `Gv_Pt`, `Gt_Pv` and `Gt_Pt`. `v` means vanilla and `t` means tuned. Each P model reassesses **all predicted negatives of its own G model**; every G predicted positive is retained. Standalone `Gv` and `Gt` are controls. Figure C1 contains `Gv`, `Gv_Pv`, `Gv_Pt`; Figure C2 contains `Gt`, `Gt_Pv`, `Gt_Pt`.

## Files

| File | Purpose |
| --- | --- |
| `run_2x2.py` | Portable restricted-data runner; fresh fits, local aggregate outputs only |
| `locked_protocol.json` | Sanitized frozen feature order, parameters, dtypes, folds and limitations |
| `verify_aggregates.py` | Standard-library arithmetic and per-fold budget checks on user-supplied local aggregates |
| `plot_confusion.py` | Figures C1/C2 from user-supplied local aggregate counts |
| `test_*.py` | Synthetic checks without XGBoost fitting or empirical result files |
| `provenance.json` | Source-code/protocol filenames and SHA-256 hashes without personal paths |
| `VALIDATION.md` | Completed checks and explicit unrun checks |
| `requirements*.txt` | Recorded training and separate rendering dependency versions |

## Fixed protocol

G uses 20 preselected features and P uses six. Their exact order and raw dtypes are in `locked_protocol.json`. Both variants within each family use numeric mean imputation, `StandardScaler` and `XGBClassifier`, with float64 model input. The frozen inputs contained finite, nonmissing numeric features. Features and tuned parameters were preselected.

Outer and inner splits use five-fold `StratifiedKFold(shuffle=True, random_state=42)`. General-file row order is preserved, and training rows stay in their original order. G is trained on the outer-training population; P is trained on all policy borrowers within that population. Each P model and calibrator is shared unchanged across both G arms.

G thresholds use raw inner-OOF probabilities. P uses logistic Platt calibration fitted to raw positive probabilities from policy-training inner OOF predictions, followed by its threshold on those calibrated training scores. Thresholds use the first finite threshold attaining the maximum Youden J from `roc_curve(drop_intermediate=False)`. Decisions use `>=`. Outer-test labels are used for evaluation, not for gating, calibration or native threshold selection.

For the matched comparison, each native two-stage decision mask stays unchanged. Within each outer-test fold, standalone tuned G ranks all held-out borrowers by descending score and flags exactly as many as that arm's final two-stage count. Ties use ascending original row position. The secondary comparison uses the arm's own G score and is labelled separately. This is a retrospective batch-ranking benchmark, not a prospective fixed-threshold policy.

**Table R1 matches final flagged counts per fold, not false-positive rates.** The runner selects `Gv_Pt` versus standalone tuned G from panel B for this table. It computes the corresponding counts from the authorised local inputs.

## Checks without borrower data

The aggregate-verifier tests use only Python's standard library and generate their own small toy tables:

```bash
python -B -m unittest discover -s appendix_c -p test_aggregates.py -v
```

Use the recorded training environment (Python 3.10.20; versions in `requirements.txt`) for all synthetic protocol checks:

```bash
python -B -m unittest discover -s appendix_c -p "test_*.py" -v
python -B appendix_c/run_2x2.py --help
```

These tests cover ID alignment, policy-feature replacement, target agreement, retention of G positives, negative gating, fold-level budget matching, deterministic ties, estimator parameters, float64 preprocessing and aggregate corruption detection. They do not fit XGBoost models or access borrower data.

## Restricted input schema and full rerun

Numerical reproduction requires authorised access to the KCB research dataset. Keep restricted data and outputs outside this repository.

| Input option | Required columns and alignment |
| --- | --- |
| `--general-csv` | One row per unique, nonmissing `KCB_DEID1_ENCRYPT`; binary `DLQ_ANY_YN`; all 20 G features in the protocol |
| `--policy-csv` | Unique policy IDs forming a subset of the general IDs; agreeing target labels; all six P features |
| `--mapped-csv` | Exactly the general ID set, with the six P features mapped for every borrower; row order may differ |

IDs are read as strings and used only for alignment. General-file row order defines the outer-fold input order and tie-break positions. The mapped P frame is aligned by ID; actual policy-row P values overwrite the corresponding mapped values. Required features must have the protocol's numeric int64/float64 dtypes and contain no missing or infinite values. Each outer-training general and policy population must support inner five-fold stratification. Exact numerical reproduction still requires the original cohort, values and row order; structural validation cannot establish dataset identity.

With the recorded Python/package versions, run from the repository root:

```bash
python -B appendix_c/run_2x2.py \
  --general-csv ../restricted_data/06_general_borrower_dataset_g_model_features.csv \
  --policy-csv ../restricted_data/06_policy_loan_borrower_dataset_p_model_features.csv \
  --mapped-csv ../restricted_data/06_all_borrowers_aligned_to_p_model_features.csv \
  --output-dir ../restricted_outputs/appendix_c_new_run \
  --run-training
```

The paths are portable examples. `--run-training` is explicit, and the output directory must be new. The runner requires the recorded Python/package versions, uses CPU XGBoost, performs no hyperparameter search, and refits all four families: 4 x 5 outer folds x (5 inner fits + 1 outer fit) = **120 XGBoost fits**. The recorded experiment fitted 90 new models and reused 30 independently checked Gt models. This implementation removes that private-cache dependency by refitting Gt as well. It writes only local aggregate CSVs, fold thresholds and version metadata; it does not save IDs, row-level predictions or fitted models.

The local outputs are `panel_A.csv` (six native configurations), `native_fold_results.csv` (method-fold counts), `panel_B.csv` (four primary and four separately labelled secondary matched comparisons), `matched_fold_budgets.csv`, `table_R1.csv`, `thresholds.csv` and `run_metadata.json`.

## Verify local aggregates and render C1/C2

The verifier uses only the standard library and does not fit models:

```bash
python -B appendix_c/verify_aggregates.py --results-dir ../restricted_outputs/appendix_c_new_run
```

It checks counts, metrics, FN recovery and FP arithmetic, all native fold sums, all matched fold budgets and R1 source-row selection. Cohort totals are inferred from the supplied Gv row and must agree across all methods. An optional `--provenance local_manifest.json` verifies local SHA-256 hashes using an `aggregate_sources` list of `{file, sha256}` records and a `derived_R1` record. No empirical hash manifest is bundled.

Use a separate Python 3.12.14 environment with `requirements-figures.txt` to render local aggregates:

```bash
python -B appendix_c/plot_confusion.py \
  --panel-a ../restricted_outputs/appendix_c_new_run/panel_A.csv \
  --output-dir ../figure_outputs/appendix_c
```

The output directory must be new. Each figure is 180 x 84 mm with 600 dpi PNG and vector PDF/SVG, plus a 150 dpi proof. Black text is at least 8 pt. Colors encode actual-class row proportions on a shared 0-1 scale; cell text contains raw counts and signed changes from the corresponding G baseline. Footers report FN recovery and additional FP per recovered FN. Captions explain the descriptive interpretation. Generated outputs stay local.

## Provenance and limits

The code is adapted from the frozen `kcb2_harmonized_2x2_20261025` experiment. `provenance.json` records hashes of its actual runner and original protocol. The published protocol removes personal paths, execution details and cohort counts. The plotting code is adapted from the inspected annotated-figure builder, preserving the current C1/C2 family mapping.

The portable runner has **not** been rerun on the full private dataset. Synthetic checks, source review and local aggregate arithmetic do not establish bitwise numerical equivalence after refitting cached Gt models. Exact reproduction depends on input values/order, software and numerical execution. The historical Appendix C implementation's original numerical/runtime provenance is not resolved by this release.

Feature selection and tuning are **not fully nested** within the outer folds. Native calibration and thresholds use outer-training data; the final-flag benchmark uses retrospective test-batch ranking. Fixed G/P feature sets differ; the factorial contrasts concern vanilla/tuned specifications within each family. Results are descriptive and do not establish significance, net benefit, formal fairness or deployment performance.

No empirical result tables, cohort counts, raw/cleaned borrower data, row-level predictions, models, private logs, credentials or manuscript drafts are included. Updating this GitHub branch does not establish that an anonymous review mirror has refreshed its contents.
