# Code-only release validation

Validation used existing local environments during preparation on 2026-10-28. No packages were installed and no XGBoost models were fitted.

## Completed

- Ten automated checks passed: seven synthetic protocol checks and three aggregate-verifier checks. Only the imputer/scaler are fitted to tiny synthetic arrays. The aggregate tests generate toy tables in temporary directories; no empirical results are bundled.
- Local reference-aggregate verification checked confusion-derived metrics, recovery/FP arithmetic, native fold sums, all four primary and four secondary matched comparisons, all 40 per-fold budgets and R1 row selection. The empirical tables used for this check remain local.
- Figures C1/C2 were rendered locally with Python 3.12.14, matplotlib 3.11.1 and NumPy 2.3.5. Both 180 x 84 mm figures produced 4,251 x 1,984 pixel PNGs at 600 dpi. All exported cell counts, signed changes and cell colors were checked against the local source aggregates and expected row proportions. PDF/SVG contained no raster images; PDF fonts were embedded and SVG text was outlined. Layout checks found no clipping or text overlap; text was black and at least 8 pt. The 150 dpi proofs were visually inspected for legibility and correct vanilla/tuned mapping. Generated empirical figures remain local.
- Syntax, CLI help and recorded training-environment checks passed. The publication scope excludes empirical CSVs and model-result counts. Original-paper code remains unchanged except the authorised obsolete Appendix-specific deletion and README updates.

The source experiment's saved-model verification is source evidence only; it is not verification of the portable refactor.

## Not run

- Full restricted-data training or numerical replay of the portable runner.
- New hyperparameter search, nested feature/tuning selection, significance tests or deployment evaluation.
- Any refresh or validation of an anonymous review mirror.

The repository had no CI workflow when this change was prepared. Local checks do not imply GitHub CI success; remote status must be checked against the exact published commit.
