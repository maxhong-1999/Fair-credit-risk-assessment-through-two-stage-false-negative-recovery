from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLAGS_PATH = REPO_ROOT / "outputs" / "two_stage" / "06_two_stage_decision_flags.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "proxy_loss"

LGD_COMMON_ASSUMPTIONS = (0.20, 0.35, 0.45, 0.50, 0.60, 0.75)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --paths points to a YAML file.") from exc
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
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


def _find_flags_path(config: dict[str, Any], config_path: Path) -> Path:
    for raw in _iter_strings(config):
        if raw.endswith("06_two_stage_decision_flags.csv"):
            return _resolve_path(raw, config_path)
    model_outputs_dir = config.get("model_outputs_dir")
    if model_outputs_dir:
        return _resolve_path(f"{model_outputs_dir}/06_two_stage_decision_flags.csv", config_path)
    return DEFAULT_FLAGS_PATH


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    return series.astype(str).str.strip().str.upper().isin({"TRUE", "T", "Y", "YES", "1"})


def _as_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)
    return series.astype(str).str.strip().str.upper().map({"Y": 1, "N": 0, "1": 1, "0": 0}).fillna(0).astype(int)


def _ead(flags: pd.DataFrame) -> pd.Series:
    if "EAD_PROXY" in flags.columns:
        return pd.to_numeric(flags["EAD_PROXY"], errors="coerce").fillna(0)
    return pd.Series(1.0, index=flags.index)


def expected_loss(flags: pd.DataFrame, approve_col: str, lgd: float, scope_mask: pd.Series | None = None) -> dict[str, Any]:
    if scope_mask is None:
        scope_mask = pd.Series(True, index=flags.index)
    y = _as_target(flags["DLQ_ANY_YN"])
    approve = _as_bool(flags[approve_col])
    ead = _ead(flags)

    loss_mask = scope_mask & approve & (y == 1)
    approved_mask = scope_mask & approve
    return {
        "approve_col": approve_col,
        "lgd": float(lgd),
        "scope_n": int(scope_mask.sum()),
        "approved_n": int(approved_mask.sum()),
        "approved_bad_n": int(loss_mask.sum()),
        "ead_approved": float(ead[approved_mask].sum()),
        "ead_approved_bad": float(ead[loss_mask].sum()),
        "proxy_expected_loss": float((ead[loss_mask] * lgd).sum()),
    }


def build_proxy_loss_tables(flags: pd.DataFrame, base_lgd: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"DLQ_ANY_YN", "APPROVE_S1", "APPROVE_S2_FINAL", "REVIEW_FLAG"}
    missing = sorted(required.difference(flags.columns))
    if missing:
        raise ValueError(f"Missing required decision-flag columns: {missing}")

    review_mask = _as_bool(flags["REVIEW_FLAG"])
    all_mask = pd.Series(True, index=flags.index)

    summary_rows = []
    for scope_name, scope_mask in {
        "all_borrowers": all_mask,
        "stage1_fn_review_target": review_mask,
    }.items():
        s1 = expected_loss(flags, "APPROVE_S1", base_lgd, scope_mask)
        s2 = expected_loss(flags, "APPROVE_S2_FINAL", base_lgd, scope_mask)
        summary_rows.append({**s1, "scope": scope_name, "stage": "stage1"})
        summary_rows.append({**s2, "scope": scope_name, "stage": "stage2"})
        summary_rows.append(
            {
                "scope": scope_name,
                "stage": "difference_stage1_minus_stage2",
                "lgd": float(base_lgd),
                "scope_n": int(scope_mask.sum()),
                "approved_n": int(s1["approved_n"] - s2["approved_n"]),
                "approved_bad_n": int(s1["approved_bad_n"] - s2["approved_bad_n"]),
                "ead_approved": float(s1["ead_approved"] - s2["ead_approved"]),
                "ead_approved_bad": float(s1["ead_approved_bad"] - s2["ead_approved_bad"]),
                "proxy_expected_loss": float(s1["proxy_expected_loss"] - s2["proxy_expected_loss"]),
            }
        )

    sensitivity_rows = []
    for lgd in LGD_COMMON_ASSUMPTIONS:
        s1 = expected_loss(flags, "APPROVE_S1", lgd, review_mask)
        s2 = expected_loss(flags, "APPROVE_S2_FINAL", lgd, review_mask)
        sensitivity_rows.append(
            {
                "scope": "stage1_fn_review_target",
                "lgd": float(lgd),
                "stage1_proxy_expected_loss": s1["proxy_expected_loss"],
                "stage2_proxy_expected_loss": s2["proxy_expected_loss"],
                "avoided_proxy_expected_loss": float(s1["proxy_expected_loss"] - s2["proxy_expected_loss"]),
                "stage1_approved_bad_n": s1["approved_bad_n"],
                "stage2_approved_bad_n": s2["approved_bad_n"],
                "recovered_bad_n": int(s1["approved_bad_n"] - s2["approved_bad_n"]),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(sensitivity_rows)


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.paths).resolve()
    config = _load_yaml(config_path)
    flags_path = Path(args.flags).resolve() if args.flags else _find_flags_path(config, config_path)
    if not flags_path.exists():
        raise FileNotFoundError(f"Decision flags not found: {flags_path}")

    flags = pd.read_csv(flags_path)
    summary, sensitivity = build_proxy_loss_tables(flags, base_lgd=args.base_lgd)

    configured_output_dir = config.get("audit_outputs_dir")
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif configured_output_dir:
        output_dir = _resolve_path(str(configured_output_dir), config_path)
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "07_proxy_loss_summary.csv", index=False)
    sensitivity.to_csv(output_dir / "07_proxy_loss_lgd_sensitivity_common_assumptions.csv", index=False)
    with (output_dir / "07_proxy_loss_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "flags_path": str(flags_path),
                "base_lgd": args.base_lgd,
                "lgd_common_assumptions": list(LGD_COMMON_ASSUMPTIONS),
                "ead_source": "EAD_PROXY if present, otherwise unit exposure",
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute proxy expected-loss summaries from two-stage decision flags.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--flags", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-lgd", type=float, default=0.45)
    args = parser.parse_args()
    args.paths = args.config
    return args


if __name__ == "__main__":
    run(parse_args())
