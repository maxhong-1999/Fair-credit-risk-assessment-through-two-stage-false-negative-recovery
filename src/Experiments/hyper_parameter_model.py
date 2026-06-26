from __future__ import annotations

import argparse
import json
import time
import warnings
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "private" / "processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "model_training"

TARGET_COL = "DLQ_ANY_YN"
ID_COL = "KCB_DEID1_ENCRYPT"
RANDOM_STATE = 42

def _normalise_alias(value: str) -> str:
    return value.lower().replace(" ", "").replace("-", "").replace("_", "")


MODEL_DATASETS = {
    "G-model": {
        "file": "06_general_borrower_dataset_g_model_features.csv",
        "family": "G",
        "paper_model": "G-model",
        "borrower_dataset": "general borrower dataset",
        "legacy_input_id": "C_C",
        "description": "general borrower dataset using the G-model feature set",
    },
    "P-model": {
        "file": "06_policy_loan_borrower_dataset_p_model_features.csv",
        "family": "P",
        "paper_model": "P-model",
        "borrower_dataset": "policy loan borrower dataset",
        "legacy_input_id": "T_T",
        "description": "policy loan borrower dataset using the P-model feature set",
    },
}

DATASET_ALIAS_GROUPS = {
    "G-model": {"G_model", "G-model", "G", "general", "general_borrower", "C_C"},
    "P-model": {"P_model", "P-model", "P", "policy", "policy_loan", "policy_loan_borrower", "T_T"},
}
DATASET_ALIASES = {
    _normalise_alias(alias): canonical
    for canonical, aliases in DATASET_ALIAS_GROUPS.items()
    for alias in aliases
}

MODEL_ALIAS_GROUPS = {
    "LogisticRegression": {"LogisticRegression", "logistic_regression", "logistic", "LR"},
    "RandomForest": {"RandomForest", "random_forest", "RF"},
    "XGBoost": {"XGBoost", "xgb", "XGB"},
    "1D-CNN": {"1D-CNN", "1DCNN", "CNN"},
    "TabNet": {"TabNet"},
    "Transformer": {"Transformer"},
}
MODEL_ALIASES = {
    _normalise_alias(alias): canonical
    for canonical, aliases in MODEL_ALIAS_GROUPS.items()
    for alias in aliases
}

SUPPLEMENTARY_SEARCH_SPACES: dict[str, dict[str, dict[str, list[Any]]]] = {
    "G": {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["liblinear", "lbfgs", "saga"],
            "class_weight": [None, "balanced"],
            "max_iter": [1000, 2000, 5000],
            "tol": [1e-4, 1e-5],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.8],
            "class_weight": [None, "balanced"],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 6, 9, 12],
            "learning_rate": [0.01, 0.1, 0.2, 0.3],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 1.0],
            "reg_lambda": [0, 0.1, 1.0],
        },
        "1D-CNN": {
            "filters": [32, 64, 128],
            "kernel_size": [3, 5, 7],
            "dropout": [0.2, 0.3, 0.4, 0.5],
            "dense_units": [64, 128, 256],
            "batch_size": [1024, 2048, 4096],
            "learning_rate": [1e-4, 3e-4, 1e-3],
        },
        "TabNet": {
            "n_d": [32, 64, 128],
            "n_a": [32, 64, 128],
            "n_steps": [3, 5, 7],
            "gamma": [1.0, 1.5, 2.0],
            "lambda_sparse": [1e-4, 1e-3, 1e-2],
            "learning_rate": [0.01, 0.02, 0.05],
            "batch_size": [1024, 2048, 4096],
        },
        "Transformer": {
            "d_model": [64, 128, 256],
            "nhead": [4, 8, 16],
            "num_layers": [2, 4, 6],
            "dropout": [0.1, 0.2, 0.3],
            "dense_units": [64, 128, 256],
            "batch_size": [1024, 2048, 4096],
            "learning_rate": [0.001, 0.01, 0.1],
        },
    },
    "P": {
        "LogisticRegression": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "solver": ["liblinear", "lbfgs"],
            "class_weight": [None, "balanced", {0: 1, 1: 3}],
            "tol": [1e-4, 1e-5],
        },
        "RandomForest": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": ["sqrt", "log2", 0.5, 0.8],
            "class_weight": [None, "balanced"],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [2, 4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2, 0.3],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0, 0.01, 0.1, 1.0],
        },
        "1D-CNN": {
            "filters": [16, 32, 64],
            "kernel_size": [2, 3, 4],
            "dropout": [0.1, 0.2, 0.3, 0.4],
            "dense_units": [32, 64, 128],
            "batch_size": [1024, 2048, 4096],
            "learning_rate": [1e-4, 3e-4, 1e-3],
        },
        "TabNet": {
            "n_d": [16, 32, 64],
            "n_a": [16, 32, 64],
            "n_steps": [2, 3, 5],
            "gamma": [1.0, 1.5, 2.0],
            "lambda_sparse": [1e-3, 1e-2, 1e-1],
            "learning_rate": [0.01, 0.02, 0.05],
            "batch_size": [1024, 2048, 4096],
        },
        "Transformer": {
            "d_model": [32, 64, 128],
            "nhead": [2, 4, 8],
            "num_layers": [1, 2, 4],
            "dropout": [0.1, 0.2, 0.3],
            "dense_units": [32, 64, 128],
            "batch_size": [1024, 2048, 4096],
            "learning_rate": [0.001, 0.01, 0.1],
        },
    },
}

VANILLA_DEEP_DEFAULTS = {
    "1D-CNN": {"filters": 64, "kernel_size": 3, "dropout": 0.0, "dense_units": 64, "batch_size": 1024, "learning_rate": 1e-3},
    "TabNet": {"batch_size": 1024},
    "Transformer": {"d_model": 32, "nhead": 4, "num_layers": 2, "dropout": 0.0, "dense_units": 64, "batch_size": 1024, "learning_rate": 1e-3},
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
    candidates = [(config_path.parent / path).resolve(), (REPO_ROOT / path).resolve(), (Path.cwd() / path).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _find_path(config: dict[str, Any], config_path: Path, suffix: str, default: Path) -> Path:
    for raw in _iter_strings(config):
        if raw.endswith(suffix):
            return _resolve_path(raw, config_path)
    return default


def _canonical_model(name: str) -> str:
    key = _normalise_alias(name)
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_ALIASES[key]


def _canonical_dataset(name: str) -> str:
    key = _normalise_alias(name)
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown model dataset: {name}. Use G-model or P-model; legacy C_C/T_T aliases are accepted.")
    return DATASET_ALIASES[key]


def _model_family(dataset_name: str) -> str:
    return MODEL_DATASETS[_canonical_dataset(dataset_name)]["family"]


def _space(dataset_name: str, model_name: str) -> dict[str, list[Any]]:
    return SUPPLEMENTARY_SEARCH_SPACES[_model_family(dataset_name)][model_name]


def _binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.fillna(0).astype(int)
    mapped = series.astype(str).str.strip().str.upper().map({"Y": 1, "N": 0, "YES": 1, "NO": 0, "TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique()[:10])
        raise ValueError(f"Cannot map target values to binary labels: {bad}")
    return mapped.astype(int)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    x = df.drop(columns=[c for c in [TARGET_COL, ID_COL] if c in df.columns])
    cat_cols = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in x.columns if c not in cat_cols]
    num_tr = Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler(with_mean=True, with_std=True))])
    cat_tr = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _one_hot_encoder())])
    return ColumnTransformer([("num", num_tr, num_cols), ("cat", cat_tr, cat_cols)], remainder="drop")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    y = _binary_target(df[TARGET_COL])
    x = df.drop(columns=[TARGET_COL])
    if ID_COL in x.columns:
        x = x.drop(columns=[ID_COL])
    return x, y


def dense_array(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, score)
    idx = int(np.argmax(tpr - fpr))
    threshold = thresholds[idx] if idx < len(thresholds) else 0.5
    if np.isinf(threshold) or np.isnan(threshold):
        threshold = 0.5
    return float(threshold)


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def metrics_row(y_true: np.ndarray, score: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    if threshold is None:
        threshold = youden_threshold(y_true, score)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "ks": ks_statistic(y_true, score),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold_youden": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _product_size(space: dict[str, list[Any]]) -> int:
    return int(reduce(mul, (len(v) for v in space.values()), 1))


def base_estimator(model_name: str, seed: int):
    if model_name == "LogisticRegression":
        return LogisticRegression()
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1)
    if model_name == "XGBoost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed.") from exc
        return xgb.XGBClassifier(random_state=seed, n_jobs=-1, objective="binary:logistic", eval_metric="logloss")
    raise ValueError(f"Not a sklearn-pipeline model: {model_name}")


def fit_predict_sklearn(df: pd.DataFrame, dataset_name: str, model_name: str, mode: str, n_splits: int, seed: int, classical_iter: int, inner_cv: int):
    x, y = split_xy(df)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(df), dtype=float)
    fold_rows = []
    best_params_by_fold = []
    start = time.time()
    for fold, (tr, va) in enumerate(splitter.split(x, y), start=1):
        pipe = Pipeline([("pre", build_preprocessor(df)), ("clf", base_estimator(model_name, seed))])
        if mode == "supplementary_search":
            space = {f"clf__{k}": v for k, v in _space(dataset_name, model_name).items()}
            search = RandomizedSearchCV(
                pipe,
                param_distributions=space,
                n_iter=min(classical_iter, _product_size(_space(dataset_name, model_name))),
                scoring="roc_auc",
                cv=StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed),
                random_state=seed,
                n_jobs=-1,
                refit=True,
            )
            estimator = search.fit(x.iloc[tr], y.iloc[tr])
            best_params_by_fold.append(estimator.best_params_)
        else:
            estimator = pipe.fit(x.iloc[tr], y.iloc[tr])
            best_params_by_fold.append({})
        score = estimator.predict_proba(x.iloc[va])[:, 1]
        oof[va] = score
        row = metrics_row(y.iloc[va].to_numpy(), score)
        row.update({"fold": fold, "selected_params": json.dumps(best_params_by_fold[-1], default=str, sort_keys=True)})
        fold_rows.append(row)
    overall = metrics_row(y.to_numpy(), oof)
    overall.update({"runtime_seconds": time.time() - start})
    return oof, overall, fold_rows, best_params_by_fold


def build_1dcnn_model(input_shape: tuple[int, int], params: dict[str, Any]):
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, GlobalAveragePooling1D, MaxPooling1D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    filters = int(params["filters"])
    kernel_size = int(params["kernel_size"])
    dropout = float(params.get("dropout", 0.0))
    cur = input_shape[0]
    model = Sequential()
    model.add(Conv1D(filters, kernel_size, activation="relu", input_shape=input_shape, padding="same"))
    model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    if cur >= 8:
        model.add(MaxPooling1D(2))
        cur //= 2
    model.add(Conv1D(filters * 2, kernel_size, activation="relu", padding="same"))
    model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    if cur >= 4:
        model.add(MaxPooling1D(2))
        cur //= 2
    model.add(Conv1D(filters * 4, kernel_size, activation="relu", padding="same"))
    model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    if cur >= 2:
        model.add(MaxPooling1D(2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(int(params["dense_units"]), activation="relu"))
    model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=float(params["learning_rate"])), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def build_transformer_model(input_shape: tuple[int, int], params: dict[str, Any]):
    import tensorflow as tf
    from tensorflow.keras.layers import Add, Dense, Dropout, GlobalAveragePooling1D, Input, LayerNormalization, MultiHeadAttention
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    d_model = int(params["d_model"])
    nhead = int(params["nhead"])
    dropout = float(params.get("dropout", 0.0))
    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs)
    x = LayerNormalization()(x)
    for _ in range(int(params["num_layers"])):
        attn_out = MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead, dropout=dropout)(x, x)
        attn_out = Dense(d_model)(attn_out)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)
        ffn = Dense(d_model * 2, activation="relu")(x)
        if dropout > 0:
            ffn = Dropout(dropout)(ffn)
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(int(params["dense_units"]), activation="relu")(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=float(params["learning_rate"])), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def fit_keras_once(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, params: dict[str, Any], epochs: int, early_stop: bool, patience: int):
    import tensorflow as tf

    builder = build_1dcnn_model if model_name == "1D-CNN" else build_transformer_model
    model = builder((x_train.shape[1], 1), params)
    callbacks = []
    if early_stop:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience, restore_best_weights=True, verbose=0))
    model.fit(
        x_train.reshape(x_train.shape[0], x_train.shape[1], 1),
        y_train,
        validation_data=(x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1), y_valid),
        epochs=epochs,
        batch_size=int(params["batch_size"]),
        callbacks=callbacks,
        verbose=0,
    )
    score = model.predict(x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1), batch_size=4096, verbose=0).reshape(-1)
    tf.keras.backend.clear_session()
    return score


def fit_predict_keras(df: pd.DataFrame, dataset_name: str, model_name: str, mode: str, n_splits: int, seed: int, deep_iter: int, epochs: int, patience: int):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("tensorflow is not installed.") from exc

    x, y = split_xy(df)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(df), dtype=float)
    fold_rows = []
    best_params_by_fold = []
    start = time.time()
    for fold, (tr, va) in enumerate(splitter.split(x, y), start=1):
        tf.keras.utils.set_random_seed(seed + fold)
        pre = build_preprocessor(df)
        pre.fit(x.iloc[tr])
        x_tr_full = dense_array(pre.transform(x.iloc[tr]))
        y_tr_full = y.iloc[tr].to_numpy()
        x_va = dense_array(pre.transform(x.iloc[va]))
        y_va = y.iloc[va].to_numpy()
        if mode == "supplementary_search":
            candidates = list(ParameterSampler(_space(dataset_name, model_name), n_iter=deep_iter, random_state=seed + fold))
            idx_train, idx_search = train_test_split(np.arange(len(y_tr_full)), test_size=0.2, stratify=y_tr_full, random_state=seed + fold)
            best_auc = -np.inf
            best_params = candidates[0]
            for params in candidates:
                score = fit_keras_once(model_name, x_tr_full[idx_train], y_tr_full[idx_train], x_tr_full[idx_search], y_tr_full[idx_search], params, epochs, True, patience)
                auc = roc_auc_score(y_tr_full[idx_search], score)
                if auc > best_auc:
                    best_auc = auc
                    best_params = params
            params = best_params
            early_stop = True
        else:
            params = VANILLA_DEEP_DEFAULTS[model_name]
            early_stop = False
        score = fit_keras_once(model_name, x_tr_full, y_tr_full, x_va, y_va, params, epochs, early_stop, patience)
        oof[va] = score
        best_params_by_fold.append(params if mode == "supplementary_search" else {})
        row = metrics_row(y_va, score)
        row.update({"fold": fold, "selected_params": json.dumps(best_params_by_fold[-1], default=str, sort_keys=True)})
        fold_rows.append(row)
    overall = metrics_row(y.to_numpy(), oof)
    overall.update({"runtime_seconds": time.time() - start})
    return oof, overall, fold_rows, best_params_by_fold


def fit_tabnet_once(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, params: dict[str, Any], early_stop: bool, patience: int):
    from pytorch_tabnet.tab_model import TabNetClassifier

    if "n_d" in params:
        model = TabNetClassifier(
            n_d=int(params["n_d"]),
            n_a=int(params["n_a"]),
            n_steps=int(params["n_steps"]),
            gamma=float(params["gamma"]),
            lambda_sparse=float(params["lambda_sparse"]),
            seed=RANDOM_STATE,
            verbose=0,
            optimizer_params={"lr": float(params["learning_rate"])},
        )
    else:
        model = TabNetClassifier(seed=RANDOM_STATE, verbose=0)
    fit_kwargs = {
        "X_train": x_train,
        "y_train": y_train,
        "max_epochs": 50,
        "batch_size": int(params.get("batch_size", 1024)),
        "virtual_batch_size": max(64, min(256, int(params.get("batch_size", 1024)) // 8)),
        "num_workers": 0,
        "drop_last": False,
    }
    if early_stop:
        fit_kwargs.update({"eval_set": [(x_valid, y_valid)], "eval_metric": ["auc"], "patience": patience})
    model.fit(**fit_kwargs)
    return model.predict_proba(x_valid)[:, 1]


def fit_predict_tabnet(df: pd.DataFrame, dataset_name: str, mode: str, n_splits: int, seed: int, deep_iter: int, patience: int):
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pytorch-tabnet is not installed.") from exc

    x, y = split_xy(df)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(df), dtype=float)
    fold_rows = []
    best_params_by_fold = []
    start = time.time()
    for fold, (tr, va) in enumerate(splitter.split(x, y), start=1):
        pre = build_preprocessor(df)
        pre.fit(x.iloc[tr])
        x_tr_full = dense_array(pre.transform(x.iloc[tr]))
        y_tr_full = y.iloc[tr].to_numpy()
        x_va = dense_array(pre.transform(x.iloc[va]))
        y_va = y.iloc[va].to_numpy()
        if mode == "supplementary_search":
            candidates = list(ParameterSampler(_space(dataset_name, "TabNet"), n_iter=deep_iter, random_state=seed + fold))
            idx_train, idx_search = train_test_split(np.arange(len(y_tr_full)), test_size=0.2, stratify=y_tr_full, random_state=seed + fold)
            best_auc = -np.inf
            best_params = candidates[0]
            for params in candidates:
                score = fit_tabnet_once(x_tr_full[idx_train], y_tr_full[idx_train], x_tr_full[idx_search], y_tr_full[idx_search], params, True, patience)
                auc = roc_auc_score(y_tr_full[idx_search], score)
                if auc > best_auc:
                    best_auc = auc
                    best_params = params
            params = best_params
            early_stop = True
        else:
            params = VANILLA_DEEP_DEFAULTS["TabNet"]
            early_stop = False
        score = fit_tabnet_once(x_tr_full, y_tr_full, x_va, y_va, params, early_stop, patience)
        oof[va] = score
        best_params_by_fold.append(params if mode == "supplementary_search" else {})
        row = metrics_row(y_va, score)
        row.update({"fold": fold, "selected_params": json.dumps(best_params_by_fold[-1], default=str, sort_keys=True)})
        fold_rows.append(row)
    overall = metrics_row(y.to_numpy(), oof)
    overall.update({"runtime_seconds": time.time() - start})
    return oof, overall, fold_rows, best_params_by_fold


def fit_predict_model(df: pd.DataFrame, dataset_name: str, model_name: str, mode: str, args: argparse.Namespace):
    if model_name in {"LogisticRegression", "RandomForest", "XGBoost"}:
        return fit_predict_sklearn(df, dataset_name, model_name, mode, args.n_splits, args.seed, args.classical_search_iter, args.inner_cv)
    if model_name in {"1D-CNN", "Transformer"}:
        return fit_predict_keras(df, dataset_name, model_name, mode, args.n_splits, args.seed, args.deep_search_iter, args.deep_epochs, args.deep_patience)
    if model_name == "TabNet":
        return fit_predict_tabnet(df, dataset_name, mode, args.n_splits, args.seed, args.deep_search_iter, args.deep_patience)
    raise ValueError(f"Unknown model: {model_name}")


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.paths).resolve()
    config = _load_yaml(config_path)
    configured_output_dir = config.get("model_outputs_dir")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (_resolve_path(str(configured_output_dir), config_path) if configured_output_dir else DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_datasets = [_canonical_dataset(name) for name in args.datasets]
    requested_models = [_canonical_model(name) for name in args.models]
    selected_features_dir = config.get("selected_features_dir")
    analysis_mode = f"{args.mode}_model_comparison"

    all_metrics = []
    all_fold_metrics = []
    all_oof = []
    manifest = []

    for dataset_name in requested_datasets:
        dataset_info = MODEL_DATASETS[dataset_name]
        suffix = dataset_info["file"]
        default_input = _resolve_path(f"{selected_features_dir}/{suffix}", config_path) if selected_features_dir else DEFAULT_DATA_DIR / suffix
        dataset_path = _find_path(config, config_path, suffix, default_input)
        if not dataset_path.exists():
            manifest.append({"paper_model": dataset_info["paper_model"], "borrower_dataset": dataset_info["borrower_dataset"], "legacy_input_id": dataset_info["legacy_input_id"], "input_file": suffix, "path": str(dataset_path), "status": "missing_input"})
            continue

        df = pd.read_csv(dataset_path)
        for model_name in requested_models:
            try:
                oof, overall, fold_rows, selected_params = fit_predict_model(df, dataset_name, model_name, args.mode, args)
            except Exception as exc:
                manifest.append({"paper_model": dataset_info["paper_model"], "borrower_dataset": dataset_info["borrower_dataset"], "legacy_input_id": dataset_info["legacy_input_id"], "input_file": suffix, "model": model_name, "path": str(dataset_path), "status": "skipped", "reason": str(exc)})
                if args.strict:
                    raise
                warnings.warn(f"Skipped {dataset_name}/{model_name}: {exc}")
                continue

            overall.update({
                "paper_model": dataset_info["paper_model"],
                "borrower_dataset": dataset_info["borrower_dataset"],
                "legacy_input_id": dataset_info["legacy_input_id"],
                "input_file": suffix,
                "model": model_name,
                "analysis_mode": analysis_mode,
                "model_family": _model_family(dataset_name),
                "n_rows": int(len(df)),
                "n_splits": int(args.n_splits),
                "search_space": json.dumps(_space(dataset_name, model_name) if args.mode == "supplementary_search" else {}, default=str, ensure_ascii=False, sort_keys=True),
                "selected_params_by_fold": json.dumps(selected_params, default=str, ensure_ascii=False, sort_keys=True),
            })
            all_metrics.append(overall)
            for row in fold_rows:
                row.update({"paper_model": dataset_info["paper_model"], "borrower_dataset": dataset_info["borrower_dataset"], "legacy_input_id": dataset_info["legacy_input_id"], "input_file": suffix, "model": model_name, "analysis_mode": analysis_mode})
                all_fold_metrics.append(row)
            oof_df = pd.DataFrame({"paper_model": dataset_info["paper_model"], "borrower_dataset": dataset_info["borrower_dataset"], "legacy_input_id": dataset_info["legacy_input_id"], "input_file": suffix, "model": model_name, "analysis_mode": analysis_mode, "row_index": np.arange(len(df)), "y_true": _binary_target(df[TARGET_COL]).to_numpy(), "oof_score": oof})
            if ID_COL in df.columns:
                oof_df[ID_COL] = df[ID_COL].to_numpy()
            all_oof.append(oof_df)
            manifest.append({"paper_model": dataset_info["paper_model"], "borrower_dataset": dataset_info["borrower_dataset"], "legacy_input_id": dataset_info["legacy_input_id"], "input_file": suffix, "model": model_name, "path": str(dataset_path), "status": "completed", "analysis_mode": analysis_mode, "n_rows": int(len(df))})

    stem = "05_vanilla_model_comparison" if args.mode == "vanilla" else "05_supplementary_hyperparameter_search"
    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(output_dir / f"{stem}_metrics.csv", index=False)
    if all_fold_metrics:
        pd.DataFrame(all_fold_metrics).to_csv(output_dir / f"{stem}_fold_metrics.csv", index=False)
    if all_oof:
        pd.concat(all_oof, ignore_index=True).to_csv(output_dir / f"{stem}_oof_predictions.csv", index=False)
    with (output_dir / f"{stem}_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Supplementary Appendix B hyperparameter-model comparison.')
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=["G-model", "P-model"], help="Model datasets to evaluate. Use manuscript labels G-model/P-model; legacy C_C/T_T aliases are accepted.")
    parser.add_argument("--models", nargs="+", default=["LogisticRegression", "RandomForest", "XGBoost", "1D-CNN", "TabNet", "Transformer"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--classical-search-iter", type=int, default=50)
    parser.add_argument("--deep-search-iter", type=int, default=20)
    parser.add_argument("--deep-epochs", type=int, default=100)
    parser.add_argument("--deep-patience", type=int, default=10)
    parser.add_argument("--strict", action="store_true", help="Fail instead of recording skipped optional models.")
    args = parser.parse_args()
    args.paths = args.config
    args.mode = 'supplementary_search'
    return args


if __name__ == "__main__":
    run(parse_args())
