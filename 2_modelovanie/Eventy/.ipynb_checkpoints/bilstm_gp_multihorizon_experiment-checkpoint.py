from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, layers


SEED = 42
TARGET_COLS = [f"DST+{h}" for h in range(1, 7)]
FEATURE_SETS = {
    "dst_only": ["dst"],
    "dst_v": ["dst", "v"],
    "dst_bz": ["dst", "bz_gsm"],
    "dst_bz_v": ["dst", "bz_gsm", "v"],
}


@dataclass
class ExperimentConfig:
    lookbacks: list[int]
    feature_keys: list[str]
    thresholds: list[float]
    epochs: int = 50
    batch_size: int = 256
    hidden: int = 64
    dropout: float = 0.2
    patience: int = 8
    max_gp_points: int = 4000
    gp_restarts: int = 1
    class_prob_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.10, 0.90, 17))


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def default_paths(base_dir: Path | None = None) -> dict[str, Path]:
    if base_dir is None:
        base_dir = Path.cwd()
    output_root = base_dir / "event_bilstm_gp_multihorizon"
    paths = {
        "data": base_dir.parent.parent / "0_datasety" / "event_omni_prepared.csv",
        "output_root": output_root,
        "models": output_root / "models",
        "predictions": output_root / "predictions",
        "metrics": output_root / "metrics",
        "splits": output_root / "splits",
    }
    for key in ["output_root", "models", "predictions", "metrics", "splits"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def load_event_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["time1"] = pd.to_datetime(df["time1"], utc=True)
    df = df.sort_values(["event_no", "time1"]).reset_index(drop=True)
    for col in ["bz_gsm", "v", "dst"] + TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_event_metadata(df_events: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df_events.groupby("event_no")
        .agg(
            start_time=("time1", "min"),
            end_time=("time1", "max"),
            n_rows=("time1", "size"),
            min_dst=("dst", "min"),
        )
        .reset_index()
    )
    meta["duration_hours"] = meta["n_rows"].astype(int)
    meta["severity_bin"] = pd.cut(
        meta["min_dst"],
        bins=[-np.inf, -200, -100, -75, -50, np.inf],
        labels=["extreme", "strong", "moderate", "mild", "other"],
        include_lowest=True,
    ).astype(str)
    return meta


def _maybe_stratify(labels: pd.Series):
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def split_events(event_meta: pd.DataFrame, seed: int = SEED, test_size: float = 0.15, val_size: float = 0.15):
    stratify_all = _maybe_stratify(event_meta["severity_bin"])
    train_val, test = train_test_split(
        event_meta,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_all,
    )
    val_relative = val_size / (1.0 - test_size)
    stratify_train_val = _maybe_stratify(train_val["severity_bin"])
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        random_state=seed,
        stratify=stratify_train_val,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_multihorizon_sequences(
    df_part: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    split_name: str,
):
    x_list, y_list, time_list, event_list = [], [], [], []
    for event_no, g in df_part.groupby("event_no"):
        g = g.sort_values("time1").reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g[TARGET_COLS].to_numpy(dtype=np.float32)
        times = g["time1"].to_numpy()
        valid_targets = np.isfinite(targets).all(axis=1)
        for idx in range(lookback - 1, len(g)):
            if not valid_targets[idx]:
                continue
            x_list.append(feats[idx - lookback + 1 : idx + 1])
            y_list.append(targets[idx])
            time_list.append(times[idx])
            event_list.append(int(event_no))
    if not x_list:
        raise ValueError(
            "No valid sequences were created for "
            f'split="{split_name}", lookback={lookback}, '
            f"features={feature_cols}, events={df_part['event_no'].nunique()}."
        )
    x = np.stack(x_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    t = pd.to_datetime(np.array(time_list), utc=True)
    ev = np.array(event_list, dtype=np.int32)
    return x, y, t, ev


def fit_scalers(x_train: np.ndarray, y_train: np.ndarray):
    n_features = x_train.shape[-1]
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(x_train.reshape(-1, n_features))
    y_scaler.fit(y_train)
    return x_scaler, y_scaler


def transform_xy(x: np.ndarray, y: np.ndarray, x_scaler: StandardScaler, y_scaler: StandardScaler):
    n_samples, lookback, n_features = x.shape
    x_scaled = x_scaler.transform(x.reshape(-1, n_features)).reshape(n_samples, lookback, n_features).astype(np.float32)
    y_scaled = y_scaler.transform(y).astype(np.float32)
    return x_scaled, y_scaled


def build_bilstm_multihorizon_model(lookback: int, n_features: int, n_targets: int, hidden: int, dropout: float):
    inp = layers.Input(shape=(lookback, n_features), name="x")
    x = layers.Bidirectional(layers.LSTM(hidden, dropout=dropout), name="bilstm")(inp)
    x = layers.Dropout(dropout, name="dropout")(x)
    emb = layers.Dense(hidden, activation="relu", name="emb")(x)
    out = layers.Dense(n_targets, name="y")(emb)
    model = Model(inp, out, name="bilstm_multihorizon")
    emb_model = Model(inp, emb, name="embedding_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model, emb_model


def fit_gp_models(emb_train: np.ndarray, y_true_s: np.ndarray, y_pred_s: np.ndarray, max_gp_points: int, n_restarts: int):
    residuals = y_true_s - y_pred_s
    gp_models = []
    train_sizes = []
    for horizon in range(residuals.shape[1]):
        x_gp = emb_train
        y_gp = residuals[:, horizon]
        if len(x_gp) > max_gp_points:
            rng = np.random.default_rng(SEED + horizon)
            idx = rng.choice(len(x_gp), size=max_gp_points, replace=False)
            x_sub = x_gp[idx]
            y_sub = y_gp[idx]
        else:
            x_sub = x_gp
            y_sub = y_gp
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-2,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=SEED + horizon,
        )
        gp.fit(x_sub, y_sub)
        gp_models.append(gp)
        train_sizes.append(len(x_sub))
    return gp_models, train_sizes


def predict_gp_models(gp_models: list[GaussianProcessRegressor], emb_data: np.ndarray, y_scale: np.ndarray):
    mu_s = []
    std = []
    for horizon, gp in enumerate(gp_models):
        pred_mu_s, pred_std_s = gp.predict(emb_data, return_std=True)
        mu_s.append(pred_mu_s)
        std.append(pred_std_s * float(y_scale[horizon]))
    return np.column_stack(mu_s), np.column_stack(std)


def choose_probability_cutoff(y_true_binary: np.ndarray, y_prob: np.ndarray, grid: np.ndarray) -> float:
    if len(np.unique(y_true_binary)) < 2:
        return 0.5
    best_cutoff = 0.5
    best_score = -np.inf
    for cutoff in grid:
        y_pred = (y_prob >= cutoff).astype(int)
        score = f1_score(y_true_binary, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_cutoff = float(cutoff)
    return best_cutoff


def regression_rows(feature_key: str, lookback: int, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray | None = None):
    rows = []
    for idx, horizon in enumerate(range(1, 7)):
        row = {
            "feature_set": feature_key,
            "lookback_hours": lookback,
            "horizon_hours": horizon,
            "model": model_name,
            "mae": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))),
        }
        if y_std is not None:
            z90 = norm.ppf(0.95)
            z95 = norm.ppf(0.975)
            lo90 = y_pred[:, idx] - z90 * y_std[:, idx]
            hi90 = y_pred[:, idx] + z90 * y_std[:, idx]
            lo95 = y_pred[:, idx] - z95 * y_std[:, idx]
            hi95 = y_pred[:, idx] + z95 * y_std[:, idx]
            row["picp_90"] = float(np.mean((y_true[:, idx] >= lo90) & (y_true[:, idx] <= hi90)))
            row["mpiw_90"] = float(np.mean(hi90 - lo90))
            row["picp_95"] = float(np.mean((y_true[:, idx] >= lo95) & (y_true[:, idx] <= hi95)))
            row["mpiw_95"] = float(np.mean(hi95 - lo95))
        rows.append(row)
    return rows


def classification_rows_point(feature_key: str, lookback: int, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, dst_threshold: float):
    rows = []
    for idx, horizon in enumerate(range(1, 7)):
        y_true_bin = (y_true[:, idx] <= dst_threshold).astype(int)
        y_pred_bin = (y_pred[:, idx] <= dst_threshold).astype(int)
        rows.append(
            {
                "feature_set": feature_key,
                "lookback_hours": lookback,
                "horizon_hours": horizon,
                "model": model_name,
                "dst_threshold": float(dst_threshold),
                "decision_rule": "point_threshold",
                "prob_cutoff": np.nan,
                "precision_class1": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
                "recall_class1": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
                "f1_class1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true_bin, y_pred_bin)),
                "positive_rate_true": float(np.mean(y_true_bin)),
                "positive_rate_pred": float(np.mean(y_pred_bin)),
            }
        )
    return rows


def classification_rows_prob(
    feature_key: str,
    lookback: int,
    model_name: str,
    y_val_true: np.ndarray,
    y_val_mean: np.ndarray,
    y_val_std: np.ndarray,
    y_test_true: np.ndarray,
    y_test_mean: np.ndarray,
    y_test_std: np.ndarray,
    dst_threshold: float,
    prob_grid: np.ndarray,
):
    rows = []
    safe_val_std = np.maximum(y_val_std, 1e-6)
    safe_test_std = np.maximum(y_test_std, 1e-6)
    for idx, horizon in enumerate(range(1, 7)):
        y_val_bin = (y_val_true[:, idx] <= dst_threshold).astype(int)
        y_test_bin = (y_test_true[:, idx] <= dst_threshold).astype(int)
        y_val_prob = norm.cdf((dst_threshold - y_val_mean[:, idx]) / safe_val_std[:, idx])
        y_test_prob = norm.cdf((dst_threshold - y_test_mean[:, idx]) / safe_test_std[:, idx])
        cutoff = choose_probability_cutoff(y_val_bin, y_val_prob, prob_grid)
        y_test_pred = (y_test_prob >= cutoff).astype(int)
        rows.append(
            {
                "feature_set": feature_key,
                "lookback_hours": lookback,
                "horizon_hours": horizon,
                "model": model_name,
                "dst_threshold": float(dst_threshold),
                "decision_rule": "probability_threshold",
                "prob_cutoff": float(cutoff),
                "precision_class1": float(precision_score(y_test_bin, y_test_pred, zero_division=0)),
                "recall_class1": float(recall_score(y_test_bin, y_test_pred, zero_division=0)),
                "f1_class1": float(f1_score(y_test_bin, y_test_pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_test_bin, y_test_pred)),
                "positive_rate_true": float(np.mean(y_test_bin)),
                "positive_rate_pred": float(np.mean(y_test_pred)),
            }
        )
    return rows


def run_one_experiment(
    df_events: pd.DataFrame,
    train_event_ids: list[int],
    val_event_ids: list[int],
    test_event_ids: list[int],
    feature_key: str,
    lookback: int,
    config: ExperimentConfig,
    paths: dict[str, Path],
):
    feature_cols = FEATURE_SETS[feature_key]
    run_name = f"{feature_key}_L{lookback}"
    print(f"Running {run_name}")
    train_df = df_events[df_events["event_no"].isin(train_event_ids)].copy()
    val_df = df_events[df_events["event_no"].isin(val_event_ids)].copy()
    test_df = df_events[df_events["event_no"].isin(test_event_ids)].copy()

    x_train_raw, y_train_raw, _, _ = make_multihorizon_sequences(
        train_df, feature_cols, lookback, split_name="train"
    )
    x_val_raw, y_val_raw, t_val, ev_val = make_multihorizon_sequences(
        val_df, feature_cols, lookback, split_name="val"
    )
    x_test_raw, y_test_raw, t_test, ev_test = make_multihorizon_sequences(
        test_df, feature_cols, lookback, split_name="test"
    )

    x_scaler, y_scaler = fit_scalers(x_train_raw, y_train_raw)
    x_train, y_train = transform_xy(x_train_raw, y_train_raw, x_scaler, y_scaler)
    x_val, y_val = transform_xy(x_val_raw, y_val_raw, x_scaler, y_scaler)
    x_test, y_test = transform_xy(x_test_raw, y_test_raw, x_scaler, y_scaler)

    model, emb_model = build_bilstm_multihorizon_model(
        lookback=lookback,
        n_features=x_train.shape[-1],
        n_targets=y_train.shape[-1],
        hidden=config.hidden,
        dropout=config.dropout,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=config.patience, restore_best_weights=True, mode="min"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=max(2, config.patience // 2), min_lr=1e-5, mode="min"),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(paths["models"] / f"{run_name}.keras")

    y_train_lstm_s = model.predict(x_train, batch_size=1024, verbose=0)
    y_val_lstm_s = model.predict(x_val, batch_size=1024, verbose=0)
    y_test_lstm_s = model.predict(x_test, batch_size=1024, verbose=0)
    emb_train = emb_model.predict(x_train, batch_size=1024, verbose=0)
    emb_val = emb_model.predict(x_val, batch_size=1024, verbose=0)
    emb_test = emb_model.predict(x_test, batch_size=1024, verbose=0)

    gp_models, gp_train_sizes = fit_gp_models(
        emb_train=emb_train,
        y_true_s=y_train,
        y_pred_s=y_train_lstm_s,
        max_gp_points=config.max_gp_points,
        n_restarts=config.gp_restarts,
    )
    y_val_gp_mu_s, y_val_gp_std = predict_gp_models(gp_models, emb_val, y_scaler.scale_)
    y_test_gp_mu_s, y_test_gp_std = predict_gp_models(gp_models, emb_test, y_scaler.scale_)

    y_val_lstm = y_scaler.inverse_transform(y_val_lstm_s)
    y_test_lstm = y_scaler.inverse_transform(y_test_lstm_s)
    y_val_hybrid = y_scaler.inverse_transform(y_val_lstm_s + y_val_gp_mu_s)
    y_test_hybrid = y_scaler.inverse_transform(y_test_lstm_s + y_test_gp_mu_s)
    y_val_persistence = np.repeat(x_val_raw[:, -1, 0:1], len(TARGET_COLS), axis=1)
    y_test_persistence = np.repeat(x_test_raw[:, -1, 0:1], len(TARGET_COLS), axis=1)

    regression_records = []
    regression_records.extend(regression_rows(feature_key, lookback, "persistence", y_test_raw, y_test_persistence))
    regression_records.extend(regression_rows(feature_key, lookback, "bilstm", y_test_raw, y_test_lstm))
    regression_records.extend(regression_rows(feature_key, lookback, "bilstm_gp", y_test_raw, y_test_hybrid, y_std=y_test_gp_std))

    classification_records = []
    for dst_threshold in config.thresholds:
        classification_records.extend(classification_rows_point(feature_key, lookback, "persistence", y_test_raw, y_test_persistence, dst_threshold))
        classification_records.extend(classification_rows_point(feature_key, lookback, "bilstm", y_test_raw, y_test_lstm, dst_threshold))
        classification_records.extend(
            classification_rows_prob(
                feature_key,
                lookback,
                "bilstm_gp",
                y_val_raw,
                y_val_hybrid,
                y_val_gp_std,
                y_test_raw,
                y_test_hybrid,
                y_test_gp_std,
                dst_threshold,
                config.class_prob_grid,
            )
        )

    predictions = {"time": t_test, "event_no": ev_test}
    for idx, horizon in enumerate(range(1, 7)):
        predictions[f"y_true_h{horizon}"] = y_test_raw[:, idx]
        predictions[f"y_persistence_h{horizon}"] = y_test_persistence[:, idx]
        predictions[f"y_bilstm_h{horizon}"] = y_test_lstm[:, idx]
        predictions[f"y_bilstm_gp_h{horizon}"] = y_test_hybrid[:, idx]
        predictions[f"sigma_gp_h{horizon}"] = y_test_gp_std[:, idx]
        for dst_threshold in config.thresholds:
            probs = norm.cdf((dst_threshold - y_test_hybrid[:, idx]) / np.maximum(y_test_gp_std[:, idx], 1e-6))
            predictions[f"p_dst_le_{int(dst_threshold)}_h{horizon}"] = probs
    pred_df = pd.DataFrame(predictions)
    prediction_path = paths["predictions"] / f"{run_name}_predictions.csv"
    pred_df.to_csv(prediction_path, index=False)

    return {
        "run_name": run_name,
        "history": history.history,
        "regression_records": regression_records,
        "classification_records": classification_records,
        "prediction_path": str(prediction_path),
        "gp_train_sizes": gp_train_sizes,
    }


def run_full_grid(df_events: pd.DataFrame, train_meta: pd.DataFrame, val_meta: pd.DataFrame, test_meta: pd.DataFrame, config: ExperimentConfig, paths: dict[str, Path]):
    train_event_ids = train_meta["event_no"].tolist()
    val_event_ids = val_meta["event_no"].tolist()
    test_event_ids = test_meta["event_no"].tolist()

    all_regression_records = []
    all_classification_records = []
    run_overview = []

    for feature_key in config.feature_keys:
        for lookback in config.lookbacks:
            result = run_one_experiment(
                df_events=df_events,
                train_event_ids=train_event_ids,
                val_event_ids=val_event_ids,
                test_event_ids=test_event_ids,
                feature_key=feature_key,
                lookback=lookback,
                config=config,
                paths=paths,
            )
            all_regression_records.extend(result["regression_records"])
            all_classification_records.extend(result["classification_records"])
            run_overview.append(
                {
                    "run_name": result["run_name"],
                    "prediction_path": result["prediction_path"],
                    "gp_train_sizes": result["gp_train_sizes"],
                }
            )

    regression_df = pd.DataFrame(all_regression_records)
    classification_df = pd.DataFrame(all_classification_records)
    overview_df = pd.DataFrame(run_overview)

    regression_df.to_csv(paths["metrics"] / "regression_summary.csv", index=False)
    classification_df.to_csv(paths["metrics"] / "classification_summary.csv", index=False)
    overview_df.to_csv(paths["metrics"] / "run_overview.csv", index=False)

    return regression_df, classification_df, overview_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the event-based BiLSTM + GP multihorizon experiment."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory that contains this experiment module and notebook.",
    )
    parser.add_argument(
        "--lookbacks",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
        help="Lookback windows in hours.",
    )
    parser.add_argument(
        "--feature-keys",
        nargs="+",
        default=list(FEATURE_SETS.keys()),
        choices=list(FEATURE_SETS.keys()),
        help="Feature set keys to evaluate.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[-20.0, -50.0],
        help="DST thresholds used for classification metrics.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-gp-points", type=int, default=4000)
    parser.add_argument("--gp-restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    base_dir = args.base_dir.resolve()
    paths = default_paths(base_dir)
    df = load_event_dataset(paths["data"])
    event_meta = build_event_metadata(df)
    train_meta, val_meta, test_meta = split_events(event_meta, seed=args.seed)

    train_meta.to_csv(paths["splits"] / "train_events.csv", index=False)
    val_meta.to_csv(paths["splits"] / "val_events.csv", index=False)
    test_meta.to_csv(paths["splits"] / "test_events.csv", index=False)

    print(f"Base dir: {base_dir}")
    print(f"Dataset: {paths['data']}")
    print(f"Rows: {len(df):,}")
    print(f"Events: {df['event_no'].nunique():,}")
    print(f"Lookbacks: {args.lookbacks}")
    print(f"Feature sets: {args.feature_keys}")
    print(f"Thresholds: {args.thresholds}")

    config = ExperimentConfig(
        lookbacks=args.lookbacks,
        feature_keys=args.feature_keys,
        thresholds=args.thresholds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        dropout=args.dropout,
        patience=args.patience,
        max_gp_points=args.max_gp_points,
        gp_restarts=args.gp_restarts,
    )
    regression_df, classification_df, overview_df = run_full_grid(
        df_events=df,
        train_meta=train_meta,
        val_meta=val_meta,
        test_meta=test_meta,
        config=config,
        paths=paths,
    )

    print()
    print(f"Saved regression summary: {paths['metrics'] / 'regression_summary.csv'}")
    print(f"Saved classification summary: {paths['metrics'] / 'classification_summary.csv'}")
    print(f"Saved run overview: {paths['metrics'] / 'run_overview.csv'}")
    print(f"Regression rows: {len(regression_df):,}")
    print(f"Classification rows: {len(classification_df):,}")
    print(f"Runs: {len(overview_df):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
