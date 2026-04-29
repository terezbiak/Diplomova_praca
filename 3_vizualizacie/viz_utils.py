from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOOKBACK_HOURS = [6, 12, 18, 24, 30, 36, 42, 48]
DEFAULT_CONTINUOUS_VARIANT = "DST+BZ_GSM"
DEFAULT_NEW_CONTINUOUS_VARIANT = "DST+BZ_GSM+V"
DEFAULT_EVENT_FEATURE_KEY = "dst_bz"
EVENT_FEATURE_KEYS = ["dst_only", "dst_v", "dst_bz", "dst_bz_v"]
EVENT_THRESHOLDS = [-20.0, -50.0]
Z95 = 1.96
Z99 = 2.576

VIZ_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VIZ_ROOT.parent
CONTINUOUS_ROOT = PROJECT_ROOT / "2_modelovanie" / "Bidirectional_LSTM_Model"
EVENT_ROOT = PROJECT_ROOT / "2_modelovanie" / "Eventy" / "event_bilstm_gp_multihorizon"
LEGACY_EVENT_ROOT = PROJECT_ROOT / "2_modelovanie" / "Eventy"
OUTPUT_ROOT = VIZ_ROOT / "vystupy_nove_data"

CONTINUOUS_FILE_PATTERN = re.compile(r"prediction_comparison_DST\+(\d+)_LSTM_(\d+)H\.csv$")
EVENT_FILE_PATTERN = re.compile(r"(.+)_L(\d+)_predictions\.csv$")


def _normalize_time(series: pd.Series) -> pd.Series:
    time_values = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return time_values.dt.tz_convert(None)
    except AttributeError:
        return time_values


def filter_date_range(
    df: pd.DataFrame,
    time_column: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    filtered[time_column] = _normalize_time(filtered[time_column])
    if start_date is not None:
        filtered = filtered[filtered[time_column] >= pd.Timestamp(start_date)]
    if end_date is not None:
        end_timestamp = pd.Timestamp(end_date)
        if end_timestamp == end_timestamp.normalize():
            end_timestamp = end_timestamp + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        filtered = filtered[filtered[time_column] <= end_timestamp]
    return filtered.copy()


def continuous_csv_path(
    horizon_hours: int,
    lookback_hours: int,
    variant_folder: str = DEFAULT_CONTINUOUS_VARIANT,
) -> Path:
    return (
        CONTINUOUS_ROOT
        / variant_folder
        / "datasets"
        / f"prediction_comparison_DST+{horizon_hours}_LSTM_{lookback_hours}H.csv"
    )


def event_prediction_path(
    lookback_hours: int,
    feature_key: str = DEFAULT_EVENT_FEATURE_KEY,
) -> Path:
    candidate_paths = [
        LEGACY_EVENT_ROOT / "predictions" / f"{feature_key}_L{lookback_hours}_predictions.csv",
        EVENT_ROOT / "predictions" / f"{feature_key}_L{lookback_hours}_predictions.csv",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return candidate_paths[0]


def load_continuous_prediction_frame(
    horizon_hours: int,
    lookback_hours: int,
    variant_folder: str = DEFAULT_CONTINUOUS_VARIANT,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame | None, Path]:
    csv_path = continuous_csv_path(horizon_hours, lookback_hours, variant_folder)
    if not csv_path.exists():
        return None, csv_path

    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = filter_date_range(df, "time", start_date, end_date)
    return ensure_numeric(
        df,
        [
            "y_true",
            "y_lstm",
            "y_hybrid",
            "hybrid_lower",
            "hybrid_upper",
            "y_pred_persistence",
        ],
    ), csv_path


def load_event_prediction_frame(
    horizon_hours: int,
    lookback_hours: int,
    feature_key: str = DEFAULT_EVENT_FEATURE_KEY,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame | None, Path]:
    csv_path = event_prediction_path(lookback_hours, feature_key)
    if not csv_path.exists():
        return None, csv_path

    df = pd.read_csv(csv_path)
    horizon_suffix = f"_h{horizon_hours}"
    required_columns = [
        "time",
        f"y_true{horizon_suffix}",
        f"y_bilstm{horizon_suffix}",
        f"y_bilstm_gp{horizon_suffix}",
        f"y_persistence{horizon_suffix}",
        f"sigma_gp{horizon_suffix}",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing columns in {csv_path.name}: {', '.join(missing_columns)}"
        )

    prepared = pd.DataFrame(
        {
            "time": df["time"],
            "DST_true": df[f"y_true{horizon_suffix}"],
            "DST_lstm": df[f"y_bilstm{horizon_suffix}"],
            "DST_pred": df[f"y_bilstm_gp{horizon_suffix}"],
            "DST_persistence": df[f"y_persistence{horizon_suffix}"],
            "sigma_nT": df[f"sigma_gp{horizon_suffix}"],
        }
    )
    prepared = ensure_numeric(
        prepared,
        ["DST_true", "DST_lstm", "DST_pred", "DST_persistence", "sigma_nT"],
    )
    prepared["DST_p025"] = prepared["DST_pred"] - Z95 * prepared["sigma_nT"]
    prepared["DST_p975"] = prepared["DST_pred"] + Z95 * prepared["sigma_nT"]
    prepared["DST_p005"] = prepared["DST_pred"] - Z99 * prepared["sigma_nT"]
    prepared["DST_p995"] = prepared["DST_pred"] + Z99 * prepared["sigma_nT"]
    prepared = filter_date_range(prepared, "time", start_date, end_date)
    return prepared, csv_path


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def plot_interval(
    ax: plt.Axes,
    x: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    label: str,
    color: str,
    alpha: float = 0.3,
    zorder: int = 1,
) -> None:
    x_values = pd.to_datetime(x, errors="coerce")
    lower_values = pd.to_numeric(lower, errors="coerce")
    upper_values = pd.to_numeric(upper, errors="coerce")

    mask = x_values.notna() & lower_values.notna() & upper_values.notna()
    if mask.sum() == 0:
        return

    ax.fill_between(
        x_values[mask],
        lower_values[mask],
        upper_values[mask],
        label=label,
        color=color,
        alpha=alpha,
        zorder=zorder,
    )


def discover_continuous_runs(
    variant_folder: str = DEFAULT_NEW_CONTINUOUS_VARIANT,
) -> pd.DataFrame:
    datasets_root = CONTINUOUS_ROOT / variant_folder / "datasets"
    rows: list[dict[str, object]] = []
    if not datasets_root.exists():
        return pd.DataFrame(columns=["variant", "horizon_hours", "lookback_hours", "path"])

    for csv_path in sorted(datasets_root.glob("prediction_comparison_DST+*_LSTM_*H.csv")):
        match = CONTINUOUS_FILE_PATTERN.match(csv_path.name)
        if not match:
            continue
        rows.append(
            {
                "variant": variant_folder,
                "horizon_hours": int(match.group(1)),
                "lookback_hours": int(match.group(2)),
                "path": csv_path,
            }
        )

    return pd.DataFrame(rows)


def discover_event_runs() -> pd.DataFrame:
    candidate_roots = [
        LEGACY_EVENT_ROOT / "predictions",
        EVENT_ROOT / "predictions",
    ]
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()

    for predictions_root in candidate_roots:
        if not predictions_root.exists():
            continue
        for csv_path in sorted(predictions_root.glob("*_L*_predictions.csv")):
            if csv_path in seen:
                continue
            seen.add(csv_path)
            match = EVENT_FILE_PATTERN.match(csv_path.name)
            if not match:
                continue
            rows.append(
                {
                    "feature_key": match.group(1),
                    "lookback_hours": int(match.group(2)),
                    "path": csv_path,
                }
            )

    return pd.DataFrame(rows)


def available_continuous_lookbacks(
    variant_folder: str = DEFAULT_NEW_CONTINUOUS_VARIANT,
) -> list[int]:
    runs = discover_continuous_runs(variant_folder)
    if runs.empty:
        return []
    return sorted(runs["lookback_hours"].dropna().astype(int).unique().tolist())


def available_continuous_horizons(
    variant_folder: str = DEFAULT_NEW_CONTINUOUS_VARIANT,
) -> list[int]:
    runs = discover_continuous_runs(variant_folder)
    if runs.empty:
        return []
    return sorted(runs["horizon_hours"].dropna().astype(int).unique().tolist())


def available_event_lookbacks() -> list[int]:
    runs = discover_event_runs()
    if runs.empty:
        return []
    return sorted(runs["lookback_hours"].dropna().astype(int).unique().tolist())


def available_event_feature_keys() -> list[str]:
    runs = discover_event_runs()
    if runs.empty:
        return []
    return sorted(runs["feature_key"].dropna().astype(str).unique().tolist())


def _binary_classification_stats(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    true_positive = ((y_true <= threshold) & (y_pred <= threshold)).sum()
    false_positive = ((y_true > threshold) & (y_pred <= threshold)).sum()
    false_negative = ((y_true <= threshold) & (y_pred > threshold)).sum()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
    }


def summarize_continuous_predictions(
    variant_folder: str = DEFAULT_NEW_CONTINUOUS_VARIANT,
) -> pd.DataFrame:
    runs = discover_continuous_runs(variant_folder)
    rows: list[dict[str, object]] = []

    for run in runs.to_dict("records"):
        df = pd.read_csv(run["path"])
        df = ensure_numeric(
            df,
            [
                "y_true",
                "y_pred_persistence",
                "y_lstm",
                "y_hybrid",
            ],
        ).dropna(subset=["y_true", "y_pred_persistence", "y_lstm", "y_hybrid"])
        if df.empty:
            continue

        y_true = df["y_true"].to_numpy(dtype=float)
        y_persistence = df["y_pred_persistence"].to_numpy(dtype=float)
        y_lstm = df["y_lstm"].to_numpy(dtype=float)
        y_hybrid = df["y_hybrid"].to_numpy(dtype=float)

        row: dict[str, object] = {
            "variant": run["variant"],
            "horizon_hours": int(run["horizon_hours"]),
            "lookback_hours": int(run["lookback_hours"]),
            "rows": len(df),
            "mae_persistence": float(np.mean(np.abs(y_true - y_persistence))),
            "mae_lstm": float(np.mean(np.abs(y_true - y_lstm))),
            "mae_hybrid": float(np.mean(np.abs(y_true - y_hybrid))),
            "rmse_persistence": float(np.sqrt(np.mean(np.square(y_true - y_persistence)))),
            "rmse_lstm": float(np.sqrt(np.mean(np.square(y_true - y_lstm)))),
            "rmse_hybrid": float(np.sqrt(np.mean(np.square(y_true - y_hybrid)))),
            "path": str(run["path"]),
        }
        for threshold in EVENT_THRESHOLDS:
            stats = _binary_classification_stats(y_true, y_hybrid, threshold)
            suffix = f"m{abs(int(threshold))}"
            row[f"precision_{suffix}"] = stats["precision"]
            row[f"recall_{suffix}"] = stats["recall"]
            row[f"f1_{suffix}"] = stats["f1"]
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_event_predictions() -> pd.DataFrame:
    runs = discover_event_runs()
    rows: list[dict[str, object]] = []

    for run in runs.to_dict("records"):
        df = pd.read_csv(run["path"])
        for horizon_hours in range(1, 7):
            suffix = f"_h{horizon_hours}"
            required_columns = [
                f"y_true{suffix}",
                f"y_persistence{suffix}",
                f"y_bilstm{suffix}",
                f"y_bilstm_gp{suffix}",
            ]
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                continue

            horizon_df = ensure_numeric(df.copy(), required_columns).dropna(subset=required_columns)
            if horizon_df.empty:
                continue

            y_true = horizon_df[f"y_true{suffix}"].to_numpy(dtype=float)
            y_persistence = horizon_df[f"y_persistence{suffix}"].to_numpy(dtype=float)
            y_bilstm = horizon_df[f"y_bilstm{suffix}"].to_numpy(dtype=float)
            y_bilstm_gp = horizon_df[f"y_bilstm_gp{suffix}"].to_numpy(dtype=float)

            row = {
                "feature_key": str(run["feature_key"]),
                "lookback_hours": int(run["lookback_hours"]),
                "horizon_hours": horizon_hours,
                "rows": len(horizon_df),
                "mae_persistence": float(np.mean(np.abs(y_true - y_persistence))),
                "mae_bilstm": float(np.mean(np.abs(y_true - y_bilstm))),
                "mae_bilstm_gp": float(np.mean(np.abs(y_true - y_bilstm_gp))),
                "rmse_persistence": float(np.sqrt(np.mean(np.square(y_true - y_persistence)))),
                "rmse_bilstm": float(np.sqrt(np.mean(np.square(y_true - y_bilstm)))),
                "rmse_bilstm_gp": float(np.sqrt(np.mean(np.square(y_true - y_bilstm_gp)))),
                "path": str(run["path"]),
            }
            for threshold in EVENT_THRESHOLDS:
                stats = _binary_classification_stats(y_true, y_bilstm_gp, threshold)
                suffix_name = f"m{abs(int(threshold))}"
                row[f"precision_{suffix_name}"] = stats["precision"]
                row[f"recall_{suffix_name}"] = stats["recall"]
                row[f"f1_{suffix_name}"] = stats["f1"]
            rows.append(row)

    return pd.DataFrame(rows)


def _sorted_feature_keys(preferred_keys: list[str], available_keys: list[str]) -> list[str]:
    ordered = [key for key in preferred_keys if key in available_keys]
    ordered.extend(key for key in available_keys if key not in ordered)
    return ordered


def _heatmap(
    ax: plt.Axes,
    pivot_df: pd.DataFrame,
    title: str,
    cmap: str = "viridis_r",
    value_fmt: str = "{:.2f}",
) -> None:
    if pivot_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return

    values = pivot_df.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([str(value) for value in pivot_df.columns])
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels([str(value) for value in pivot_df.index])
    ax.set_xlabel(pivot_df.columns.name or "")
    ax.set_ylabel(pivot_df.index.name or "")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    threshold = np.nanmean(values)
    for row_idx in range(pivot_df.shape[0]):
        for col_idx in range(pivot_df.shape[1]):
            value = pivot_df.iat[row_idx, col_idx]
            if pd.isna(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                value_fmt.format(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )


def save_event_summary_csv(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "event_summary_metrics.csv"
    summary_df.sort_values(["feature_key", "lookback_hours", "horizon_hours"]).to_csv(output_path, index=False)
    return output_path


def save_continuous_summary_csv(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "continuous_summary_metrics.csv"
    summary_df.sort_values(["variant", "lookback_hours", "horizon_hours"]).to_csv(output_path, index=False)
    return output_path


def plot_event_metric_heatmaps(
    summary_df: pd.DataFrame,
    metric: str,
    title_prefix: str,
    output_path: Path,
) -> None:
    horizons = sorted(summary_df["horizon_hours"].dropna().astype(int).unique().tolist())
    if not horizons:
        return

    rows = math.ceil(len(horizons) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, 3)

    for idx, horizon in enumerate(horizons):
        ax = axes_array[idx // 3, idx % 3]
        horizon_df = summary_df[summary_df["horizon_hours"] == horizon]
        pivot_df = horizon_df.pivot(
            index="feature_key",
            columns="lookback_hours",
            values=metric,
        ).sort_index()
        pivot_df.index.name = "Feature set"
        pivot_df.columns.name = "Lookback [h]"
        _heatmap(ax, pivot_df, f"{title_prefix} | DST+{horizon}")

    for idx in range(len(horizons), rows * 3):
        axes_array[idx // 3, idx % 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_continuous_metric_heatmap(
    summary_df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    pivot_df = summary_df.pivot(
        index="horizon_hours",
        columns="lookback_hours",
        values=metric,
    ).sort_index()
    pivot_df.index.name = "Horizon [h]"
    pivot_df.columns.name = "Lookback [h]"

    fig, ax = plt.subplots(figsize=(10, 6))
    _heatmap(ax, pivot_df, title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_event_lines_by_feature(
    summary_df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    feature_keys = _sorted_feature_keys(EVENT_FEATURE_KEYS, available_event_feature_keys())
    if not feature_keys:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes_array = axes.flatten()

    for ax, feature_key in zip(axes_array, feature_keys):
        feature_df = summary_df[summary_df["feature_key"] == feature_key]
        for lookback_hours in sorted(feature_df["lookback_hours"].dropna().astype(int).unique().tolist()):
            lookback_df = feature_df[feature_df["lookback_hours"] == lookback_hours].sort_values("horizon_hours")
            ax.plot(
                lookback_df["horizon_hours"],
                lookback_df[metric],
                marker="o",
                linewidth=2,
                label=f"L={lookback_hours} h",
            )
        ax.set_title(feature_key)
        ax.set_xlabel("Horizon [h]")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        if not feature_df.empty:
            ax.legend(fontsize=8)

    for idx in range(len(feature_keys), len(axes_array)):
        axes_array[idx].axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_continuous_lines_by_lookback(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lookbacks = sorted(summary_df["lookback_hours"].dropna().astype(int).unique().tolist())
    if not lookbacks:
        return

    rows = math.ceil(len(lookbacks) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4.5 * rows), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).reshape(rows, 2)

    for idx, lookback_hours in enumerate(lookbacks):
        ax = axes_array[idx // 2, idx % 2]
        lookback_df = summary_df[summary_df["lookback_hours"] == lookback_hours].sort_values("horizon_hours")
        ax.plot(lookback_df["horizon_hours"], lookback_df["mae_persistence"], marker="o", linewidth=2, label="Persistence")
        ax.plot(lookback_df["horizon_hours"], lookback_df["mae_lstm"], marker="o", linewidth=2, label="BiLSTM")
        ax.plot(lookback_df["horizon_hours"], lookback_df["mae_hybrid"], marker="o", linewidth=2, label="BiLSTM+GP")
        ax.set_title(f"Lookback = {lookback_hours} h")
        ax.set_xlabel("Horizon [h]")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(lookbacks), rows * 2):
        axes_array[idx // 2, idx % 2].axis("off")

    fig.suptitle("Continuous experiment | MAE by horizon", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_continuous_window_grid(
    continuous_variant: str,
    horizon_hours: int,
    start_date: str | None,
    end_date: str | None,
    output_path: Path,
) -> None:
    lookbacks = available_continuous_lookbacks(continuous_variant)
    if not lookbacks:
        return

    rows = math.ceil(len(lookbacks) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(18, 4.8 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, 2)

    for idx, lookback_hours in enumerate(lookbacks):
        ax = axes_array[idx // 2, idx % 2]
        df, csv_path = load_continuous_prediction_frame(
            horizon_hours=horizon_hours,
            lookback_hours=lookback_hours,
            variant_folder=continuous_variant,
            start_date=start_date,
            end_date=end_date,
        )
        if df is None or df.empty:
            ax.text(0.5, 0.5, f"Missing or empty file\n{csv_path.name}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Continuous | DST+{horizon_hours} | L={lookback_hours} h")
            ax.axis("off")
            continue

        df = df.sort_values("time").copy()
        plot_interval(ax, df["time"], df["hybrid_lower"], df["hybrid_upper"], "Hybrid interval", "lightcoral", alpha=0.25, zorder=1)
        ax.plot(df["time"], df["y_true"], label="Real values", color="black", linewidth=2.3, zorder=3)
        ax.plot(df["time"], df["y_lstm"], label="BiLSTM", color="tab:orange", linewidth=2, zorder=4)
        ax.plot(df["time"], df["y_hybrid"], label="BiLSTM+GP", color="tab:red", linewidth=2, zorder=5)
        ax.plot(df["time"], df["y_pred_persistence"], label="Persistence", color="tab:green", linewidth=1.8, linestyle="--", zorder=4)
        ax.set_title(f"Continuous | DST+{horizon_hours} | L={lookback_hours} h")
        ax.set_xlabel("Time")
        ax.set_ylabel("DST")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    for idx in range(len(lookbacks), rows * 2):
        axes_array[idx // 2, idx % 2].axis("off")

    fig.suptitle(f"{continuous_variant} | time-window comparison", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_event_window_by_feature(
    horizon_hours: int,
    lookback_hours: int,
    start_date: str | None,
    end_date: str | None,
    output_path: Path,
) -> None:
    feature_keys = _sorted_feature_keys(EVENT_FEATURE_KEYS, available_event_feature_keys())
    if not feature_keys:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes_array = axes.flatten()

    for ax, feature_key in zip(axes_array, feature_keys):
        df, csv_path = load_event_prediction_frame(
            horizon_hours=horizon_hours,
            lookback_hours=lookback_hours,
            feature_key=feature_key,
            start_date=start_date,
            end_date=end_date,
        )
        if df is None or df.empty:
            ax.text(0.5, 0.5, f"Missing or empty file\n{csv_path.name}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{feature_key} | DST+{horizon_hours} | L={lookback_hours} h")
            ax.axis("off")
            continue

        df = df.sort_values("time").copy()
        plot_interval(ax, df["time"], df["DST_p005"], df["DST_p995"], "99% interval", "lightskyblue", alpha=0.35, zorder=1)
        plot_interval(ax, df["time"], df["DST_p025"], df["DST_p975"], "95% interval", "deepskyblue", alpha=0.25, zorder=2)
        ax.plot(df["time"], df["DST_true"], label="Real values", color="black", linewidth=2.3, zorder=4)
        ax.plot(df["time"], df["DST_lstm"], label="BiLSTM", color="tab:orange", linewidth=2, zorder=4)
        ax.plot(df["time"], df["DST_pred"], label="BiLSTM+GP", color="tab:blue", linewidth=2, zorder=5)
        ax.plot(df["time"], df["DST_persistence"], label="Persistence", color="tab:green", linewidth=1.8, linestyle="--", zorder=3)
        ax.set_title(f"{feature_key} | DST+{horizon_hours} | L={lookback_hours} h")
        ax.set_xlabel("Time")
        ax.set_ylabel("DST")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    for idx in range(len(feature_keys), len(axes_array)):
        axes_array[idx].axis("off")

    fig.suptitle("Event experiment | feature-set comparison", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_continuous_vs_event_window(
    continuous_variant: str,
    event_feature_key: str,
    horizon_hours: int,
    start_date: str | None,
    end_date: str | None,
    output_path: Path,
) -> None:
    shared_lookbacks = sorted(
        set(available_continuous_lookbacks(continuous_variant)).intersection(available_event_lookbacks())
    )
    if not shared_lookbacks:
        return

    fig, axes = plt.subplots(len(shared_lookbacks), 2, figsize=(20, 4.8 * len(shared_lookbacks)))
    axes_array = np.atleast_2d(axes)

    for row_idx, lookback_hours in enumerate(shared_lookbacks):
        ax_left = axes_array[row_idx, 0]
        ax_right = axes_array[row_idx, 1]

        df_cont, cont_path = load_continuous_prediction_frame(
            horizon_hours=horizon_hours,
            lookback_hours=lookback_hours,
            variant_folder=continuous_variant,
            start_date=start_date,
            end_date=end_date,
        )
        df_event, event_path = load_event_prediction_frame(
            horizon_hours=horizon_hours,
            lookback_hours=lookback_hours,
            feature_key=event_feature_key,
            start_date=start_date,
            end_date=end_date,
        )

        if df_cont is not None and not df_cont.empty:
            df_cont = df_cont.sort_values("time").copy()
            plot_interval(ax_left, df_cont["time"], df_cont["hybrid_lower"], df_cont["hybrid_upper"], "Hybrid interval", "lightcoral", alpha=0.25, zorder=1)
            ax_left.plot(df_cont["time"], df_cont["y_true"], label="Real values", color="black", linewidth=2.3, zorder=3)
            ax_left.plot(df_cont["time"], df_cont["y_lstm"], label="BiLSTM", color="tab:orange", linewidth=2, zorder=4)
            ax_left.plot(df_cont["time"], df_cont["y_hybrid"], label="BiLSTM+GP", color="tab:red", linewidth=2, zorder=5)
            ax_left.plot(df_cont["time"], df_cont["y_pred_persistence"], label="Persistence", color="tab:green", linewidth=1.8, linestyle="--", zorder=4)
            ax_left.legend(fontsize=8)
        else:
            ax_left.text(0.5, 0.5, f"Missing or empty file\n{cont_path.name}", ha="center", va="center", transform=ax_left.transAxes)

        ax_left.set_title(f"Continuous | DST+{horizon_hours} | L={lookback_hours} h")
        ax_left.set_xlabel("Time")
        ax_left.set_ylabel("DST")
        ax_left.grid(alpha=0.3)
        ax_left.tick_params(axis="x", rotation=45)

        if df_event is not None and not df_event.empty:
            df_event = df_event.sort_values("time").copy()
            plot_interval(ax_right, df_event["time"], df_event["DST_p005"], df_event["DST_p995"], "99% interval", "lightskyblue", alpha=0.35, zorder=1)
            plot_interval(ax_right, df_event["time"], df_event["DST_p025"], df_event["DST_p975"], "95% interval", "deepskyblue", alpha=0.25, zorder=2)
            ax_right.plot(df_event["time"], df_event["DST_true"], label="Real values", color="black", linewidth=2.3, zorder=4)
            ax_right.plot(df_event["time"], df_event["DST_lstm"], label="BiLSTM", color="tab:orange", linewidth=2, zorder=4)
            ax_right.plot(df_event["time"], df_event["DST_pred"], label="BiLSTM+GP", color="tab:blue", linewidth=2, zorder=5)
            ax_right.plot(df_event["time"], df_event["DST_persistence"], label="Persistence", color="tab:green", linewidth=1.8, linestyle="--", zorder=3)
            ax_right.legend(fontsize=8)
        else:
            ax_right.text(0.5, 0.5, f"Missing or empty file\n{event_path.name}", ha="center", va="center", transform=ax_right.transAxes)

        ax_right.set_title(f"Event | DST+{horizon_hours} | L={lookback_hours} h | {event_feature_key}")
        ax_right.set_xlabel("Time")
        ax_right.set_ylabel("DST")
        ax_right.grid(alpha=0.3)
        ax_right.tick_params(axis="x", rotation=45)

    fig.suptitle("Continuous vs event experiment", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
