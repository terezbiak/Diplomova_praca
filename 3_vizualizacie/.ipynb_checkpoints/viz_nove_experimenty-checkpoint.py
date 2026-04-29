from __future__ import annotations

import argparse
from pathlib import Path

from viz_utils import (
    DEFAULT_NEW_CONTINUOUS_VARIANT,
    OUTPUT_ROOT,
    available_continuous_horizons,
    available_event_lookbacks,
    plot_continuous_lines_by_lookback,
    plot_continuous_metric_heatmap,
    plot_continuous_vs_event_window,
    plot_continuous_window_grid,
    plot_event_lines_by_feature,
    plot_event_metric_heatmaps,
    plot_event_window_by_feature,
    save_continuous_summary_csv,
    save_event_summary_csv,
    summarize_continuous_predictions,
    summarize_event_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualizations for the new event and continuous experiment CSV files."
    )
    parser.add_argument(
        "--continuous-variant",
        default=DEFAULT_NEW_CONTINUOUS_VARIANT,
        help="Continuous variant folder under 2_modelovanie/Bidirectional_LSTM_Model.",
    )
    parser.add_argument(
        "--event-feature-key",
        default="dst_bz_v",
        help="Default event feature key for side-by-side time plots.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional start date for time-window plots, e.g. 2024-12-31.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional end date for time-window plots, e.g. 2025-01-03.",
    )
    parser.add_argument(
        "--window-horizon",
        type=int,
        default=1,
        help="Forecast horizon used in example time-window plots.",
    )
    parser.add_argument(
        "--window-lookback",
        type=int,
        default=6,
        help="Lookback used for the event feature comparison window plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated PNG and CSV outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    continuous_summary = summarize_continuous_predictions(args.continuous_variant)
    event_summary = summarize_event_predictions()

    if not continuous_summary.empty:
        save_continuous_summary_csv(continuous_summary, args.output_dir)
        plot_continuous_metric_heatmap(
            continuous_summary,
            metric="mae_hybrid",
            title=f"Continuous | BiLSTM+GP MAE | {args.continuous_variant}",
            output_path=args.output_dir / "continuous_mae_hybrid_heatmap.png",
        )
        plot_continuous_metric_heatmap(
            continuous_summary,
            metric="f1_m20",
            title=f"Continuous | F1 at DST <= -20 | {args.continuous_variant}",
            output_path=args.output_dir / "continuous_f1_m20_heatmap.png",
        )
        plot_continuous_lines_by_lookback(
            continuous_summary,
            output_path=args.output_dir / "continuous_mae_by_horizon.png",
        )

    if not event_summary.empty:
        save_event_summary_csv(event_summary, args.output_dir)
        plot_event_metric_heatmaps(
            event_summary,
            metric="mae_bilstm_gp",
            title_prefix="Event | BiLSTM+GP MAE",
            output_path=args.output_dir / "event_mae_bilstm_gp_heatmaps.png",
        )
        plot_event_metric_heatmaps(
            event_summary,
            metric="f1_m20",
            title_prefix="Event | F1 at DST <= -20",
            output_path=args.output_dir / "event_f1_m20_heatmaps.png",
        )
        plot_event_metric_heatmaps(
            event_summary,
            metric="f1_m50",
            title_prefix="Event | F1 at DST <= -50",
            output_path=args.output_dir / "event_f1_m50_heatmaps.png",
        )
        plot_event_lines_by_feature(
            event_summary,
            metric="mae_bilstm_gp",
            title="Event experiment | BiLSTM+GP MAE by horizon",
            output_path=args.output_dir / "event_mae_by_feature.png",
        )
        plot_event_lines_by_feature(
            event_summary,
            metric="f1_m20",
            title="Event experiment | F1 at DST <= -20 by horizon",
            output_path=args.output_dir / "event_f1_m20_by_feature.png",
        )

    if args.start_date or args.end_date:
        if args.window_horizon in available_continuous_horizons(args.continuous_variant):
            plot_continuous_window_grid(
                continuous_variant=args.continuous_variant,
                horizon_hours=args.window_horizon,
                start_date=args.start_date,
                end_date=args.end_date,
                output_path=args.output_dir / f"continuous_window_DST+{args.window_horizon}.png",
            )
        if args.window_horizon in range(1, 7) and args.window_lookback in available_event_lookbacks():
            plot_event_window_by_feature(
                horizon_hours=args.window_horizon,
                lookback_hours=args.window_lookback,
                start_date=args.start_date,
                end_date=args.end_date,
                output_path=args.output_dir / f"event_feature_window_DST+{args.window_horizon}_L{args.window_lookback}.png",
            )
        if args.window_horizon in range(1, 7):
            plot_continuous_vs_event_window(
                continuous_variant=args.continuous_variant,
                event_feature_key=args.event_feature_key,
                horizon_hours=args.window_horizon,
                start_date=args.start_date,
                end_date=args.end_date,
                output_path=args.output_dir / f"continuous_vs_event_DST+{args.window_horizon}.png",
            )

    print(f"Output directory: {args.output_dir}")
    if not continuous_summary.empty:
        print(f"Continuous rows summarized: {len(continuous_summary)}")
    else:
        print("Continuous summary: no matching CSV files found.")
    if not event_summary.empty:
        print(f"Event rows summarized: {len(event_summary)}")
    else:
        print("Event summary: no matching CSV files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
