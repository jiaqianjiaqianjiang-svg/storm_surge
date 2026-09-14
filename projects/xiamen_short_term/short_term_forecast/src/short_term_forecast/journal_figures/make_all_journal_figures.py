"""Unified entrypoint for all journal-quality short-term forecast figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import io_utils
from .plot_event_comparison import plot_event_comparison
from .plot_peak_analysis import plot_peak_analysis
from .plot_residual_diagnostics import plot_residual_diagnostics
from .plot_rolling_forecast import plot_rolling_forecast
from .plot_window_comparison import plot_window_metrics, plot_window_scatter, plot_window_timeseries, select_window_experiments


def _path_text(paths: dict[str, Path] | None, extension: str) -> str:
    if not paths:
        return ""
    value = paths.get(extension)
    return str(value) if value is not None else ""


def _manifest_row(
    figure_name: str,
    source_experiments: list[str],
    paths: dict[str, Path] | None,
    status: str = "ok",
    warning: str = "",
) -> dict[str, str]:
    return {
        "figure_name": figure_name,
        "source_experiments": ";".join(source_experiments),
        "output_png": _path_text(paths, "png"),
        "status": status,
        "warning": warning,
    }


def _safe_call(manifest_name: str, sources: list[str], rows: list[dict[str, str]], func, *args, **kwargs) -> None:
    try:
        paths = func(*args, **kwargs)
        if paths:
            rows.append(_manifest_row(manifest_name, sources, paths))
        else:
            rows.append(_manifest_row(manifest_name, sources, None, status="skipped", warning="missing required inputs"))
    except Exception as exc:  # keep batch figure generation alive on remote machines
        io_utils.warn(f"{manifest_name} failed: {exc}")
        rows.append(_manifest_row(manifest_name, sources, None, status="failed", warning=str(exc)))


def make_all_journal_figures(results_root: Path, output_dir: Path, language: str = "en") -> pd.DataFrame:
    experiments = io_utils.scan_results(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []

    window_experiments = list(select_window_experiments(experiments).values())
    window_sources = [exp.name for exp in window_experiments]
    _safe_call("figure_window_metrics", window_sources, manifest_rows, plot_window_metrics, window_experiments, output_dir, language=language)
    _safe_call("figure_window_timeseries", window_sources, manifest_rows, plot_window_timeseries, window_experiments, output_dir, language=language)
    _safe_call("figure_window_scatter", window_sources, manifest_rows, plot_window_scatter, window_experiments, output_dir, language=language)

    rolling_experiments = [exp for exp in experiments if "rolling_forecast" in exp.files]
    for experiment in rolling_experiments:
        csv_path = experiment.files["rolling_forecast"]
        stem = experiment.name.replace(" ", "_")
        _safe_call(
            f"figure_rolling_forecast_{stem}",
            [experiment.name],
            manifest_rows,
            plot_rolling_forecast,
            csv_path,
            output_dir,
            figure_name=f"figure_rolling_forecast_{stem}",
            language=language,
        )
        _safe_call(
            f"figure_peak_analysis_{stem}",
            [experiment.name],
            manifest_rows,
            plot_peak_analysis,
            csv_path,
            output_dir,
            figure_name=f"figure_peak_analysis_{stem}",
            language=language,
        )
        _safe_call(
            f"figure_residual_diagnostics_{stem}",
            [experiment.name],
            manifest_rows,
            plot_residual_diagnostics,
            csv_path,
            output_dir,
            figure_name=f"figure_residual_diagnostics_{stem}",
            language=language,
        )

    if len(rolling_experiments) >= 2:
        _safe_call(
            "figure_event_comparison",
            [exp.name for exp in rolling_experiments],
            manifest_rows,
            plot_event_comparison,
            [exp.files["rolling_forecast"] for exp in rolling_experiments],
            output_dir,
            language=language,
        )
    else:
        manifest_rows.append(
            _manifest_row(
                "figure_event_comparison",
                [exp.name for exp in rolling_experiments],
                None,
                status="skipped",
                warning="need at least two rolling forecast CSVs",
            )
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"[OUTPUT] figure_manifest.csv: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all journal-quality forecast figures from an existing results root.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_all_journal_figures(args.results_root, args.output_dir, language=args.language)


if __name__ == "__main__":
    main()
