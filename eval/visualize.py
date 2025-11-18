# eval/visualize.py
"""
visualize.py

Minimal visualization of aggregated metrics for llm-eval-pro.

Usage (from project root):

    python -m eval.visualize

This will read outputs/aggregated_metrics.csv and write one PNG per metric.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def visualize_aggregated(path: Path) -> None:
    df = pd.read_csv(path)

    if "model" not in df.columns:
        raise ValueError("Expected a 'model' column in aggregated_metrics.csv")

    metric_cols = [c for c in df.columns if c != "model"]
    if not metric_cols:
        raise ValueError("No metric columns found in aggregated_metrics.csv")

    out_dir = path.parent

    for metric in metric_cols:
        plt.figure(figsize=(6, 4))
        plt.title(f"Model comparison – {metric}")
        plt.bar(df["model"], df[metric])
        plt.ylabel(metric)
        plt.xlabel("model")
        plt.tight_layout()

        out_path = out_dir / f"{metric}_comparison.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved chart: {out_path}")


def main() -> None:
    root = _project_root()
    aggregated = root / "outputs" / "aggregated_metrics.csv"
    if not aggregated.exists():
        raise SystemExit(
            f"{aggregated} not found. Run `python -m eval.compare_models` first."
        )
    visualize_aggregated(aggregated)


if __name__ == "__main__":
    main()
