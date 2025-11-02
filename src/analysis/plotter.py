import os
from typing import Optional

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gmean

sns.set_theme(style="whitegrid", palette="muted")
PLOT_DIR = "results/plots"
BASELINE_OPT = "PostgreSQL_Default"  # Define our main baseline


def load_data(results_file: str) -> Optional[pd.DataFrame]:
    """Loads the CSV results file."""
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return None

    print(f"Loading results from {results_file}")
    df = pd.read_csv(results_file)

    # Ensure correct dtypes
    df["inference_time_ms"] = pd.to_numeric(df["inference_time_ms"])
    df["actual_latency_ms"] = pd.to_numeric(df["actual_latency_ms"])
    df["end_to_end_time_ms"] = pd.to_numeric(df["end_to_end_time_ms"])

    return df


def plot_geometric_mean_latency(df: pd.DataFrame, output_path: str):
    """
    Plots the geometric mean of the actual query latency.
    This is the standard for benchmarks to avoid skew from outliers.
    """
    print("Generating Plot 1: Geometric Mean Latency...")
    # Add a small epsilon to avoid gmean(0)
    gmean_latency = df.groupby("optimizer_name")["actual_latency_ms"].apply(
        lambda x: gmean(x + 1e-9)
    )
    gmean_latency = gmean_latency.sort_values().reset_index()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=gmean_latency, x="optimizer_name", y="actual_latency_ms")
    ax.set_title(
        "Overall Performance (Geometric Mean of Query Latency)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_ylabel("Geometric Mean Latency (ms) - Lower is Better")
    ax.set_xlabel("Optimizer")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_end_to_end_stacked(df: pd.DataFrame, output_path: str):
    """
    Plots the mean End-to-End time, stacked to show
    Inference vs. Execution. This is the paper's key chart.
    """
    print("Generating Plot 2: End-to-End Time (Stacked Bar)...")
    # Get the mean of the two time components
    df_agg = df.groupby("optimizer_name")[
        ["inference_time_ms", "actual_latency_ms"]
    ].mean()

    # Sort by total time
    df_agg["total"] = df_agg["inference_time_ms"] + df_agg["actual_latency_ms"]
    df_agg = df_agg.sort_values("total", ascending=False)
    df_agg = df_agg.drop(columns="total")

    # Plot stacked bar
    ax = df_agg.plot(
        kind="barh",  # Horizontal is easier to read
        stacked=True,
        figsize=(12, 7),
        colormap="viridis",
    )

    ax.set_title("Mean End-to-End Execution Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Mean Time (ms) - Lower is Better")
    ax.set_ylabel("Optimizer")
    ax.legend(["Inference Time", "Execution Latency"], loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_per_query_comparison(df: pd.DataFrame, output_path: str):
    """
    Plots a scatter plot comparing each optimizer's
    end-to-end time against the baseline.
    """
    print("Generating Plot 3: Per-Query Scatter Comparison...")

    df_pivot = df.pivot(
        index="query_name", columns="optimizer_name", values="end_to_end_time_ms"
    )

    lqo_optimizers = [
        opt for opt in df["optimizer_name"].unique() if opt != BASELINE_OPT
    ]

    for lqo_name in lqo_optimizers:
        plt.figure(figsize=(8, 8))

        ax = sns.scatterplot(data=df_pivot, x=BASELINE_OPT, y=lqo_name, alpha=0.7, s=50)

        # Add parity line (x=y)
        min_val = min(df_pivot[BASELINE_OPT].min(), df_pivot[lqo_name].min()) * 0.9
        max_val = max(df_pivot[BASELINE_OPT].max(), df_pivot[lqo_name].max()) * 1.1
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Parity (x=y)")

        # Use log scale for better visibility
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        ax.set_title(
            f"Per-Query Performance: {lqo_name} vs. {BASELINE_OPT}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(f"{BASELINE_OPT} End-to-End Time (ms) - (Log Scale)")
        ax.set_ylabel(f"{lqo_name} End-to-End Time (ms) - (Log Scale)")
        ax.legend()
        ax.grid(which="both", linestyle="--", linewidth=0.5)

        plot_name = os.path.join(
            os.path.dirname(output_path),
            f"{os.path.basename(output_path).split('.')[0]}_{lqo_name}.png",
        )
        plt.tight_layout()
        plt.savefig(plot_name)
        plt.close()
        print(f"  Saved {plot_name}")


@click.command()
@click.argument("results_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--out_dir", default=PLOT_DIR, help="Directory to save plots.")
def main(results_file: str, out_dir: str):
    """
    Analyzes the results from an experiment CSV file
    and generates performance plots.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = load_data(results_file)
    if df is None:
        return

    # Generate all plots
    plot_geometric_mean_latency(df, os.path.join(out_dir, "1_gmean_latency.png"))

    plot_end_to_end_stacked(df, os.path.join(out_dir, "2_end_to_end_stacked.png"))

    plot_per_query_comparison(
        df, os.path.join(out_dir, "3_scatter_comparison")  # Base name
    )

    print("\n--- Analysis Complete ---")
    print(f"All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
