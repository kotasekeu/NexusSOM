"""
plot_pareto_evolution.py — visualize Pareto front evolution from EA run.

Usage:
  python app/tools/plot_pareto_evolution.py <results_dir>
  python app/tools/plot_pareto_evolution.py <results_dir> --output my_plot.png
"""

import argparse
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def load_pareto_csv(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "pareto_front.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: pareto_front.csv not found in {results_dir}")

    df = pd.read_csv(path)
    df["generation"] = df["generation"].astype(int)
    for col in ("raw_mqe_ratio", "raw_te", "dead_ratio", "constraint_violation",
                "map_m", "map_n", "duration"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["constraint_violation"] = df["constraint_violation"].fillna(0.0)
    df["is_feasible"] = ~(df["is_penalized"].astype(str).str.lower() == "true")
    df["map_area"] = df["map_m"] * df["map_n"]
    return df


def _gen_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in sorted(df["generation"].unique()):
        gdf = df[df["generation"] == g]
        rows.append({
            "gen":          g,
            "size":         len(gdf),
            "feasible":     gdf["is_feasible"].sum(),
            "best_ratio":   gdf["raw_mqe_ratio"].min(),
            "median_ratio": gdf["raw_mqe_ratio"].median(),
            "worst_ratio":  gdf["raw_mqe_ratio"].max(),
            "mean_cv":      gdf["constraint_violation"].mean(),
        })
    return pd.DataFrame(rows)


def plot_evolution(df: pd.DataFrame, results_dir: str, output_path: str = None):
    generations = sorted(df["generation"].unique())
    n_gens = len(generations)
    stats = _gen_stats(df)

    cmap = cm.viridis
    norm = plt.Normalize(vmin=min(generations), vmax=max(generations))

    def gen_color(g):
        return cmap(norm(g))

    def gen_alpha(g):
        idx = generations.index(g)
        return 0.35 + 0.65 * (idx / max(n_gens - 1, 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Pareto Front Evolution — {os.path.basename(os.path.normpath(results_dir))}",
        fontsize=13, y=0.99
    )

    # ------------------------------------------------------------------ #
    # Panel 1: Archive size + feasible count per generation               #
    # ------------------------------------------------------------------ #
    ax1 = axes[0, 0]
    ax1b = ax1.twinx()

    ax1.bar(stats["gen"], stats["size"], alpha=0.35, color="steelblue", zorder=2)
    ax1b.plot(stats["gen"], stats["feasible"], "o-", color="green",
              linewidth=2, markersize=7, zorder=3)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Archive size", color="steelblue")
    ax1b.set_ylabel("Feasible solutions", color="green")
    ax1b.set_ylim(bottom=0)
    ax1.set_title("Archive Size & Feasibility per Generation")
    ax1.set_xticks(generations)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1b.tick_params(axis="y", labelcolor="green")

    handles = [
        Line2D([0], [0], color="steelblue", alpha=0.6, linewidth=8, label="Archive size"),
        Line2D([0], [0], color="green", marker="o", linewidth=2, label="Feasible count"),
    ]
    ax1.legend(handles=handles, loc="upper left", fontsize=8)

    # ------------------------------------------------------------------ #
    # Panel 2: Quality (mqe ratio) evolution                              #
    # ------------------------------------------------------------------ #
    ax2 = axes[0, 1]

    ax2.fill_between(stats["gen"], stats["worst_ratio"], stats["best_ratio"],
                     alpha=0.12, color="royalblue")
    ax2.plot(stats["gen"], stats["best_ratio"],   "o-",  color="royalblue",
             linewidth=2, markersize=7, label="Best ratio")
    ax2.plot(stats["gen"], stats["median_ratio"], "s--", color="steelblue",
             linewidth=1.5, alpha=0.8, label="Median ratio")
    ax2.plot(stats["gen"], stats["worst_ratio"],  "^:",  color="lightsteelblue",
             linewidth=1.2, alpha=0.7, label="Worst ratio")
    ax2.axhline(1.0, color="red", linestyle=":", linewidth=1.2, alpha=0.5, label="Ratio = 1 (no improvement)")

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("raw_mqe_improvement_ratio  (↓ better)")
    ax2.set_title("MQE Quality Evolution")
    ax2.set_xticks(generations)
    ax2.legend(fontsize=8)

    # ------------------------------------------------------------------ #
    # Panel 3: Pareto scatter mqe vs topographic error                    #
    # ------------------------------------------------------------------ #
    ax3 = axes[1, 0]
    _scatter_panel(ax3, df, generations, gen_color, gen_alpha,
                   xcol="raw_mqe_ratio", ycol="raw_te",
                   xlabel="raw_mqe_ratio  (↓ better)",
                   ylabel="topographic_error  (↓ better)",
                   title="Pareto Front: MQE ratio vs Topographic Error")
    _add_colorbar(fig, ax3, cmap, norm, generations)

    # ------------------------------------------------------------------ #
    # Panel 4: Pareto scatter mqe vs dead ratio                           #
    # ------------------------------------------------------------------ #
    ax4 = axes[1, 1]
    _scatter_panel(ax4, df, generations, gen_color, gen_alpha,
                   xcol="raw_mqe_ratio", ycol="dead_ratio",
                   xlabel="raw_mqe_ratio  (↓ better)",
                   ylabel="dead_neuron_ratio  (↓ better)",
                   title="Pareto Front: MQE ratio vs Dead Neuron Ratio")
    _add_colorbar(fig, ax4, cmap, norm, generations)

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(results_dir, "pareto_evolution.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def _scatter_panel(ax, df, generations, gen_color, gen_alpha, xcol, ycol,
                   xlabel, ylabel, title):
    for g in generations:
        gdf = df[df["generation"] == g]
        color  = gen_color(g)
        alpha  = gen_alpha(g)
        size_f = 55 + 25 * (generations.index(g) / max(len(generations) - 1, 1))

        feasible   = gdf[gdf["is_feasible"]]
        infeasible = gdf[~gdf["is_feasible"]]

        if len(feasible):
            ax.scatter(feasible[xcol], feasible[ycol],
                       c=[color], s=size_f, alpha=alpha,
                       marker="o", edgecolors="none", zorder=3)
        if len(infeasible):
            ax.scatter(infeasible[xcol], infeasible[ycol],
                       c=[color], s=size_f * 0.6, alpha=alpha * 0.55,
                       marker="x", linewidths=1.5, zorder=2)

    # Highlight final generation with a ring
    final_gdf = df[df["generation"] == generations[-1]]
    ax.scatter(final_gdf[xcol], final_gdf[ycol],
               s=110, facecolors="none", edgecolors="black",
               linewidths=1.2, zorder=4, label="Final archive")

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label="Feasible"),
        Line2D([0], [0], marker="x", color="gray", markersize=8,
               linewidth=1.5, label="Infeasible"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="black", markersize=10, label="Final archive"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")


def _add_colorbar(fig, ax, cmap, norm, generations):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Generation", fontsize=8)
    cbar.set_ticks(generations)


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto front evolution from EA run.")
    parser.add_argument("results_dir", help="Path to EA results directory")
    parser.add_argument("--output", "-o",
                        help="Output image path (default: results_dir/pareto_evolution.png)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit(f"ERROR: directory not found: {args.results_dir}")

    df = load_pareto_csv(args.results_dir)
    print(f"Loaded {len(df)} rows, {df['generation'].nunique()} generations, "
          f"{df['uid'].nunique()} unique solutions.")
    plot_evolution(df, args.results_dir, args.output)


if __name__ == "__main__":
    main()
