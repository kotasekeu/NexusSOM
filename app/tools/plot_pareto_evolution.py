"""
plot_pareto_evolution.py — visualize Pareto front evolution from EA run.

Usage:
  python app/tools/plot_pareto_evolution.py <results_dir>
  python app/tools/plot_pareto_evolution.py <results_dir> --output my_plot.png
  python app/tools/plot_pareto_evolution.py <results_dir> --no3d   # skip 3D plot
"""

import argparse
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd


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

    if df["raw_mqe_ratio"].isna().all():
        sys.exit(
            "ERROR: raw_mqe_ratio is NaN for all rows.\n"
            "This run was likely generated with save_checkpoints=false or "
            "mqe_evaluations_per_run too low (e.g. 3).\n"
            "Use a config with save_checkpoints=true and mqe_evaluations_per_run>=200."
        )

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


def plot_evolution(df: pd.DataFrame, results_dir: str, output_path: str = None,
                   fixed_range: bool = False):
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

    # Layout: 2 panels top row + 3 panels bottom row via 6-column GridSpec
    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.52,
                  top=0.93, bottom=0.08, left=0.06, right=0.97)

    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 3:])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax5 = fig.add_subplot(gs[1, 4:])

    fig.suptitle(
        f"Pareto Front Evolution — {os.path.basename(os.path.normpath(results_dir))}",
        fontsize=14, y=1.01,
    )

    # ------------------------------------------------------------------ #
    # Panel 1: Archive size — stacked if any infeasible, plain otherwise   #
    # ------------------------------------------------------------------ #
    has_any_infeasible = (~df["is_feasible"]).any()
    if has_any_infeasible:
        infeasible_counts = stats["size"] - stats["feasible"]
        ax1.bar(stats["gen"], stats["feasible"],
                label="Feasible", color="steelblue", alpha=0.85, zorder=2)
        ax1.bar(stats["gen"], infeasible_counts, bottom=stats["feasible"],
                label="Infeasible", color="tomato", alpha=0.85, zorder=2)
        ax1.legend(fontsize=9)
    else:
        ax1.bar(stats["gen"], stats["size"], color="steelblue", alpha=0.75, zorder=2)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Archive size")
    ax1.set_title("Archive Size per Generation")
    ax1.set_xticks(generations)

    # ------------------------------------------------------------------ #
    # Panel 2: MQE quality — Y axis clipped to actual data range          #
    # ------------------------------------------------------------------ #
    y_lo = stats["best_ratio"].min() * 0.97
    y_hi = stats["worst_ratio"].max() * 1.03

    ax2.fill_between(stats["gen"], stats["worst_ratio"], stats["best_ratio"],
                     alpha=0.12, color="royalblue")
    ax2.plot(stats["gen"], stats["best_ratio"],   "o-",  color="royalblue",
             linewidth=2, markersize=7, label="Best ratio")
    ax2.plot(stats["gen"], stats["median_ratio"], "s--", color="steelblue",
             linewidth=1.5, alpha=0.8, label="Median ratio")
    ax2.plot(stats["gen"], stats["worst_ratio"],  "^:",  color="lightsteelblue",
             linewidth=1.2, alpha=0.7, label="Worst ratio")

    ax2.set_ylim(y_lo, y_hi)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("raw_mqe_improvement_ratio  (↓ better)")
    ax2.set_title("MQE Quality Evolution")
    ax2.set_xticks(generations)
    ax2.legend(fontsize=8)
    ax2.text(0.98, 0.03,
             "Ratio = 1.0 means no improvement over random init",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=7, color="gray", style="italic")

    # ------------------------------------------------------------------ #
    # Panels 3–5: all 3 pairwise 2D projections                           #
    # ------------------------------------------------------------------ #
    _scatter_panel(ax3, df, generations, gen_color, gen_alpha,
                   xcol="raw_mqe_ratio", ycol="raw_te",
                   xlabel="MQE ratio  (→ better)",
                   ylabel="Topographic error  (↑ better)",
                   title="MQE ratio vs Topographic Error",
                   fixed_range=fixed_range)
    _add_colorbar(fig, ax3, cmap, norm, generations)

    _scatter_panel(ax4, df, generations, gen_color, gen_alpha,
                   xcol="raw_mqe_ratio", ycol="dead_ratio",
                   xlabel="MQE ratio  (→ better)",
                   ylabel="Dead neuron ratio  (↑ better)",
                   title="MQE ratio vs Dead Neuron Ratio",
                   fixed_range=fixed_range)
    _add_colorbar(fig, ax4, cmap, norm, generations)

    _scatter_panel(ax5, df, generations, gen_color, gen_alpha,
                   xcol="raw_te", ycol="dead_ratio",
                   xlabel="Topographic error  (→ better)",
                   ylabel="Dead neuron ratio  (↑ better)",
                   title="Topographic Error vs Dead Neuron Ratio",
                   fixed_range=fixed_range)
    _add_colorbar(fig, ax5, cmap, norm, generations)

    if output_path is None:
        output_path = os.path.join(results_dir, "pareto_evolution.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def _scatter_panel(ax, df, generations, gen_color, gen_alpha, xcol, ycol,
                   xlabel, ylabel, title, fixed_range: bool = False):
    has_infeasible = False
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
            has_infeasible = True
            ax.scatter(infeasible[xcol], infeasible[ycol],
                       c=[color], s=size_f * 0.6, alpha=alpha * 0.55,
                       marker="x", linewidths=1.5, zorder=2)

    # Final archive — red star, high contrast against viridis palette
    final_gdf = df[df["generation"] == generations[-1]]
    ax.scatter(final_gdf[xcol], final_gdf[ycol],
               s=220, c="red", marker="*", zorder=5)

    ax.invert_xaxis()
    ax.invert_yaxis()
    if fixed_range:
        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(1.0, 0.0)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label="Feasible"),
    ]
    if has_infeasible:
        legend_elements.append(
            Line2D([0], [0], marker="x", color="gray", markersize=8,
                   linewidth=1.5, label="Infeasible")
        )
    legend_elements.append(
        Line2D([0], [0], marker="*", color="red", markersize=11,
               label=f"Final archive (gen {generations[-1]})")
    )
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")


def _add_colorbar(fig, ax, cmap, norm, generations):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Generation", fontsize=8)
    cbar.set_ticks(generations)


def export_csv(df: pd.DataFrame, stats: pd.DataFrame, results_dir: str, csv_path: str = None):
    if csv_path is None:
        csv_path = os.path.join(results_dir, "pareto_evolution_stats.csv")

    stats.to_csv(csv_path, index=False)
    print(f"Saved stats: {csv_path}")

    param_cols = [c for c in df.columns if c not in (
        "generation", "uid", "is_penalized", "is_feasible", "map_area",
        "raw_mqe_ratio", "raw_te", "dead_ratio", "constraint_violation",
        "map_m", "map_n", "duration",
    )]
    obj_cols = ["raw_mqe_ratio", "raw_te", "dead_ratio", "constraint_violation"]

    rows = []
    for g in sorted(df["generation"].unique()):
        gdf = df[df["generation"] == g]
        for col in param_cols + obj_cols:
            if col not in gdf.columns:
                continue
            numeric = pd.to_numeric(gdf[col], errors="coerce").dropna()
            if numeric.empty:
                continue
            rows.append({
                "generation": g,
                "param": col,
                "mean":   round(numeric.mean(), 6),
                "median": round(numeric.median(), 6),
                "std":    round(numeric.std(), 6),
                "min":    round(numeric.min(), 6),
                "max":    round(numeric.max(), 6),
                "p25":    round(numeric.quantile(0.25), 6),
                "p75":    round(numeric.quantile(0.75), 6),
            })

    dist_path = csv_path.replace(".csv", "_param_dist.csv")
    pd.DataFrame(rows).to_csv(dist_path, index=False)
    print(f"Saved param distributions: {dist_path}")


def plot_pareto_3d(df: pd.DataFrame, results_dir: str, output_path: str = None,
                   guide_lines: bool = False, space_grid: int = 0,
                   lattice: bool = False, elev: float = 22, azim: float = 225,
                   fixed_range: bool = False):
    """3D scatter with shadow projections on all three walls.
    guide_lines=True  — lines from each point to all three walls.
    space_grid=N      — set N equal divisions on each axis (ticks); 0 = matplotlib auto.
    lattice=True      — draw dashed 3D lattice; uses space_grid divisions (default 5)."""
    generations = sorted(df["generation"].unique())
    cmap = cm.viridis
    norm = plt.Normalize(vmin=min(generations), vmax=max(generations))

    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection="3d")

    final_gen = generations[-1]
    final_uids = set(df[df["generation"] == final_gen]["uid"])

    xs_all = df["raw_mqe_ratio"].values
    ys_all = df["raw_te"].values
    zs_all = df["dead_ratio"].values

    pad_x = (xs_all.max() - xs_all.min()) * 0.08
    pad_y = (ys_all.max() - ys_all.min()) * 0.08
    pad_z = max((zs_all.max() - zs_all.min()) * 0.08, zs_all.max() * 0.02)

    # Axes are inverted: low values (good) appear top-right-front.
    # With inverted X and Y, the background panes are on the MIN side:
    #   x=x_MIN  → TE×Dead projection (back wall after inversion)
    #   y=y_MIN  → MQE×Dead projection (back wall after inversion)
    #   z=z_MAX  → MQE×TE projection (ceiling = worst Z, visually behind data)
    x_wall = xs_all.min() - pad_x        # back face after x-inversion → TE×Dead
    y_wall = ys_all.min() - pad_y        # back face after y-inversion → MQE×Dead
    z_floor = zs_all.max() + pad_z       # ceiling (worst dead ratio)  → MQE×TE

    has_infeasible = False
    for g in generations:
        gdf = df[df["generation"] == g]
        color  = cmap(norm(g))
        idx    = generations.index(g)
        alpha  = 0.35 + 0.65 * (idx / max(len(generations) - 1, 1))
        size_f = 55 + 35 * (idx / max(len(generations) - 1, 1))

        feasible   = gdf[gdf["is_feasible"]]
        infeasible = gdf[~gdf["is_feasible"]]

        xs = gdf["raw_mqe_ratio"].values
        ys = gdf["raw_te"].values
        zs = gdf["dead_ratio"].values
        colors_g = [color] * len(gdf)

        # Shadow projections on the three background panes
        ax.scatter(xs, ys, z_floor,                           # floor  → MQE×TE
                   c=colors_g, s=size_f * 0.28, alpha=0.25,
                   marker="o", edgecolors="none", zorder=1)
        ax.scatter(xs, np.full_like(ys, y_wall), zs,          # y_MAX  → MQE×Dead
                   c=colors_g, s=size_f * 0.28, alpha=0.25,
                   marker="o", edgecolors="none", zorder=1)
        ax.scatter(np.full_like(xs, x_wall), ys, zs,          # x_MAX  → TE×Dead
                   c=colors_g, s=size_f * 0.28, alpha=0.25,
                   marker="o", edgecolors="none", zorder=1)

        if len(feasible):
            ax.scatter(
                feasible["raw_mqe_ratio"],
                feasible["raw_te"],
                feasible["dead_ratio"],
                c=[color], s=size_f, alpha=alpha,
                marker="o", edgecolors="none", zorder=3,
            )
        if len(infeasible):
            has_infeasible = True
            ax.scatter(
                infeasible["raw_mqe_ratio"],
                infeasible["raw_te"],
                infeasible["dead_ratio"],
                c=[color], s=size_f * 0.5, alpha=alpha * 0.5,
                marker="x", linewidths=1.5, zorder=2,
            )

    # Final archive — red stars
    final_df = df[df["uid"].isin(final_uids) & (df["generation"] == final_gen)]
    ax.scatter(
        final_df["raw_mqe_ratio"],
        final_df["raw_te"],
        final_df["dead_ratio"],
        s=280, c="red", marker="*", zorder=6,
    )

    if guide_lines:
        for xi, yi, zi in zip(final_df["raw_mqe_ratio"],
                              final_df["raw_te"],
                              final_df["dead_ratio"]):
            ax.plot([xi, xi], [yi, yi], [zi, z_floor], color="red", alpha=0.12, lw=0.7)
            ax.plot([xi, xi], [yi, y_wall], [zi, zi],  color="red", alpha=0.12, lw=0.7)
            ax.plot([xi, x_wall], [yi, yi], [zi, zi],  color="red", alpha=0.12, lw=0.7)

    # Orthographic projection removes perspective distortion — shadows align
    # exactly with the actual point positions when viewed straight-on.
    ax.set_proj_type('ortho')
    ax.view_init(elev=elev, azim=azim)

    if fixed_range:
        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(1.0, 0.0)
        ax.set_zlim(1.0, 0.0)
        x0, x1 = 0.0, 1.0
        y0, y1 = 0.0, 1.0
        z0, z1 = 0.0, 1.0
    else:
        ax.set_xlim(x_wall, xs_all.max() + pad_x)
        ax.set_ylim(y_wall, ys_all.max() + pad_y)
        ax.set_zlim(zs_all.min() - pad_z, z_floor)
        x0, x1 = x_wall, xs_all.max() + pad_x
        y0, y1 = y_wall, ys_all.max() + pad_y
        z0, z1 = zs_all.min() - pad_z, z_floor
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()

    # Uniform axis divisions — independent of whether lattice lines are drawn
    if space_grid > 0:
        xt = np.linspace(x0, x1, space_grid + 1)
        yt = np.linspace(y0, y1, space_grid + 1)
        zt = np.linspace(z0, z1, space_grid + 1)
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_zticks(zt)

    # 3D lattice — dashed lines at every tick intersection
    if lattice:
        n = space_grid if space_grid > 0 else 5
        xt = np.linspace(x0, x1, n + 1)
        yt = np.linspace(y0, y1, n + 1)
        zt = np.linspace(z0, z1, n + 1)
        gs = dict(color=(0.45, 0.45, 0.45, 0.22), lw=0.5, linestyle='--', zorder=0)
        for yi in yt:
            for zi in zt:
                ax.plot([x0, x1], [yi, yi], [zi, zi], **gs)
        for xi in xt:
            for zi in zt:
                ax.plot([xi, xi], [y0, y1], [zi, zi], **gs)
        for xi in xt:
            for yi in yt:
                ax.plot([xi, xi], [yi, yi], [z0, z1], **gs)

    # Light gray bounding box — panes + grid lines
    gray_pane = (0.93, 0.93, 0.93, 0.35)
    gray_edge = (0.60, 0.60, 0.60, 0.70)
    gray_grid = (0.72, 0.72, 0.72, 0.55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(gray_pane)
        axis.pane.set_edgecolor(gray_edge)
        axis._axinfo['grid']['color'] = gray_grid
    ax.grid(True)

    ax.set_xlabel("MQE ratio  (↓ better)", fontsize=11, labelpad=10)
    ax.set_ylabel("Topographic error  (↓ better)", fontsize=11, labelpad=10)
    ax.set_zlabel("Dead neuron ratio  (↓ better)", fontsize=11, labelpad=10)
    ax.set_title(
        f"Pareto Front 3D — {os.path.basename(os.path.normpath(results_dir))}\n"
        f"({len(df['uid'].unique())} solutions across {len(generations)} generations)\n"
        f"Orthographic — floor=MQE×TE  back wall=MQE×Dead  right wall=TE×Dead",
        fontsize=12, pad=16,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08)
    cbar.set_label("Generation", fontsize=10)
    cbar.set_ticks(generations)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=9, label="Feasible"),
    ]
    if has_infeasible:
        legend_elements.append(
            Line2D([0], [0], marker="x", color="gray", markersize=9,
                   linewidth=1.5, label="Infeasible")
        )
    legend_elements.append(
        Line2D([0], [0], marker="*", color="red", markersize=12,
               label=f"Final archive (gen {final_gen})")
    )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    gen_counts = df.groupby("generation")["uid"].count()
    info = "  ".join(f"G{g}:{gen_counts[g]}" for g in generations)
    fig.text(0.5, 0.01, f"Solutions per generation:  {info}",
             ha="center", fontsize=9, color="gray")

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(results_dir, "pareto_3d.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved 3D plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto front evolution from EA run.")
    parser.add_argument("results_dir", help="Path to EA results directory")
    parser.add_argument("--output", "-o",
                        help="Output image path (default: results_dir/pareto_evolution.png)")
    parser.add_argument("--output3d",
                        help="3D plot output path (default: results_dir/pareto_3d.png)")
    parser.add_argument("--no3d", action="store_true",
                        help="Skip the 3D scatter plot")
    parser.add_argument("--guide-lines", action="store_true", default=False,
                        help="Draw guide lines from each point to all three walls in 3D plot")
    parser.add_argument("--space-grid", type=int, default=0, metavar="N",
                        help="Set N equal divisions on each 3D axis (0 = matplotlib auto)")
    parser.add_argument("--lattice", action="store_true", default=False,
                        help="Draw dashed 3D lattice lines at tick intersections (uses --space-grid N, default 5)")
    parser.add_argument("--elev", type=float, default=22,
                        help="3D view elevation angle in degrees (default: 22)")
    parser.add_argument("--azim", type=float, default=225,
                        help="3D view azimuth angle in degrees (default: 225)")
    parser.add_argument("--fixed-range", action="store_true", default=False,
                        help="Fix all axes to [0, 1] for cross-run comparison")
    parser.add_argument("--csv", "-c", nargs="?", const=True, default=False,
                        help="Export plot data to CSV (optional custom path)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit(f"ERROR: directory not found: {args.results_dir}")

    df = load_pareto_csv(args.results_dir)
    print(f"Loaded {len(df)} rows, {df['generation'].nunique()} generations, "
          f"{df['uid'].nunique()} unique solutions.")

    stats = _gen_stats(df)

    if args.csv is not False:
        csv_path = args.csv if isinstance(args.csv, str) else None
        export_csv(df, stats, args.results_dir, csv_path)

    plot_evolution(df, args.results_dir, args.output,
                   fixed_range=args.fixed_range)

    if not args.no3d:
        plot_pareto_3d(df, args.results_dir, args.output3d,
                       guide_lines=args.guide_lines,
                       space_grid=args.space_grid,
                       lattice=args.lattice,
                       elev=args.elev,
                       azim=args.azim,
                       fixed_range=args.fixed_range)


if __name__ == "__main__":
    main()
