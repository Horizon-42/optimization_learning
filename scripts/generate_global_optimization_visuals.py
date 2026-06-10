from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping, differential_evolution, direct, dual_annealing, minimize, shgo


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)

BOUNDS = [(-512.0, 512.0), (-512.0, 512.0)]

mpl.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 240,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def eggholder(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(
        -(x[1] + 47.0) * np.sin(np.sqrt(abs(x[0] / 2.0 + x[1] + 47.0)))
        - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
    )


def eggholder_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        -(y + 47.0) * np.sin(np.sqrt(np.abs(x / 2.0 + y + 47.0)))
        - x * np.sin(np.sqrt(np.abs(x - (y + 47.0))))
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def run_global_optimizers() -> dict[str, object]:
    return {
        "differential_evolution": differential_evolution(
            eggholder,
            BOUNDS,
            maxiter=35,
            popsize=8,
            seed=1,
            polish=True,
            tol=1e-6,
        ),
        "dual_annealing": dual_annealing(
            eggholder,
            BOUNDS,
            maxiter=120,
            seed=2,
        ),
        "shgo": shgo(
            eggholder,
            BOUNDS,
            n=96,
            iters=2,
            sampling_method="sobol",
        ),
        "direct": direct(
            eggholder,
            BOUNDS,
            maxfun=1200,
            locally_biased=True,
        ),
        "basinhopping": basinhopping(
            eggholder,
            np.array([0.0, 0.0]),
            niter=80,
            seed=4,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": BOUNDS},
        ),
    }


def plot_eggholder_landscape(results: dict[str, object]) -> None:
    x = np.linspace(-512.0, 512.0, 520)
    y = np.linspace(-512.0, 512.0, 520)
    xx, yy = np.meshgrid(x, y)
    zz = eggholder_grid(xx, yy)

    fig, ax = plt.subplots(figsize=(8.9, 7.25))
    im = ax.imshow(
        zz,
        extent=[-512, 512, -512, 512],
        origin="lower",
        cmap="magma_r",
        interpolation="bilinear",
        aspect="equal",
    )
    contour = ax.contour(xx, yy, zz, levels=np.linspace(-900, 600, 16), colors="white", linewidths=0.45, alpha=0.32)
    ax.clabel(contour, contour.levels[::4], inline=True, fontsize=7, fmt="%.0f")

    shgo_result = results["shgo"]
    if getattr(shgo_result, "xl", None) is not None:
        local_minima = np.asarray(shgo_result.xl)
        ax.scatter(local_minima[:, 0], local_minima[:, 1], s=18, color="#ef4444", alpha=0.75, linewidth=0, label="SHGO local minima")

    markers = {
        "differential_evolution": ("o", "#22d3ee", "differential evolution"),
        "dual_annealing": ("s", "#ffffff", "dual annealing"),
        "shgo": ("X", "#dc2626", "SHGO best"),
        "direct": ("D", "#22c55e", "DIRECT"),
        "basinhopping": ("^", "#f59e0b", "basinhopping"),
    }
    for name, (marker, color, label) in markers.items():
        point = np.asarray(results[name].x)
        ax.scatter(point[0], point[1], s=120, marker=marker, color=color, edgecolor="#111827", linewidth=0.75, label=label, zorder=5)

    ax.set_title("Global optimizers search different basins of the eggholder surface")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(-535, 535)
    ax.set_ylim(-535, 535)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="lower left", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("$f(x_0,x_1)$")
    ax.text(
        0.98,
        0.03,
        "Lower values are lighter.\nMany local valleys compete with the global one.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "global_eggholder_landscape")
    plt.close(fig)


def plot_global_optimizer_comparison(results: dict[str, object]) -> None:
    names = ["SHGO", "dual annealing", "differential evolution", "DIRECT", "basinhopping"]
    keys = ["shgo", "dual_annealing", "differential_evolution", "direct", "basinhopping"]
    funs = np.array([float(results[key].fun) for key in keys])
    nfev = np.array([float(getattr(results[key], "nfev", np.nan)) for key in keys])
    colors = ["#dc2626", "#7c3aed", "#0891b2", "#16a34a", "#b45309"]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9.2, 6.7), gridspec_kw={"height_ratios": [1.15, 1.0]})
    positions = np.arange(len(names))

    ax_top.bar(positions, funs, color=colors, alpha=0.88)
    ax_top.axhline(-959.64, color="#111827", linewidth=1.1, linestyle="--")
    ax_top.set_title("Same surface, different global-search tradeoffs")
    ax_top.set_ylabel("best objective value")
    ax_top.set_xticks(positions)
    ax_top.set_xticklabels([])
    ax_top.set_ylim(-1025, 80)
    ax_top.grid(True, axis="y", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_top.text(
        0.98,
        0.08,
        "known eggholder global value ~= -959.6",
        transform=ax_top.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    for pos, value in zip(positions, funs):
        label_color = "white" if value < -180 else "#111827"
        ax_top.text(pos, value / 2.0, f"{value:.1f}", ha="center", va="center", fontsize=9, color=label_color, fontweight=650)

    ax_bottom.bar(positions, nfev, color=colors, alpha=0.78)
    ax_bottom.set_yscale("log")
    ax_bottom.set_ylabel("function evaluations")
    ax_bottom.set_xticks(positions)
    ax_bottom.set_xticklabels(names, rotation=18, ha="right")
    ax_bottom.grid(True, axis="y", which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    for pos, value in zip(positions, nfev):
        ax_bottom.text(pos, value * 1.08, f"{int(value)}", ha="center", va="bottom", fontsize=8, color="#111827")

    save_figure(fig, "global_optimizer_comparison")
    plt.close(fig)


def plot_local_traps() -> None:
    rng = np.random.default_rng(7)
    starts = rng.uniform(-512.0, 512.0, size=(48, 2))
    finishes = []
    values = []

    for start in starts:
        result = minimize(eggholder, start, method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": 120})
        finishes.append(result.x)
        values.append(result.fun)

    finishes_array = np.asarray(finishes)
    values_array = np.asarray(values)

    x = np.linspace(-512.0, 512.0, 430)
    y = np.linspace(-512.0, 512.0, 430)
    xx, yy = np.meshgrid(x, y)
    zz = eggholder_grid(xx, yy)

    fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(10.2, 4.9), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax_map.imshow(
        zz,
        extent=[-512, 512, -512, 512],
        origin="lower",
        cmap="Greys",
        interpolation="bilinear",
        aspect="equal",
        alpha=0.75,
    )
    for start, finish in zip(starts, finishes_array):
        ax_map.plot([start[0], finish[0]], [start[1], finish[1]], color="#64748b", alpha=0.18, linewidth=0.7)
    scatter = ax_map.scatter(finishes_array[:, 0], finishes_array[:, 1], c=values_array, cmap="viridis", s=38, edgecolor="white", linewidth=0.35, zorder=3)
    ax_map.scatter(starts[:, 0], starts[:, 1], s=10, color="#111827", alpha=0.25, linewidth=0)
    ax_map.set_title("Local minimization falls into nearby basins")
    ax_map.set_xlabel("$x_0$")
    ax_map.set_ylabel("$x_1$")
    ax_map.set_xlim(-512, 512)
    ax_map.set_ylim(-512, 512)

    ax_hist.hist(values_array, bins=14, color="#2563eb", edgecolor="white", alpha=0.86)
    ax_hist.axvline(-959.64, color="#111827", linestyle="--", linewidth=1.2, label="global value")
    ax_hist.set_title("Distribution from random local starts")
    ax_hist.set_xlabel("final objective value")
    ax_hist.set_ylabel("number of starts")
    ax_hist.grid(True, axis="y", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_hist.legend(frameon=False, loc="upper left")
    ax_hist.text(
        0.98,
        0.94,
        f"best local run: {values_array.min():.1f}\nmedian local run: {np.median(values_array):.1f}",
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    cbar = fig.colorbar(scatter, ax=ax_map, orientation="horizontal", fraction=0.052, pad=0.08)
    cbar.set_label("final $f(x)$")
    save_figure(fig, "global_local_traps")
    plt.close(fig)


def main() -> None:
    results = run_global_optimizers()
    plot_eggholder_landscape(results)
    plot_global_optimizer_comparison(results)
    plot_local_traps()
    print(f"Wrote global optimization visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
