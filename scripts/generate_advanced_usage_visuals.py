from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult, minimize


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)

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


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def custom_bowl(x: np.ndarray) -> float:
    shifted = np.array([x[0] - 0.8, x[1] + 0.35])
    hessian = np.array([[5.0, 1.25], [1.25, 1.35]])
    return float(0.5 * shifted @ hessian @ shifted)


def coordinate_search_minimizer(
    fun,
    x0,
    args=(),
    maxfev=None,
    stepsize=0.22,
    maxiter=120,
    callback=None,
    **options,
) -> OptimizeResult:
    del options
    bestx = np.array(x0, dtype=float)
    besty = float(fun(bestx, *args))
    funcalls = 1
    niter = 0
    current_step = float(stepsize)
    maxfev = np.inf if maxfev is None else maxfev

    while niter < maxiter and funcalls < maxfev and current_step > 1.0e-4:
        improved = False
        niter += 1
        for dim in range(bestx.size):
            for sign in (-1.0, 1.0):
                testx = bestx.copy()
                testx[dim] += sign * current_step
                testy = float(fun(testx, *args))
                funcalls += 1
                if testy < besty:
                    bestx = testx
                    besty = testy
                    improved = True
                    if callback is not None:
                        callback(bestx)
                if funcalls >= maxfev:
                    break
            if funcalls >= maxfev:
                break
        if not improved:
            current_step *= 0.5

    return OptimizeResult(
        x=bestx,
        fun=besty,
        success=current_step <= 1.0e-4 or besty < 1.0e-6,
        message="custom coordinate search finished",
        nfev=funcalls,
        nit=niter,
    )


def plot_custom_minimizer_path() -> None:
    x0 = np.array([-1.2, 1.0])
    path = [x0.copy()]
    result = minimize(
        custom_bowl,
        x0,
        method=coordinate_search_minimizer,
        callback=lambda x: path.append(np.array(x, dtype=float).copy()),
        options={"stepsize": 0.24, "maxiter": 180},
    )
    path_arr = np.vstack(path)

    x = np.linspace(-1.55, 1.25, 420)
    y = np.linspace(-0.75, 1.35, 360)
    xx, yy = np.meshgrid(x, y)
    shifted_x = xx - 0.8
    shifted_y = yy + 0.35
    zz = 0.5 * (5.0 * shifted_x**2 + 2.5 * shifted_x * shifted_y + 1.35 * shifted_y**2)
    levels = np.geomspace(0.01, 12.0, 22)

    fig, ax = plt.subplots(figsize=(8.4, 5.7))
    contours = ax.contour(xx, yy, zz, levels=levels, cmap="Blues", linewidths=0.9)
    ax.clabel(contours, contours.levels[::5], inline=True, fontsize=7, fmt="%.1g")
    ax.plot(path_arr[:, 0], path_arr[:, 1], color="#f59e0b", linewidth=1.8, marker="o", markersize=3.5, label="custom method path")
    ax.scatter([x0[0]], [x0[1]], color="#111827", s=70, zorder=5, label="start")
    ax.scatter([result.x[0]], [result.x[1]], marker="*", s=240, color="#16a34a", edgecolor="#111827", linewidth=0.55, zorder=6, label="reported point")
    ax.scatter([0.8], [-0.35], marker="x", s=95, color="#dc2626", linewidth=2.2, label="true minimum")
    ax.set_title("A callable custom minimizer can plug into minimize")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.58,
        0.08,
        f"{result.nfev} function calls"
        + "\n"
        + rf"$f(x)={result.fun:.2e}$",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "advanced_custom_minimizer_path")
    plt.close(fig)


def plot_parallel_workers_model() -> None:
    n = 8
    serial = n + 1
    workers_4 = np.ceil((n + 1) / 4.0) + 0.4
    vectorized = 1.6

    fig, (ax_stencil, ax_speed) = plt.subplots(1, 2, figsize=(10.4, 4.75), gridspec_kw={"width_ratios": [1.05, 1.0]})
    origin = np.array([0.0, 0.0])
    directions = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, n, endpoint=False)])
    points = origin + 0.82 * directions
    ax_stencil.scatter(points[:, 0], points[:, 1], s=82, color="#2563eb", edgecolor="white", linewidth=0.6, label=r"$x+h e_i$")
    ax_stencil.scatter([0], [0], s=145, color="#111827", label="$x$")
    for point in points:
        ax_stencil.plot([0, point[0]], [0, point[1]], color="#93c5fd", linewidth=1.0)
    ax_stencil.set_aspect("equal")
    ax_stencil.set_xlim(-1.15, 1.15)
    ax_stencil.set_ylim(-1.15, 1.15)
    ax_stencil.set_title("Finite differences create independent probes")
    ax_stencil.set_xlabel("parameter direction")
    ax_stencil.set_ylabel("perturbation")
    ax_stencil.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_stencil.legend(frameon=False, loc="upper right")
    ax_stencil.text(
        0.06,
        0.08,
        rf"two-point gradient: $N+1={serial}$ calls",
        transform=ax_stencil.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )

    labels = ["serial", "workers=4", "vectorized map"]
    values = [serial, workers_4, vectorized]
    colors = ["#64748b", "#2563eb", "#16a34a"]
    ax_speed.bar(labels, values, color=colors)
    ax_speed.set_title("Parallelism helps when calls are expensive")
    ax_speed.set_ylabel("relative evaluation batches")
    ax_speed.grid(True, axis="y", linestyle=":", linewidth=0.7, color="#cbd5e1")
    for index, value in enumerate(values):
        ax_speed.text(index, value + 0.25, f"{value:.1f}", ha="center", va="bottom", fontsize=10, color="#374151")
    ax_speed.set_ylim(0, max(values) * 1.22)
    save_figure(fig, "advanced_parallel_workers_model")
    plt.close(fig)


def main() -> None:
    plot_custom_minimizer_path()
    plot_parallel_workers_model()
    print(f"Wrote advanced-usage visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
