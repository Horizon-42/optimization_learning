from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.optimize import minimize


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


def rosen2(x: np.ndarray) -> float:
    """Two-dimensional Rosenbrock function."""
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)


def nelder_mead_history(
    f,
    initial_simplex: np.ndarray,
    max_iter: int = 95,
    tol: float = 1e-8,
) -> list[np.ndarray]:
    """Small educational Nelder-Mead implementation that stores simplex geometry."""
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    simplex = np.array(initial_simplex, dtype=float)
    history = [simplex.copy()]

    for _ in range(max_iter):
        values = np.array([f(point) for point in simplex])
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        if np.std(values) < tol:
            break

        best, second_worst, worst = simplex
        centroid = (best + second_worst) / 2.0

        reflected = centroid + alpha * (centroid - worst)
        reflected_value = f(reflected)

        if values[0] <= reflected_value < values[1]:
            simplex[-1] = reflected
        elif reflected_value < values[0]:
            expanded = centroid + gamma * (reflected - centroid)
            simplex[-1] = expanded if f(expanded) < reflected_value else reflected
        else:
            if reflected_value < values[-1]:
                contracted = centroid + rho * (reflected - centroid)
                if f(contracted) <= reflected_value:
                    simplex[-1] = contracted
                else:
                    simplex[1:] = best + sigma * (simplex[1:] - best)
            else:
                contracted = centroid + rho * (worst - centroid)
                if f(contracted) < values[-1]:
                    simplex[-1] = contracted
                else:
                    simplex[1:] = best + sigma * (simplex[1:] - best)

        history.append(simplex.copy())

    return history


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def plot_simplex_path() -> None:
    x = np.linspace(-1.8, 1.8, 540)
    y = np.linspace(-0.7, 2.7, 540)
    xx, yy = np.meshgrid(x, y)
    zz = 100.0 * (yy - xx**2) ** 2 + (1.0 - xx) ** 2

    initial_simplex = np.array([[-1.35, 1.65], [-1.05, 2.25], [-0.55, 1.55]])
    history = nelder_mead_history(rosen2, initial_simplex)
    best_points = np.array([simplex[np.argmin([rosen2(p) for p in simplex])] for simplex in history])

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    levels = np.geomspace(0.1, 900, 34)
    contour = ax.contour(xx, yy, zz, levels=levels, cmap="viridis", linewidths=0.72)
    ax.clabel(contour, contour.levels[::5], inline=True, fontsize=8, fmt="%.1f")

    segments = np.stack([best_points[:-1], best_points[1:]], axis=1)
    lc = LineCollection(segments, cmap="magma", linewidths=2.1)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)
    ax.scatter(best_points[:, 0], best_points[:, 1], s=12, c=np.linspace(0, 1, len(best_points)), cmap="magma", zorder=4)

    for index in [0, 1, 2, 4, 8, 16, 32, len(history) - 1]:
        if index >= len(history):
            continue
        simplex = history[index]
        closed = np.vstack([simplex, simplex[0]])
        alpha = 0.24 if index != len(history) - 1 else 0.75
        ax.plot(closed[:, 0], closed[:, 1], color="#1f2937", linewidth=1.1, alpha=alpha)
        ax.fill(closed[:, 0], closed[:, 1], color="#38bdf8", alpha=0.04 if index != len(history) - 1 else 0.16)

    ax.scatter([1.0], [1.0], marker="*", s=180, color="#f59e0b", edgecolor="#111827", linewidth=0.5, zorder=5, label="global minimum")
    ax.scatter(initial_simplex[:, 0], initial_simplex[:, 1], s=42, color="#0f766e", edgecolor="white", linewidth=0.8, zorder=6, label="initial simplex")
    ax.set_title("Nelder-Mead on the Rosenbrock valley")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-0.35, 2.45)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        -1.48,
        -0.22,
        "Triangles show selected simplex shapes; the colored curve follows the current best vertex.",
        fontsize=9,
        color="#374151",
    )
    save_figure(fig, "nelder_mead_rosenbrock_path")
    plt.close(fig)


def plot_convergence() -> None:
    x0 = np.array([-1.35, 1.65])
    values: list[float] = [rosen2(x0)]

    def callback(xk: np.ndarray) -> None:
        values.append(rosen2(xk))

    result = minimize(
        rosen2,
        x0,
        method="Nelder-Mead",
        callback=callback,
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 600},
    )
    values.append(float(result.fun))

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(values, color="#2563eb", linewidth=2.2)
    ax.scatter([0, len(values) - 1], [values[0], values[-1]], color=["#dc2626", "#16a34a"], zorder=3)
    ax.set_yscale("log")
    ax.set_title("Function value decreases without gradients")
    ax.set_xlabel("callback step")
    ax.set_ylabel("$f(x)$ on a log scale")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.text(
        0.98,
        0.92,
        f"final x = ({result.x[0]:.5f}, {result.x[1]:.5f})\nfunction evaluations = {result.nfev}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "nelder_mead_convergence")
    plt.close(fig)


def main() -> None:
    plot_simplex_path()
    plot_convergence()
    print(f"Wrote Nelder-Mead visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
