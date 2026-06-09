from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.optimize import line_search, minimize


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


def rosen2_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of the two-dimensional Rosenbrock function."""
    return np.array(
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=float,
    )


def armijo_backtracking(x: np.ndarray, p: np.ndarray, g: np.ndarray, alpha0: float = 1.0) -> float:
    alpha = alpha0
    fx = rosen2(x)
    slope = float(g @ p)
    while rosen2(x + alpha * p) > fx + 1e-4 * alpha * slope:
        alpha *= 0.5
        if alpha < 1e-10:
            break
    return alpha


def bfgs_history(x0: np.ndarray, max_iter: int = 70) -> list[dict[str, np.ndarray | float]]:
    """Educational BFGS loop that records the inverse-Hessian approximation."""
    x = np.array(x0, dtype=float)
    h_inv = np.eye(2)
    history: list[dict[str, np.ndarray | float]] = [
        {
            "x": x.copy(),
            "f": rosen2(x),
            "grad": rosen2_grad(x),
            "h_inv": h_inv.copy(),
            "alpha": 0.0,
        }
    ]

    for _ in range(max_iter):
        g = rosen2_grad(x)
        if np.linalg.norm(g) < 1e-8:
            break

        p = -h_inv @ g
        alpha = line_search(rosen2, rosen2_grad, x, p, gfk=g, old_fval=rosen2(x))[0]
        if alpha is None or not np.isfinite(alpha):
            alpha = armijo_backtracking(x, p, g)

        x_next = x + alpha * p
        g_next = rosen2_grad(x_next)
        s = x_next - x
        y = g_next - g
        ys = float(y @ s)
        if ys <= 1e-12:
            break

        rho = 1.0 / ys
        eye = np.eye(2)
        h_inv = (eye - rho * np.outer(s, y)) @ h_inv @ (eye - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x = x_next
        history.append(
            {
                "x": x.copy(),
                "f": rosen2(x),
                "grad": g_next.copy(),
                "h_inv": h_inv.copy(),
                "alpha": float(alpha),
            }
        )

    return history


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def plot_bfgs_path() -> None:
    x = np.linspace(-1.45, 1.45, 540)
    y = np.linspace(-0.25, 1.75, 540)
    xx, yy = np.meshgrid(x, y)
    zz = 100.0 * (yy - xx**2) ** 2 + (1.0 - xx) ** 2

    history = bfgs_history(np.array([-1.2, 1.0]))
    points = np.array([entry["x"] for entry in history], dtype=float)
    grad_norms = np.array([np.linalg.norm(entry["grad"]) for entry in history], dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 6.35))
    levels = np.geomspace(0.03, 800, 36)
    contour = ax.contour(xx, yy, zz, levels=levels, cmap="viridis", linewidths=0.72)
    ax.clabel(contour, contour.levels[::6], inline=True, fontsize=8, fmt="%.2g")

    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidths=2.35, zorder=3)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)
    ax.scatter(points[:, 0], points[:, 1], s=26, c=np.linspace(0, 1, len(points)), cmap="plasma", edgecolor="white", linewidth=0.35, zorder=4)

    for index in [0, 1, 2, 5, 10, min(18, len(history) - 1)]:
        if index >= len(history):
            continue
        point = points[index]
        grad = np.array(history[index]["grad"], dtype=float)
        h_inv = np.array(history[index]["h_inv"], dtype=float)
        direction = -h_inv @ grad
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            ax.arrow(
                point[0],
                point[1],
                0.13 * direction[0],
                0.13 * direction[1],
                width=0.006,
                head_width=0.045,
                head_length=0.05,
                length_includes_head=True,
                color="#111827",
                alpha=0.7,
                zorder=5,
            )

    ax.scatter([1.0], [1.0], marker="*", s=190, color="#f59e0b", edgecolor="#111827", linewidth=0.5, zorder=6, label="global minimum")
    ax.scatter([points[0, 0]], [points[0, 1]], s=70, color="#dc2626", edgecolor="white", linewidth=0.8, zorder=6, label="start")
    ax.scatter([points[-1, 0]], [points[-1, 1]], s=70, color="#16a34a", edgecolor="white", linewidth=0.8, zorder=6, label="BFGS finish")
    ax.set_title("BFGS learns curvature while crossing the Rosenbrock valley")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(-1.35, 1.32)
    ax.set_ylim(-0.18, 1.65)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        -1.31,
        -0.09,
        f"{len(points) - 1} BFGS iterations; final gradient norm {grad_norms[-1]:.2e}",
        fontsize=9,
        color="#374151",
    )
    save_figure(fig, "bfgs_rosenbrock_path")
    plt.close(fig)


class EvaluationRecorder:
    def __init__(self) -> None:
        self.values: list[float] = []

    def fun(self, x: np.ndarray) -> float:
        value = rosen2(x)
        self.values.append(value)
        return value


def cumulative_best(values: list[float]) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def plot_evaluation_efficiency() -> None:
    x0 = np.array([-1.2, 1.0])

    bfgs_record = EvaluationRecorder()
    bfgs_result = minimize(
        bfgs_record.fun,
        x0,
        method="BFGS",
        jac=rosen2_grad,
        options={"gtol": 1e-9, "maxiter": 200},
    )

    nm_record = EvaluationRecorder()
    nm_result = minimize(
        nm_record.fun,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 800},
    )

    bfgs_best = cumulative_best(bfgs_record.values)
    nm_best = cumulative_best(nm_record.values)

    fig, ax = plt.subplots(figsize=(8.6, 4.95))
    ax.plot(np.arange(1, len(nm_best) + 1), nm_best, color="#64748b", linewidth=2.0, label=f"Nelder-Mead ({nm_result.nfev} f evals)")
    ax.plot(np.arange(1, len(bfgs_best) + 1), bfgs_best, color="#2563eb", linewidth=2.4, label=f"BFGS + analytic gradient ({bfgs_result.nfev} f evals)")
    ax.scatter([len(bfgs_best)], [bfgs_best[-1]], color="#2563eb", s=40, zorder=3)
    ax.scatter([len(nm_best)], [nm_best[-1]], color="#64748b", s=40, zorder=3)
    ax.set_yscale("log")
    ax.set_title("BFGS usually buys progress with fewer objective calls")
    ax.set_xlabel("objective function evaluations")
    ax.set_ylabel("best $f(x)$ seen so far")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.94,
        f"BFGS gradient evaluations: {bfgs_result.njev}\nfinal gradient norm: {np.linalg.norm(rosen2_grad(bfgs_result.x)):.2e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "bfgs_evaluation_efficiency")
    plt.close(fig)


def main() -> None:
    plot_bfgs_path()
    plot_evaluation_efficiency()
    print(f"Wrote BFGS visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
