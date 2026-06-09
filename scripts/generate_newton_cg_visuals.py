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


def rosen2_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of the two-dimensional Rosenbrock function."""
    return np.array(
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=float,
    )


def rosen2_hess(x: np.ndarray) -> np.ndarray:
    """Hessian of the two-dimensional Rosenbrock function."""
    return np.array(
        [
            [1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0, -400.0 * x[0]],
            [-400.0 * x[0], 200.0],
        ],
        dtype=float,
    )


def rosen2_hessp(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hessian-vector product for the two-dimensional Rosenbrock function."""
    return rosen2_hess(x) @ p


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def newton_cg_history(x0: np.ndarray) -> tuple[list[np.ndarray], object]:
    history = [np.array(x0, dtype=float)]

    def callback(xk: np.ndarray) -> None:
        history.append(xk.copy())

    result = minimize(
        rosen2,
        np.array(x0, dtype=float),
        method="Newton-CG",
        jac=rosen2_grad,
        hessp=rosen2_hessp,
        callback=callback,
        options={"xtol": 1e-10, "maxiter": 120},
    )
    return history, result


def plot_newton_cg_path() -> None:
    x = np.linspace(0.48, 1.42, 540)
    y = np.linspace(0.34, 1.84, 540)
    xx, yy = np.meshgrid(x, y)
    zz = 100.0 * (yy - xx**2) ** 2 + (1.0 - xx) ** 2

    history, result = newton_cg_history(np.array([1.3, 0.7]))
    points = np.array(history)
    values = np.array([rosen2(point) for point in points])
    grad_norms = np.array([np.linalg.norm(rosen2_grad(point)) for point in points])

    fig, ax = plt.subplots(figsize=(9.2, 6.35))
    levels = np.geomspace(1e-5, 260, 38)
    contour = ax.contour(xx, yy, zz, levels=levels, cmap="viridis", linewidths=0.72)
    ax.clabel(contour, contour.levels[::6], inline=True, fontsize=8, fmt="%.1g")

    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="cividis", linewidths=2.25, zorder=3)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)
    ax.scatter(points[:, 0], points[:, 1], s=28, c=np.linspace(0, 1, len(points)), cmap="cividis", edgecolor="white", linewidth=0.35, zorder=4)

    for index in [0, 1, 2, 4, 8, min(12, len(points) - 1)]:
        if index >= len(points):
            continue
        point = points[index]
        gradient = rosen2_grad(point)
        try:
            newton_direction = -np.linalg.solve(rosen2_hess(point), gradient)
        except np.linalg.LinAlgError:
            continue
        norm = np.linalg.norm(newton_direction)
        if norm > 0:
            newton_direction = newton_direction / norm
            ax.arrow(
                point[0],
                point[1],
                0.055 * newton_direction[0],
                0.055 * newton_direction[1],
                width=0.0028,
                head_width=0.018,
                head_length=0.018,
                length_includes_head=True,
                color="#111827",
                alpha=0.72,
                zorder=5,
            )

    ax.scatter([1.0], [1.0], marker="*", s=190, color="#f59e0b", edgecolor="#111827", linewidth=0.5, zorder=6, label="global minimum")
    ax.scatter([points[0, 0]], [points[0, 1]], s=70, color="#dc2626", edgecolor="white", linewidth=0.8, zorder=6, label="start")
    ax.scatter([points[-1, 0]], [points[-1, 1]], s=70, color="#16a34a", edgecolor="white", linewidth=0.8, zorder=6, label="Newton-CG finish")
    ax.set_title("Newton-CG follows Hessian-informed directions")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(0.5, 1.38)
    ax.set_ylim(0.36, 1.78)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.03,
        0.05,
        f"{result.nit} outer iterations\n{result.nhev} Hessian-vector evaluations\nfinal f={values[-1]:.1e}\nfinal gradient norm {grad_norms[-1]:.1e}",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "newton_cg_rosenbrock_path")
    plt.close(fig)


def cg_solve_history(a: np.ndarray, b: np.ndarray, max_iter: int = 4) -> list[np.ndarray]:
    x = np.zeros_like(b)
    r = b - a @ x
    d = r.copy()
    history = [x.copy()]

    for _ in range(max_iter):
        denom = float(d @ a @ d)
        if abs(denom) < 1e-14:
            break
        alpha = float(r @ r) / denom
        x = x + alpha * d
        history.append(x.copy())
        r_next = r - alpha * (a @ d)
        if np.linalg.norm(r_next) < 1e-12:
            break
        beta = float(r_next @ r_next) / float(r @ r)
        d = r_next + beta * d
        r = r_next

    return history


def plot_inner_cg_model() -> None:
    xk = np.array([1.3, 0.7])
    gradient = rosen2_grad(xk)
    hessian = rosen2_hess(xk)
    b = -gradient
    cg_points = np.array(cg_solve_history(hessian, b, max_iter=3))
    exact_step = np.linalg.solve(hessian, b)

    all_points = np.vstack([cg_points, exact_step])
    pad = np.array([0.11, 0.18])
    p0_min, p1_min = np.min(all_points, axis=0) - pad
    p0_max, p1_max = np.max(all_points, axis=0) + pad
    p0 = np.linspace(p0_min, p0_max, 420)
    p1 = np.linspace(p1_min, p1_max, 420)
    pp0, pp1 = np.meshgrid(p0, p1)

    def model_value(p: np.ndarray) -> np.ndarray:
        return np.einsum("i,i...->...", gradient, p) + 0.5 * np.einsum("i...,ij,j...->...", p, hessian, p)

    grid = np.stack([pp0, pp1], axis=0)
    qq = model_value(grid)
    q_min = model_value(exact_step)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    levels = q_min + np.geomspace(1e-4, max(float(np.max(qq - q_min)), 1.0), 28)
    ax.contour(pp0, pp1, qq, levels=levels, cmap="mako" if "mako" in plt.colormaps() else "viridis", linewidths=0.8)
    ax.plot(cg_points[:, 0], cg_points[:, 1], color="#2563eb", linewidth=2.5, marker="o", markersize=6, label="CG iterates")
    ax.scatter([exact_step[0]], [exact_step[1]], marker="*", s=190, color="#f59e0b", edgecolor="#111827", linewidth=0.5, zorder=5, label="exact Newton step")

    label_offsets = [np.array([0.012, -0.055]), np.array([0.011, 0.022]), np.array([-0.06, -0.055])]
    for index, point in enumerate(cg_points):
        offset = label_offsets[index] if index < len(label_offsets) else np.array([0.01, 0.02])
        ax.annotate(f"$p_{index}$", point + offset, fontsize=10)

    ax.set_title("Inner CG solve of the local quadratic model")
    ax.set_xlabel("step coordinate $p_0$")
    ax.set_ylabel("step coordinate $p_1$")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cbd5e1")
    ax.text(
        0.03,
        0.04,
        r"CG approximately solves $H_kp=-g_k$ using products $H_kv$.",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "newton_cg_inner_cg_model")
    plt.close(fig)


def plot_hessp_scaling() -> None:
    dimensions = np.logspace(2, 5, 160)
    dense_gib = 8.0 * dimensions**2 / 1024**3
    hessp_workspace_gib = 8.0 * 6.0 * dimensions / 1024**3

    fig, ax = plt.subplots(figsize=(8.4, 4.95))
    ax.loglog(dimensions, dense_gib, color="#dc2626", linewidth=2.4, label="dense Hessian storage")
    ax.loglog(dimensions, hessp_workspace_gib, color="#2563eb", linewidth=2.4, label="Hessian-vector workspace")
    ax.fill_between(dimensions, hessp_workspace_gib, dense_gib, color="#93c5fd", alpha=0.18)
    ax.set_title("Why Hessian-vector products matter for large problems")
    ax.set_xlabel("number of variables $N$")
    ax.set_ylabel("approximate memory, GiB")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.97,
        0.08,
        "Dense storage grows like $N^2$;\nHessian-vector products keep the method iterative.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "newton_cg_hessp_scaling")
    plt.close(fig)


def main() -> None:
    plot_newton_cg_path()
    plot_inner_cg_model()
    plot_hessp_scaling()
    print(f"Wrote Newton-CG visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
