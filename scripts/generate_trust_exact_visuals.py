from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
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


def quadratic_model(g: np.ndarray, h: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.einsum("i,i...->...", g, p) + 0.5 * np.einsum("i...,ij,j...->...", p, h, p)


def nearly_exact_step(g: np.ndarray, h: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    """Solve a tiny dense trust-region subproblem and return the Lagrange shift."""
    eig_min = float(np.min(np.linalg.eigvalsh(h)))
    lower = max(0.0, -eig_min + 1e-10)

    if lower == 0.0:
        try:
            newton_step = -np.linalg.solve(h, g)
            if np.linalg.norm(newton_step) <= radius:
                return newton_step, 0.0
        except np.linalg.LinAlgError:
            pass

    def shifted_step(lam: float) -> np.ndarray:
        return -np.linalg.solve(h + lam * np.eye(h.shape[0]), g)

    lo = lower
    hi = max(1.0, lo * 2.0 + 1.0)
    while np.linalg.norm(shifted_step(hi)) > radius:
        hi *= 2.0

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(shifted_step(mid)) > radius:
            lo = mid
        else:
            hi = mid
    return shifted_step(hi), hi


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def trust_exact_history(x0: np.ndarray) -> tuple[list[np.ndarray], object]:
    history = [np.array(x0, dtype=float)]

    def callback(xk: np.ndarray) -> None:
        history.append(xk.copy())

    result = minimize(
        rosen2,
        np.array(x0, dtype=float),
        method="trust-exact",
        jac=rosen2_grad,
        hess=rosen2_hess,
        callback=callback,
        options={"gtol": 1e-10, "maxiter": 120},
    )
    return history, result


def plot_trust_exact_path() -> None:
    x = np.linspace(0.48, 1.42, 540)
    y = np.linspace(0.34, 1.84, 540)
    xx, yy = np.meshgrid(x, y)
    zz = 100.0 * (yy - xx**2) ** 2 + (1.0 - xx) ** 2

    history, result = trust_exact_history(np.array([1.3, 0.7]))
    points = np.array(history)
    values = np.array([rosen2(point) for point in points])
    grad_norms = np.array([np.linalg.norm(rosen2_grad(point)) for point in points])

    fig, ax = plt.subplots(figsize=(9.2, 6.35))
    levels = np.geomspace(1e-6, 260, 40)
    contour = ax.contour(xx, yy, zz, levels=levels, cmap="viridis", linewidths=0.72)
    ax.clabel(contour, contour.levels[::6], inline=True, fontsize=8, fmt="%.1g")

    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="magma", linewidths=2.45, zorder=3)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)
    ax.scatter(points[:, 0], points[:, 1], s=34, c=np.linspace(0, 1, len(points)), cmap="magma", edgecolor="white", linewidth=0.35, zorder=4)

    for index in [0, 1, 2, 4, min(7, len(points) - 1)]:
        if index + 1 >= len(points):
            continue
        start = points[index]
        step = points[index + 1] - start
        ax.arrow(
            start[0],
            start[1],
            step[0] * 0.72,
            step[1] * 0.72,
            width=0.0028,
            head_width=0.018,
            head_length=0.018,
            length_includes_head=True,
            color="#111827",
            alpha=0.62,
            zorder=5,
        )

    ax.scatter([1.0], [1.0], marker="*", s=190, color="#f59e0b", edgecolor="#111827", linewidth=0.5, zorder=6, label="global minimum")
    ax.scatter([points[0, 0]], [points[0, 1]], s=70, color="#dc2626", edgecolor="white", linewidth=0.8, zorder=6, label="start")
    ax.scatter([points[-1, 0]], [points[-1, 1]], s=70, color="#16a34a", edgecolor="white", linewidth=0.8, zorder=6, label="trust-exact finish")
    ax.set_title("trust-exact uses nearly exact dense trust-region steps")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(0.5, 1.38)
    ax.set_ylim(0.36, 1.78)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.03,
        0.05,
        f"{result.nit} outer iterations\n{result.nfev} function evaluations\n{result.nhev} Hessian evaluations\nfinal f={values[-1]:.1e}\nfinal gradient norm {grad_norms[-1]:.1e}",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "trust_exact_rosenbrock_path")
    plt.close(fig)


def plot_exact_subproblem_solution() -> None:
    xk = np.array([1.3, 0.7])
    g = rosen2_grad(xk)
    h = rosen2_hess(xk)
    radius = 0.35
    trust_step, lam = nearly_exact_step(g, h, radius)
    newton_step = -np.linalg.solve(h, g)
    cauchy = -radius * g / np.linalg.norm(g)

    all_points = np.vstack([np.zeros(2), trust_step, newton_step, cauchy])
    pad = np.array([0.13, 0.16])
    p0_min, p1_min = np.min(all_points, axis=0) - pad
    p0_max, p1_max = np.max(all_points, axis=0) + pad
    p0 = np.linspace(p0_min, p0_max, 560)
    p1 = np.linspace(p1_min, p1_max, 560)
    pp0, pp1 = np.meshgrid(p0, p1)
    grid = np.stack([pp0, pp1], axis=0)
    qq = quadratic_model(g, h, grid)

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    levels = np.linspace(float(np.percentile(qq, 2)), float(np.percentile(qq, 92)), 34)
    ax.contour(pp0, pp1, qq, levels=levels, cmap="viridis", linewidths=0.82)
    ax.add_patch(Circle((0.0, 0.0), radius, fill=False, color="#2563eb", linewidth=2.25))
    ax.plot([0, trust_step[0]], [0, trust_step[1]], color="#2563eb", linewidth=2.9, marker="o", markersize=6)
    ax.scatter([newton_step[0]], [newton_step[1]], marker="*", s=180, color="#f59e0b", edgecolor="#111827", linewidth=0.5, zorder=5)
    ax.plot([0, cauchy[0]], [0, cauchy[1]], color="#64748b", linewidth=2.0, linestyle="--", marker="s", markersize=5)
    ax.scatter([0], [0], s=42, color="#111827", zorder=6)
    ax.annotate("$p=0$", (0.012, 0.028), fontsize=10)
    ax.annotate("$p(\\lambda)$", trust_step + np.array([0.018, -0.05]), fontsize=10, color="#1d4ed8")
    ax.annotate("nearly exact\nboundary step", trust_step + np.array([-0.19, 0.06]), color="#1d4ed8", fontsize=10)
    ax.annotate("Newton step\noutside radius", newton_step + np.array([0.03, 0.0]), color="#92400e", fontsize=10)
    ax.annotate("steepest\nboundary step", cauchy + np.array([-0.08, -0.12]), color="#475569", fontsize=10)
    ax.annotate("trust boundary", np.array([-0.31, -0.17]), color="#1d4ed8", fontsize=10)
    ax.set_title("Nearly exact solve of one trust-region subproblem")
    ax.set_xlabel("step coordinate $p_0$")
    ax.set_ylabel("step coordinate $p_1$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cbd5e1")
    ax.text(
        0.97,
        0.04,
        rf"$(H+\lambda I)p=-g$" + "\n" + rf"$\lambda={lam:.1f}$,  $\|p\|=\Delta$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "trust_exact_subproblem_solution")
    plt.close(fig)


def plot_dense_cost_scaling() -> None:
    sizes = np.geomspace(50, 5000, 100)
    memory_gib = sizes**2 * 8.0 / (1024.0**3)
    hessp_memory_gib = sizes * 8.0 / (1024.0**3)
    exact_factor_work = 4.0 * (sizes / 100.0) ** 3
    hessp_work = sizes / 100.0

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.7, 6.35), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.15]})
    ax_top.plot(sizes, memory_gib, color="#2563eb", linewidth=2.6, label="dense Hessian storage")
    ax_top.plot(sizes, hessp_memory_gib, color="#0f766e", linewidth=2.2, linestyle="--", label="single vector storage")
    ax_top.axhline(1.0, color="#94a3b8", linewidth=1.0, linestyle=":")
    ax_top.text(55, 1.08, "1 GiB", fontsize=9, color="#475569")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("memory (GiB)")
    ax_top.set_title("trust-exact trades fewer iterations for dense linear algebra")
    ax_top.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_top.legend(frameon=False, loc="upper left")

    ax_bottom.plot(sizes, exact_factor_work, color="#b45309", linewidth=2.6, label="about 4 dense factorizations")
    ax_bottom.plot(sizes, hessp_work, color="#0f766e", linewidth=2.2, linestyle="--", label="one Hessian-vector product scale")
    ax_bottom.set_xscale("log")
    ax_bottom.set_yscale("log")
    ax_bottom.set_xlabel("number of variables $N$")
    ax_bottom.set_ylabel("relative work")
    ax_bottom.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_bottom.legend(frameon=False, loc="upper left")
    ax_bottom.text(
        0.98,
        0.08,
        "Dense Hessian methods are strongest when N is moderate\nand full second-order information is cheap.",
        transform=ax_bottom.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "trust_exact_dense_cost")
    plt.close(fig)


def main() -> None:
    plot_trust_exact_path()
    plot_exact_subproblem_solution()
    plot_dense_cost_scaling()
    print(f"Wrote trust-exact visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
