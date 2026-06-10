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


def rosen2_hessp(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hessian-vector product for the two-dimensional Rosenbrock function."""
    return rosen2_hess(x) @ p


def quadratic_model(g: np.ndarray, h: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.einsum("i,i...->...", g, p) + 0.5 * np.einsum("i...,ij,j...->...", p, h, p)


def trust_region_step(g: np.ndarray, h: np.ndarray, radius: float) -> np.ndarray:
    """Solve the two-dimensional trust-region subproblem for visualization."""
    eig_min = float(np.min(np.linalg.eigvalsh(h)))
    lower = max(0.0, -eig_min + 1e-10)

    if lower == 0.0:
        try:
            newton_step = -np.linalg.solve(h, g)
            if np.linalg.norm(newton_step) <= radius:
                return newton_step
        except np.linalg.LinAlgError:
            pass

    def shifted_step(lam: float) -> np.ndarray:
        return -np.linalg.solve(h + lam * np.eye(h.shape[0]), g)

    lo = lower
    hi = max(1.0, lo * 2.0 + 1.0)
    while np.linalg.norm(shifted_step(hi)) > radius:
        hi *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(shifted_step(mid)) > radius:
            lo = mid
        else:
            hi = mid
    return shifted_step(hi)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def trust_ncg_history(x0: np.ndarray) -> tuple[list[np.ndarray], object]:
    history = [np.array(x0, dtype=float)]

    def callback(xk: np.ndarray) -> None:
        history.append(xk.copy())

    result = minimize(
        rosen2,
        np.array(x0, dtype=float),
        method="trust-ncg",
        jac=rosen2_grad,
        hess=rosen2_hess,
        callback=callback,
        options={"gtol": 1e-10, "maxiter": 120},
    )
    return history, result


def plot_trust_ncg_path() -> None:
    x = np.linspace(-1.55, 1.55, 540)
    y = np.linspace(-0.35, 2.45, 540)
    xx, yy = np.meshgrid(x, y)
    zz = 100.0 * (yy - xx**2) ** 2 + (1.0 - xx) ** 2

    history, result = trust_ncg_history(np.array([-1.2, 1.0]))
    points = np.array(history)
    values = np.array([rosen2(point) for point in points])
    grad_norms = np.array([np.linalg.norm(rosen2_grad(point)) for point in points])

    fig, ax = plt.subplots(figsize=(9.2, 6.35))
    levels = np.geomspace(0.1, 900, 34)
    contour = ax.contour(xx, yy, zz, levels=levels, cmap="viridis", linewidths=0.72)
    ax.clabel(contour, contour.levels[::6], inline=True, fontsize=8, fmt="%.1g")

    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="turbo", linewidths=2.35, zorder=3)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)
    ax.scatter(points[:, 0], points[:, 1], s=32, c=np.linspace(0, 1, len(points)), cmap="turbo", edgecolor="white", linewidth=0.35, zorder=4)

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
            alpha=0.6,
            zorder=5,
        )

    ax.scatter([1.0], [1.0], marker="*", s=190, color="#f59e0b", edgecolor="#111827", linewidth=0.5, zorder=6, label="global minimum")
    ax.scatter([points[0, 0]], [points[0, 1]], s=70, color="#dc2626", edgecolor="white", linewidth=0.8, zorder=6, label="start")
    ax.scatter([points[-1, 0]], [points[-1, 1]], s=70, color="#16a34a", edgecolor="white", linewidth=0.8, zorder=6, label="trust-ncg finish")
    ax.set_title("trust-ncg on the shared Rosenbrock valley")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-0.35, 2.45)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.03,
        0.05,
        f"{result.nit} outer iterations\n{result.nfev} function evaluations\n{result.nhev} Hessian evaluations\nfinal f={values[-1]:.1e}\nfinal gradient norm {grad_norms[-1]:.1e}",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "trust_ncg_rosenbrock_path")
    plt.close(fig)


def plot_trust_region_subproblem() -> None:
    xk = np.array([1.3, 0.7])
    g = rosen2_grad(xk)
    h = rosen2_hess(xk)
    radius = 0.35
    trust_step = trust_region_step(g, h, radius)
    newton_step = -np.linalg.solve(h, g)
    steepest = -radius * g / np.linalg.norm(g)

    all_points = np.vstack([np.zeros(2), trust_step, newton_step, steepest])
    pad = np.array([0.11, 0.18])
    p0_min, p1_min = np.min(all_points, axis=0) - pad
    p0_max, p1_max = np.max(all_points, axis=0) + pad
    p0 = np.linspace(p0_min, p0_max, 520)
    p1 = np.linspace(p1_min, p1_max, 520)
    pp0, pp1 = np.meshgrid(p0, p1)
    grid = np.stack([pp0, pp1], axis=0)
    qq = quadratic_model(g, h, grid)

    fig, ax = plt.subplots(figsize=(7.8, 7.1))
    levels = np.linspace(float(np.percentile(qq, 2)), float(np.percentile(qq, 92)), 32)
    ax.contour(pp0, pp1, qq, levels=levels, cmap="viridis", linewidths=0.8)
    ax.add_patch(Circle((0.0, 0.0), radius, fill=False, color="#2563eb", linewidth=2.2))
    ax.plot([0, trust_step[0]], [0, trust_step[1]], color="#2563eb", linewidth=2.8, marker="o", markersize=6)
    ax.scatter([newton_step[0]], [newton_step[1]], marker="*", s=180, color="#f59e0b", edgecolor="#111827", linewidth=0.5, zorder=5)
    ax.scatter([steepest[0]], [steepest[1]], marker="s", s=72, color="#0f766e", edgecolor="white", linewidth=0.6, zorder=5)
    ax.scatter([0], [0], s=42, color="#111827", zorder=6)
    ax.annotate("$p=0$", (0.01, 0.025), fontsize=10)
    ax.annotate("$p_k$", trust_step + np.array([0.015, -0.05]), fontsize=10)
    ax.annotate("accepted step", trust_step + np.array([-0.20, 0.04]), color="#1d4ed8", fontsize=10)
    ax.annotate("unconstrained\nNewton step", newton_step + np.array([0.03, -0.03]), color="#92400e", fontsize=10)
    ax.annotate("steepest\nboundary", steepest + np.array([-0.08, 0.06]), color="#0f766e", fontsize=10)
    ax.set_title("Trust-region subproblem")
    ax.set_xlabel("step coordinate $p_0$")
    ax.set_ylabel("step coordinate $p_1$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cbd5e1")
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.42, 1.18)
    ax.text(
        0.04,
        0.04,
        r"Minimize $m_k(p)$ while enforcing $\|p\|\leq\Delta_k$.",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "trust_ncg_subproblem")
    plt.close(fig)


def plot_radius_agreement() -> None:
    xk = np.array([1.3, 0.7])
    g = rosen2_grad(xk)
    h = rosen2_hess(xk)
    radii = np.geomspace(0.035, 1.25, 34)
    actual_reductions = []
    predicted_reductions = []
    ratios = []

    for radius in radii:
        step = trust_region_step(g, h, float(radius))
        predicted = -float(quadratic_model(g, h, step))
        actual = rosen2(xk) - rosen2(xk + step)
        actual_reductions.append(max(actual, 1e-16))
        predicted_reductions.append(max(predicted, 1e-16))
        ratios.append(actual / predicted if predicted > 0 else np.nan)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.6, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax_top.loglog(radii, predicted_reductions, color="#2563eb", linewidth=2.3, marker="o", markersize=3.5, label="predicted reduction")
    ax_top.loglog(radii, actual_reductions, color="#dc2626", linewidth=2.3, marker="s", markersize=3.5, label="actual reduction")
    ax_top.set_title("Trust radius is adjusted by model agreement")
    ax_top.set_ylabel("reduction")
    ax_top.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_top.legend(frameon=False, loc="upper left")

    ax_bottom.semilogx(radii, ratios, color="#0f766e", linewidth=2.3, marker="o", markersize=3.5)
    ax_bottom.axhspan(0.0, 0.25, color="#fee2e2", alpha=0.7, label="shrink")
    ax_bottom.axhspan(0.75, 1.25, color="#dcfce7", alpha=0.7, label="good agreement")
    ax_bottom.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    ax_bottom.set_xlabel(r"candidate trust radius $\Delta$")
    ax_bottom.set_ylabel(r"$\rho_k$")
    ax_bottom.set_ylim(-0.05, 1.35)
    ax_bottom.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_bottom.legend(frameon=False, loc="lower left", ncol=2)
    save_figure(fig, "trust_ncg_radius_agreement")
    plt.close(fig)


def main() -> None:
    plot_trust_ncg_path()
    plot_trust_region_subproblem()
    plot_radius_agreement()
    print(f"Wrote trust-ncg visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
