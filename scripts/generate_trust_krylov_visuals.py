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
    """Solve a small dense trust-region subproblem for visualization."""
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

    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(shifted_step(mid)) > radius:
            lo = mid
        else:
            hi = mid
    return shifted_step(hi)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def trust_krylov_history(x0: np.ndarray) -> tuple[list[np.ndarray], object]:
    history = [np.array(x0, dtype=float)]

    def callback(xk: np.ndarray) -> None:
        history.append(xk.copy())

    result = minimize(
        rosen2,
        np.array(x0, dtype=float),
        method="trust-krylov",
        jac=rosen2_grad,
        hess=rosen2_hess,
        callback=callback,
        options={"gtol": 1e-10, "maxiter": 120},
    )
    return history, result


def plot_trust_krylov_path() -> None:
    x = np.linspace(-1.55, 1.55, 540)
    y = np.linspace(-0.35, 2.45, 540)
    xx, yy = np.meshgrid(x, y)
    zz = 100.0 * (yy - xx**2) ** 2 + (1.0 - xx) ** 2

    history, result = trust_krylov_history(np.array([-1.2, 1.0]))
    points = np.array(history)
    values = np.array([rosen2(point) for point in points])
    grad_norms = np.array([np.linalg.norm(rosen2_grad(point)) for point in points])

    fig, ax = plt.subplots(figsize=(9.2, 6.35))
    levels = np.geomspace(0.1, 900, 34)
    contour = ax.contour(xx, yy, zz, levels=levels, cmap="viridis", linewidths=0.72)
    ax.clabel(contour, contour.levels[::6], inline=True, fontsize=8, fmt="%.1g")

    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidths=2.4, zorder=3)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)
    ax.scatter(points[:, 0], points[:, 1], s=34, c=np.linspace(0, 1, len(points)), cmap="plasma", edgecolor="white", linewidth=0.35, zorder=4)

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
    ax.scatter([points[-1, 0]], [points[-1, 1]], s=70, color="#16a34a", edgecolor="white", linewidth=0.8, zorder=6, label="trust-krylov finish")
    ax.set_title("trust-krylov on the shared Rosenbrock valley")
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
    save_figure(fig, "trust_krylov_rosenbrock_path")
    plt.close(fig)


def krylov_basis(h: np.ndarray, g: np.ndarray, max_dim: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    candidate = g / np.linalg.norm(g)

    for _ in range(max_dim):
        for q in vectors:
            candidate = candidate - q * float(q @ candidate)
        norm = np.linalg.norm(candidate)
        if norm < 1e-11:
            break
        q = candidate / norm
        vectors.append(q)
        candidate = h @ q

    return np.column_stack(vectors)


def plot_krylov_subspace_progress() -> None:
    rng = np.random.default_rng(7)
    n = 9
    random_matrix = rng.normal(size=(n, n))
    q, _ = np.linalg.qr(random_matrix)
    eigenvalues = np.array([-1.4, 0.18, 0.55, 1.1, 2.1, 4.0, 7.5, 12.0, 21.0])
    h = q @ np.diag(eigenvalues) @ q.T
    g = rng.normal(size=n)
    g = g / np.linalg.norm(g) * 2.3
    radius = 1.0

    full_step = trust_region_step(g, h, radius)
    full_reduction = -float(quadratic_model(g, h, full_step))

    dimensions = []
    reductions = []
    errors = []
    basis = krylov_basis(h, g, n)

    for dim in range(1, basis.shape[1] + 1):
        z = basis[:, :dim]
        h_small = z.T @ h @ z
        g_small = z.T @ g
        y = trust_region_step(g_small, h_small, radius)
        step = z @ y
        dimensions.append(dim)
        reductions.append(-float(quadratic_model(g, h, step)))
        errors.append(max(np.linalg.norm(step - full_step), 1e-14))

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.6, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax_top.plot(dimensions, reductions, color="#2563eb", linewidth=2.5, marker="o", label=r"Krylov-restricted solution")
    ax_top.axhline(full_reduction, color="#111827", linestyle="--", linewidth=1.1, label="full-space trust solution")
    ax_top.set_title("Projected trust-region solves improve with Krylov dimension")
    ax_top.set_ylabel("predicted model reduction")
    ax_top.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_top.legend(frameon=False, loc="lower right")

    ax_bottom.plot(dimensions, errors, color="#0f766e", linewidth=2.3, marker="s")
    ax_bottom.set_yscale("log")
    ax_bottom.set_xlabel(r"Krylov dimension $m$")
    ax_bottom.set_ylabel(r"$\|p_m-p_\star\|$")
    ax_bottom.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_bottom.text(
        0.98,
        0.84,
        "step approaches the full-space solution",
        transform=ax_bottom.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "trust_krylov_subspace_progress")
    plt.close(fig)


def plot_indefinite_model() -> None:
    h = np.array([[-1.35, 0.28], [0.28, 2.35]], dtype=float)
    g = np.array([0.46, -0.24], dtype=float)
    radius = 1.0
    step = trust_region_step(g, h, radius)
    eigvals, eigvecs = np.linalg.eigh(h)
    neg_vec = eigvecs[:, np.argmin(eigvals)]
    if neg_vec @ step < 0:
        neg_vec = -neg_vec
    cauchy = -radius * g / np.linalg.norm(g)

    p0 = np.linspace(-1.18, 1.18, 540)
    p1 = np.linspace(-1.18, 1.18, 540)
    pp0, pp1 = np.meshgrid(p0, p1)
    grid = np.stack([pp0, pp1], axis=0)
    qq = quadratic_model(g, h, grid)

    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    levels = np.linspace(float(np.percentile(qq, 4)), float(np.percentile(qq, 94)), 34)
    ax.contour(pp0, pp1, qq, levels=levels, cmap="viridis", linewidths=0.82)
    ax.add_patch(Circle((0.0, 0.0), radius, fill=False, color="#2563eb", linewidth=2.2))
    ax.plot([0, step[0]], [0, step[1]], color="#2563eb", linewidth=2.8, marker="o", markersize=6, label="trust-krylov model step")
    ax.plot([0, cauchy[0]], [0, cauchy[1]], color="#64748b", linewidth=2.0, linestyle="--", marker="s", markersize=5, label="steepest boundary step")
    ax.arrow(0, 0, neg_vec[0] * 0.78, neg_vec[1] * 0.78, width=0.008, head_width=0.055, head_length=0.065, color="#dc2626", alpha=0.72, length_includes_head=True, label="negative curvature")
    ax.scatter([0], [0], s=42, color="#111827", zorder=5)
    ax.annotate("$p=0$", (0.035, 0.035), fontsize=10)
    ax.annotate("negative\ncurvature", neg_vec * 0.52 + np.array([0.02, 0.05]), color="#991b1b", fontsize=10)
    ax.annotate("Krylov step", step + np.array([0.03, 0.02]), color="#1d4ed8", fontsize=10)
    ax.set_title("Krylov subspaces expose useful negative curvature")
    ax.set_xlabel("step coordinate $p_0$")
    ax.set_ylabel("step coordinate $p_1$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cbd5e1")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.04,
        0.04,
        r"When $H_k$ is indefinite, the best step often lies on the trust boundary.",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "trust_krylov_indefinite_model")
    plt.close(fig)


def main() -> None:
    plot_trust_krylov_path()
    plot_krylov_subspace_progress()
    plot_indefinite_model()
    print(f"Wrote trust-krylov visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
