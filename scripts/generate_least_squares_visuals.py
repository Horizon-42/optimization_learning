from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)

U_DATA = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2])
Y_DATA = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])
X0 = np.array([2.5, 3.9, 4.15, 3.9])

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


def model(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return x[0] * (u**2 + x[1] * u) / (u**2 + x[2] * u + x[3])


def residuals(x: np.ndarray, u: np.ndarray = U_DATA, y: np.ndarray = Y_DATA) -> np.ndarray:
    return model(x, u) - y


def jacobian(x: np.ndarray, u: np.ndarray = U_DATA, y: np.ndarray = Y_DATA) -> np.ndarray:
    del y
    jac = np.empty((u.size, x.size))
    den = u**2 + x[2] * u + x[3]
    num = u**2 + x[1] * u
    jac[:, 0] = num / den
    jac[:, 1] = x[0] * u / den
    jac[:, 2] = -x[0] * num * u / den**2
    jac[:, 3] = -x[0] * num / den**2
    return jac


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def solve_enzyme_fit() -> object:
    return least_squares(residuals, X0, jac=jacobian, bounds=(0.0, 100.0), args=(U_DATA, Y_DATA))


def plot_enzyme_fit() -> None:
    result = solve_enzyme_fit()
    u_grid = np.linspace(0.055, 4.2, 420)
    y_fit = model(result.x, u_grid)
    fitted_at_data = model(result.x, U_DATA)

    fig, (ax_fit, ax_resid) = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]})
    ax_fit.plot(u_grid, y_fit, color="#2563eb", linewidth=2.7, label="least_squares fit")
    ax_fit.scatter(U_DATA, Y_DATA, s=54, color="#f59e0b", edgecolor="#111827", linewidth=0.45, zorder=4, label="measurements")
    for u_value, observed, fitted in zip(U_DATA, Y_DATA, fitted_at_data):
        ax_fit.plot([u_value, u_value], [observed, fitted], color="#64748b", linewidth=0.9, alpha=0.65)
    ax_fit.set_title("Bounded nonlinear least-squares fit")
    ax_fit.set_ylabel("reaction response $y$")
    ax_fit.set_ylim(0.0, 0.235)
    ax_fit.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_fit.legend(frameon=False, loc="lower right")
    ax_fit.text(
        0.03,
        0.93,
        f"final cost {result.cost:.2e}\n{result.nfev} function evaluations\n$\\|r(x)\\|_2={np.linalg.norm(result.fun):.2e}$",
        transform=ax_fit.transAxes,
        va="top",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )

    markerline, stemlines, baseline = ax_resid.stem(U_DATA, fitted_at_data - Y_DATA, linefmt="#2563eb", markerfmt="o", basefmt="#111827")
    plt.setp(markerline, markersize=5, markerfacecolor="#2563eb", markeredgecolor="white", markeredgewidth=0.6)
    plt.setp(stemlines, linewidth=1.2)
    plt.setp(baseline, linewidth=0.8)
    ax_resid.axhline(0.0, color="#111827", linewidth=0.8)
    ax_resid.set_xlabel("independent variable $u$")
    ax_resid.set_ylabel("residual")
    ax_resid.grid(True, axis="y", linestyle=":", linewidth=0.7, color="#cbd5e1")
    save_figure(fig, "least_squares_enzyme_fit")
    plt.close(fig)


def plot_jacobian_diagnostics() -> None:
    result = solve_enzyme_fit()
    jac = jacobian(result.x)
    scaled = np.abs(jac) / np.maximum(np.max(np.abs(jac), axis=0, keepdims=True), 1e-12)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    gradient = jac.T @ result.fun

    fig, (ax_heat, ax_svd) = plt.subplots(1, 2, figsize=(10.2, 4.9), gridspec_kw={"width_ratios": [1.25, 1.0]})
    im = ax_heat.imshow(scaled, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax_heat.set_title("Scaled analytic Jacobian")
    ax_heat.set_xlabel("parameter")
    ax_heat.set_ylabel("residual index $i$")
    ax_heat.set_xticks(range(4), labels=["$x_0$", "$x_1$", "$x_2$", "$x_3$"])
    ax_heat.set_yticks(range(U_DATA.size), labels=[str(i) for i in range(U_DATA.size)])
    for i in range(scaled.shape[0]):
        for j in range(scaled.shape[1]):
            if scaled[i, j] > 0.62:
                ax_heat.text(j, i, f"{scaled[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
    cbar.set_label("relative sensitivity")

    ax_svd.semilogy(np.arange(1, len(singular_values) + 1), singular_values, marker="o", color="#2563eb", linewidth=2.4)
    ax_svd.set_title("Jacobian conditioning")
    ax_svd.set_xlabel("singular value index")
    ax_svd.set_ylabel(r"$\sigma_j(J)$")
    ax_svd.set_xticks(np.arange(1, len(singular_values) + 1))
    ax_svd.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_svd.text(
        0.98,
        0.05,
        r"$J^\mathsf{T}r$ near zero at the solution"
        + "\n"
        + rf"$\|J^\mathsf{{T}}r\|_\infty={np.linalg.norm(gradient, ord=np.inf):.1e}$",
        transform=ax_svd.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "least_squares_jacobian_diagnostics")
    plt.close(fig)


def exp_model(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return x[0] * np.exp(-x[1] * t) + x[2]


def exp_residuals(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return exp_model(x, t) - y


def exp_jacobian(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    del y
    e = np.exp(-x[1] * t)
    jac = np.empty((t.size, 3))
    jac[:, 0] = e
    jac[:, 1] = -x[0] * t * e
    jac[:, 2] = 1.0
    return jac


def plot_robust_loss() -> None:
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 5.0, 46)
    truth = np.array([1.85, 0.82, 0.22])
    clean = exp_model(truth, t)
    y = clean + rng.normal(0.0, 0.035, size=t.size)
    outlier_indices = np.array([8, 22, 37])
    y[outlier_indices] += np.array([0.78, -0.58, 0.62])

    x0 = np.array([1.0, 0.25, 0.0])
    linear = least_squares(exp_residuals, x0, jac=exp_jacobian, args=(t, y), loss="linear")
    robust = least_squares(exp_residuals, x0, jac=exp_jacobian, args=(t, y), loss="soft_l1", f_scale=0.04)

    t_grid = np.linspace(0.0, 5.0, 420)
    fig, (ax_fit, ax_resid) = plt.subplots(2, 1, figsize=(8.7, 6.35), sharex=True, gridspec_kw={"height_ratios": [2.05, 1.0]})
    ax_fit.plot(t_grid, exp_model(truth, t_grid), color="#111827", linewidth=1.8, linestyle=":", label="true curve")
    ax_fit.plot(t_grid, exp_model(linear.x, t_grid), color="#dc2626", linewidth=2.4, label="linear loss")
    ax_fit.plot(t_grid, exp_model(robust.x, t_grid), color="#2563eb", linewidth=2.7, label="soft_l1 robust loss")
    ax_fit.scatter(t, y, s=34, color="#64748b", edgecolor="white", linewidth=0.4, label="noisy data")
    ax_fit.scatter(t[outlier_indices], y[outlier_indices], s=72, color="#f59e0b", edgecolor="#111827", linewidth=0.5, zorder=5, label="outliers")
    ax_fit.set_title("Robust least squares reduces outlier leverage")
    ax_fit.set_ylabel("$y$")
    ax_fit.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_fit.legend(frameon=False, loc="upper right")

    linear_resid = exp_model(linear.x, t) - y
    robust_resid = exp_model(robust.x, t) - y
    ax_resid.plot(t, np.abs(linear_resid), color="#dc2626", linewidth=1.8, marker="o", markersize=3.5, label="linear |residual|")
    ax_resid.plot(t, np.abs(robust_resid), color="#2563eb", linewidth=1.8, marker="s", markersize=3.5, label="soft_l1 |residual|")
    ax_resid.set_xlabel("input $t$")
    ax_resid.set_ylabel("absolute residual")
    ax_resid.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_resid.legend(frameon=False, loc="upper right")
    ax_resid.text(
        0.03,
        0.93,
        "Robust loss accepts larger residuals at outliers\nso the main trend is fitted better.",
        transform=ax_resid.transAxes,
        va="top",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "least_squares_robust_loss")
    plt.close(fig)


def main() -> None:
    plot_enzyme_fit()
    plot_jacobian_diagnostics()
    plot_robust_loss()
    print(f"Wrote least-squares visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
