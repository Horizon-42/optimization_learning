from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root, root_scalar
from scipy.sparse import dia_array, eye, kron
from scipy.sparse.linalg import LinearOperator, spilu


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


def scalar_fun(x: float | np.ndarray) -> float | np.ndarray:
    return x + 2.0 * np.cos(x)


def scalar_derivative(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 - 2.0 * np.sin(x)


def recorded_root_scalar(method: str) -> tuple[object, np.ndarray]:
    history: list[tuple[float, float]] = []

    def f(x: float) -> float:
        value = float(scalar_fun(float(x)))
        history.append((float(x), value))
        return value

    def fp(x: float) -> float:
        return float(scalar_derivative(float(x)))

    if method == "brentq":
        result = root_scalar(f, method="brentq", bracket=(-2.0, 0.0), xtol=1.0e-12)
    elif method == "newton":
        result = root_scalar(f, method="newton", x0=-0.5, fprime=fp, xtol=1.0e-12)
    else:
        raise ValueError(method)
    return result, np.array(history)


def plot_scalar_roots() -> None:
    brent, brent_history = recorded_root_scalar("brentq")
    newton, newton_history = recorded_root_scalar("newton")
    xs = np.linspace(-2.25, 0.65, 620)
    ys = scalar_fun(xs)

    fig, ax = plt.subplots(figsize=(9.1, 5.05))
    ax.plot(xs, ys, color="#111827", linewidth=2.2, label=r"$f(x)=x+2\cos x$")
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.axvspan(-2.0, 0.0, color="#dbeafe", alpha=0.55, label="brentq sign-change bracket")
    ax.scatter(brent_history[:, 0], brent_history[:, 1], color="#2563eb", s=48, edgecolor="white", linewidth=0.45, zorder=5, label="brentq evaluations")
    ax.scatter(newton_history[:, 0], newton_history[:, 1], color="#f59e0b", marker="s", s=46, edgecolor="white", linewidth=0.45, zorder=6, label="Newton evaluations")
    ax.scatter([brent.root], [0.0], marker="*", s=240, color="#16a34a", edgecolor="#111827", linewidth=0.55, zorder=7, label="root")

    for x_value in newton_history[:4, 0]:
        slope = scalar_derivative(x_value)
        y_value = scalar_fun(x_value)
        tangent_x = np.linspace(x_value - 0.35, x_value + 0.35, 60)
        ax.plot(tangent_x, y_value + slope * (tangent_x - x_value), color="#f59e0b", linewidth=1.0, alpha=0.5)

    ax.set_title("Scalar root finding: bracketed safety versus derivative speed")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.60,
        0.10,
        rf"$x^\star={brent.root:.8f}$"
        + "\n"
        + f"brentq calls: {brent.function_calls}"
        + "\n"
        + f"Newton calls: {newton.function_calls}",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "root_scalar_methods")
    plt.close(fig)


def system_fun(x: np.ndarray) -> np.ndarray:
    return np.array([x[0] * np.cos(x[1]) - 4.0, x[0] * x[1] - x[1] - 5.0])


def system_jac(x: np.ndarray) -> np.ndarray:
    return np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [x[1], x[0] - 1.0]])


def plot_system_root() -> None:
    result = root(system_fun, np.array([1.0, 1.0]), jac=system_jac)
    x0 = np.linspace(0.5, 8.0, 440)
    x1 = np.linspace(-0.2, 2.0, 360)
    xx, yy = np.meshgrid(x0, x1)
    f1 = xx * np.cos(yy) - 4.0
    f2 = xx * yy - yy - 5.0
    norm = np.sqrt(f1**2 + f2**2)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    im = ax.contourf(xx, yy, np.log10(norm + 1.0e-4), levels=28, cmap="viridis")
    ax.contour(xx, yy, f1, levels=[0.0], colors="#f97316", linewidths=2.2)
    ax.contour(xx, yy, f2, levels=[0.0], colors="#38bdf8", linewidths=2.2)
    ax.scatter([1.0], [1.0], color="#111827", s=72, label="initial guess", zorder=5)
    ax.scatter([result.x[0]], [result.x[1]], marker="*", s=260, color="#16a34a", edgecolor="#111827", linewidth=0.6, label="root", zorder=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\log_{10}\|F(x)\|_2$")
    ax.set_title("System roots occur where residual contours intersect")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.56,
        0.08,
        rf"$x^\star=({result.x[0]:.4f}, {result.x[1]:.4f})$"
        + "\n"
        + rf"$\|F(x^\star)\|_2={np.linalg.norm(system_fun(result.x)):.1e}$",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "root_system_contours")
    plt.close(fig)


def large_residual(v: np.ndarray, nx: int, ny: int, hx: float, hy: float) -> np.ndarray:
    p = v.reshape((nx, ny))
    p_left = p_right = p_bottom = 0.0
    p_top = 1.0
    d2x = np.zeros_like(p)
    d2y = np.zeros_like(p)
    d2x[1:-1] = (p[2:] - 2.0 * p[1:-1] + p[:-2]) / hx / hx
    d2x[0] = (p[1] - 2.0 * p[0] + p_left) / hx / hx
    d2x[-1] = (p_right - 2.0 * p[-1] + p[-2]) / hx / hx
    d2y[:, 1:-1] = (p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]) / hy / hy
    d2y[:, 0] = (p[:, 1] - 2.0 * p[:, 0] + p_bottom) / hy / hy
    d2y[:, -1] = (p_top - 2.0 * p[:, -1] + p[:, -2]) / hy / hy
    return (d2x + d2y + 5.0 * np.cosh(p).mean() ** 2).ravel()


def poisson_preconditioner(nx: int, ny: int, hx: float, hy: float) -> LinearOperator:
    x_diagonals = np.zeros((3, nx))
    x_diagonals[0, :] = 1.0 / hx / hx
    x_diagonals[1, :] = -2.0 / hx / hx
    x_diagonals[2, :] = 1.0 / hx / hx
    lx = dia_array((x_diagonals, [-1, 0, 1]), shape=(nx, nx))

    y_diagonals = np.zeros((3, ny))
    y_diagonals[0, :] = 1.0 / hy / hy
    y_diagonals[1, :] = -2.0 / hy / hy
    y_diagonals[2, :] = 1.0 / hy / hy
    ly = dia_array((y_diagonals, [-1, 0, 1]), shape=(ny, ny))

    jacobian_approx = kron(lx, eye(ny)) + kron(eye(nx), ly)
    ilu = spilu(jacobian_approx.tocsc())
    return LinearOperator(shape=(nx * ny, nx * ny), matvec=ilu.solve)


def solve_large_root(preconditioning: bool) -> tuple[object, int, list[float]]:
    nx = ny = 24
    hx = 1.0 / (nx - 1.0)
    hy = 1.0 / (ny - 1.0)
    call_count = 0
    history: list[float] = []

    def residual(v: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return large_residual(v, nx, ny, hx, hy)

    def callback(x: np.ndarray, f: np.ndarray) -> None:
        del x
        history.append(float(np.abs(f).max()))

    jac_options = {}
    if preconditioning:
        jac_options["inner_M"] = poisson_preconditioner(nx, ny, hx, hy)

    result = root(
        residual,
        np.zeros(nx * ny),
        method="krylov",
        callback=callback,
        options={"fatol": 1.0e-8, "jac_options": jac_options},
    )
    final_residual = float(np.abs(large_residual(result.x, nx, ny, hx, hy)).max())
    history.append(final_residual)
    return result, call_count, history


def plot_krylov_preconditioning() -> None:
    no_precond, no_precond_calls, no_precond_history = solve_large_root(False)
    precond, precond_calls, precond_history = solve_large_root(True)
    nx = ny = 24
    solution = precond.x.reshape((nx, ny))

    fig, (ax_solution, ax_calls) = plt.subplots(1, 2, figsize=(10.6, 4.75), gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.subplots_adjust(wspace=0.48)
    im = ax_solution.imshow(solution.T, origin="lower", extent=(0, 1, 0, 1), cmap="magma", aspect="equal")
    ax_solution.set_title("Krylov root solution field")
    ax_solution.set_xlabel("$x$")
    ax_solution.set_ylabel("$y$")
    cbar = fig.colorbar(im, ax=ax_solution, fraction=0.046, pad=0.02)
    cbar.set_label("$P(x,y)$")

    labels = ["no preconditioner", "ILU preconditioner"]
    counts = [no_precond_calls, precond_calls]
    colors = ["#64748b", "#2563eb"]
    ax_calls.bar(labels, counts, color=colors)
    ax_calls.set_title("Preconditioning cuts residual evaluations")
    ax_calls.set_ylabel("residual calls")
    ax_calls.set_ylim(0.0, max(counts) * 1.18)
    ax_calls.grid(True, axis="y", linestyle=":", linewidth=0.7, color="#cbd5e1")
    for index, count in enumerate(counts):
        ax_calls.text(index, count + max(counts) * 0.035, str(count), ha="center", va="bottom", fontsize=10, color="#374151")
    ax_calls.text(
        0.5,
        0.08,
        rf"final residuals: {no_precond_history[-1]:.1e} and {precond_history[-1]:.1e}",
        transform=ax_calls.transAxes,
        ha="center",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "root_krylov_preconditioning")
    plt.close(fig)


def main() -> None:
    plot_scalar_roots()
    plot_system_root()
    plot_krylov_preconditioning()
    print(f"Wrote root-finding visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
