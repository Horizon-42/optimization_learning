from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)

X0 = np.array([0.5, 0.0])
BOUNDS = Bounds([0.0, -0.5], [1.0, 2.0])
BOUNDS_AS_PAIRS = [(0.0, 1.0), (-0.5, 2.0)]

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


def rosen(x: np.ndarray) -> float:
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)


def rosen_der(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ]
    )


def rosen_hess(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0, -400.0 * x[0]],
            [-400.0 * x[0], 200.0],
        ]
    )


def nonlinear_cons(x: np.ndarray) -> np.ndarray:
    return np.array([x[0] ** 2 + x[1], x[0] ** 2 - x[1]])


def nonlinear_cons_jac(x: np.ndarray) -> np.ndarray:
    return np.array([[2.0 * x[0], 1.0], [2.0 * x[0], -1.0]])


def nonlinear_cons_hess(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    del x
    return np.array([[2.0 * v[0] + 2.0 * v[1], 0.0], [0.0, 0.0]])


def linear_constraint() -> LinearConstraint:
    return LinearConstraint(np.array([[1.0, 2.0], [2.0, 1.0]]), [-np.inf, 1.0], [1.0, 1.0])


def nonlinear_constraint() -> NonlinearConstraint:
    return NonlinearConstraint(
        nonlinear_cons,
        [-np.inf, -np.inf],
        [1.0, 1.0],
        jac=nonlinear_cons_jac,
        hess=nonlinear_cons_hess,
    )


def slsqp_constraints() -> list[dict[str, Callable[[np.ndarray], np.ndarray]]]:
    ineq_cons = {
        "type": "ineq",
        "fun": lambda x: np.array(
            [
                1.0 - x[0] - 2.0 * x[1],
                1.0 - x[0] ** 2 - x[1],
                1.0 - x[0] ** 2 + x[1],
            ]
        ),
        "jac": lambda x: np.array([[-1.0, -2.0], [-2.0 * x[0], -1.0], [-2.0 * x[0], 1.0]]),
    }
    eq_cons = {
        "type": "eq",
        "fun": lambda x: np.array([2.0 * x[0] + x[1] - 1.0]),
        "jac": lambda x: np.array([[2.0, 1.0]]),
    }
    return [eq_cons, ineq_cons]


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def solve_trust_constr() -> tuple[object, np.ndarray]:
    path = [X0.copy()]

    def callback(xk: np.ndarray, state: object | None = None) -> bool:
        del state
        path.append(np.array(xk, dtype=float).copy())
        return False

    result = minimize(
        rosen,
        X0,
        method="trust-constr",
        jac=rosen_der,
        hess=rosen_hess,
        constraints=[linear_constraint(), nonlinear_constraint()],
        bounds=BOUNDS,
        callback=callback,
        options={"verbose": 0, "gtol": 1.0e-10, "xtol": 1.0e-10},
    )
    if np.linalg.norm(path[-1] - result.x) > 1.0e-10:
        path.append(result.x.copy())
    return result, np.vstack(path)


def solve_slsqp() -> tuple[object, np.ndarray]:
    path = [X0.copy()]

    def callback(xk: np.ndarray) -> None:
        path.append(np.array(xk, dtype=float).copy())

    result = minimize(
        rosen,
        X0,
        method="SLSQP",
        jac=rosen_der,
        constraints=slsqp_constraints(),
        bounds=BOUNDS_AS_PAIRS,
        callback=callback,
        options={"ftol": 1.0e-10, "maxiter": 100, "disp": False},
    )
    if np.linalg.norm(path[-1] - result.x) > 1.0e-10:
        path.append(result.x.copy())
    return result, np.vstack(path)


def inequality_slacks(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            1.0 - x[0] - 2.0 * x[1],
            1.0 - x[0] ** 2 - x[1],
            1.0 - x[0] ** 2 + x[1],
            x[0],
            1.0 - x[0],
            x[1] + 0.5,
            2.0 - x[1],
        ]
    )


def equality_residual(x: np.ndarray) -> float:
    return float(2.0 * x[0] + x[1] - 1.0)


def max_constraint_violation(x: np.ndarray) -> float:
    return float(max(0.0, -np.min(inequality_slacks(x)), abs(equality_residual(x))))


def feasible_equality_segment() -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, 700)
    ys = 1.0 - 2.0 * xs
    points = np.column_stack([xs, ys])
    mask = np.array([np.min(inequality_slacks(point)) >= -1.0e-12 for point in points])
    mask &= ys >= BOUNDS.lb[1] - 1.0e-12
    mask &= ys <= BOUNDS.ub[1] + 1.0e-12
    return xs[mask], ys[mask]


def draw_constraint_geometry(ax: plt.Axes) -> None:
    x_grid = np.linspace(-0.08, 1.04, 360)
    y_grid = np.linspace(-0.55, 1.08, 360)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = 100.0 * (yy - xx**2) ** 2 + (1.0 - xx) ** 2
    levels = np.geomspace(0.05, 220.0, 20)
    contours = ax.contour(xx, yy, zz, levels=levels, cmap="Blues", linewidths=0.95)
    ax.clabel(contours, contours.levels[::4], inline=True, fontsize=7, fmt="%.1g")

    line_x = np.linspace(-0.05, 1.05, 400)
    ax.plot(line_x, (1.0 - line_x) / 2.0, color="#f59e0b", linewidth=1.8, linestyle="--", label=r"$x_0+2x_1\leq1$")
    ax.plot(line_x, 1.0 - 2.0 * line_x, color="#7c3aed", linewidth=2.0, label=r"$2x_0+x_1=1$")
    ax.plot(line_x, 1.0 - line_x**2, color="#64748b", linewidth=1.35, linestyle=":", label=r"$x_0^2+x_1\leq1$")
    ax.plot(line_x, line_x**2 - 1.0, color="#64748b", linewidth=1.35, linestyle="-.", label=r"$x_0^2-x_1\leq1$")

    seg_x, seg_y = feasible_equality_segment()
    ax.plot(seg_x, seg_y, color="#16a34a", linewidth=5.0, solid_capstyle="round", label="feasible equality segment")
    ax.axvspan(BOUNDS.lb[0], BOUNDS.ub[0], color="#ecfdf5", alpha=0.26)
    ax.axhspan(BOUNDS.lb[1], BOUNDS.ub[1], color="#ecfdf5", alpha=0.18)
    ax.set_xlim(-0.08, 1.04)
    ax.set_ylim(-0.55, 1.08)
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")


def plot_feasible_geometry() -> None:
    trust_result, trust_path = solve_trust_constr()
    slsqp_result, slsqp_path = solve_slsqp()

    fig, ax = plt.subplots(figsize=(8.5, 6.25))
    draw_constraint_geometry(ax)
    ax.plot(trust_path[:, 0], trust_path[:, 1], color="#2563eb", marker="o", markersize=4.0, linewidth=2.0, label="trust-constr path")
    ax.plot(slsqp_path[:, 0], slsqp_path[:, 1], color="#dc2626", marker="s", markersize=4.0, linewidth=1.8, label="SLSQP path")
    ax.scatter([X0[0]], [X0[1]], color="#111827", s=70, zorder=6, label="start")
    ax.scatter([trust_result.x[0]], [trust_result.x[1]], marker="*", s=250, color="#16a34a", edgecolor="#111827", linewidth=0.6, zorder=7, label="solution")
    ax.set_title("Constrained Rosenbrock minimization follows feasible geometry")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.text(
        0.04,
        0.06,
        rf"$x^\star=({trust_result.x[0]:.4f}, {trust_result.x[1]:.4f})$"
        + "\n"
        + rf"$f(x^\star)={trust_result.fun:.4f}$",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "constrained_rosenbrock_geometry")
    plt.close(fig)


def plot_method_progress() -> None:
    solutions = [
        ("trust-constr", "#2563eb", *solve_trust_constr()),
        ("SLSQP", "#dc2626", *solve_slsqp()),
    ]
    f_star = min(result.fun for _, _, result, _ in solutions)

    fig, (ax_obj, ax_violation) = plt.subplots(1, 2, figsize=(10.4, 4.65))
    for name, color, result, path in solutions:
        objective = np.array([rosen(point) for point in path])
        gap = np.maximum(objective - f_star, 1.0e-14)
        violation = np.maximum(np.array([max_constraint_violation(point) for point in path]), 1.0e-14)
        iterations = np.arange(path.shape[0])
        ax_obj.semilogy(iterations, gap, color=color, marker="o", markersize=4.0, linewidth=2.1, label=f"{name}: {result.nit} iterations")
        ax_violation.semilogy(iterations, violation, color=color, marker="o", markersize=4.0, linewidth=2.1, label=name)

    ax_obj.set_title("Objective progress")
    ax_obj.set_xlabel("recorded iterate")
    ax_obj.set_ylabel(r"$f(x_k)-f(x^\star)$")
    ax_obj.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_obj.legend(frameon=False, loc="upper right")

    ax_violation.set_title("Maximum constraint violation")
    ax_violation.set_xlabel("recorded iterate")
    ax_violation.set_ylabel("violation")
    ax_violation.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_violation.legend(frameon=False, loc="upper right")
    ax_violation.text(
        0.04,
        0.08,
        "Violation combines inequality slack,\nbox bounds, and equality residual.",
        transform=ax_violation.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "constrained_method_progress")
    plt.close(fig)


def plot_constraint_slack() -> None:
    result, _ = solve_trust_constr()
    labels = [
        r"$1-x_0-2x_1$",
        r"$1-x_0^2-x_1$",
        r"$1-x_0^2+x_1$",
        r"$x_0-\ell_0$",
        r"$u_0-x_0$",
        r"$x_1-\ell_1$",
        r"$u_1-x_1$",
    ]
    slacks = inequality_slacks(result.x)
    equality_abs = abs(equality_residual(result.x))

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    y_pos = np.arange(len(labels))
    colors = ["#f59e0b", "#64748b", "#64748b", "#16a34a", "#16a34a", "#16a34a", "#16a34a"]
    ax.barh(y_pos, slacks, color=colors, alpha=0.88)
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_title("Positive slack means an inequality is satisfied")
    ax.set_xlabel("slack at the trust-constr solution")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.text(
        0.98,
        0.09,
        rf"equality residual $|2x_0+x_1-1|={equality_abs:.1e}$",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "constrained_constraint_slack")
    plt.close(fig)


def main() -> None:
    plot_feasible_geometry()
    plot_method_progress()
    plot_constraint_slack()
    print(f"Wrote constrained-minimization visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
