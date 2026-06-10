from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, linprog, milp


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


def solve_visual_lp() -> object:
    c = np.array([-3.0, -2.0])
    a_ub = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    b_ub = np.array([4.0, 5.0, 3.0, 3.0])
    return linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=[(0.0, None), (0.0, None)], method="highs")


def plot_linprog_geometry() -> None:
    result = solve_visual_lp()
    xs = np.linspace(0.0, 3.35, 420)
    ys = np.linspace(0.0, 3.35, 420)
    xx, yy = np.meshgrid(xs, ys)
    feasible = (xx + yy <= 4.0) & (2.0 * xx + yy <= 5.0) & (xx <= 3.0) & (yy <= 3.0)
    score = 3.0 * xx + 2.0 * yy

    fig, ax = plt.subplots(figsize=(8.3, 5.85))
    ax.contourf(xx, yy, feasible.astype(float), levels=[-0.1, 0.5, 1.1], colors=["#ffffff", "#d1fae5"], alpha=0.88)
    contours = ax.contour(xx, yy, score, levels=[3, 5, 7, 9], colors="#93c5fd", linewidths=1.1)
    ax.clabel(contours, inline=True, fontsize=8, fmt="value %.0f")

    line_x = np.linspace(0.0, 3.35, 300)
    ax.plot(line_x, 4.0 - line_x, color="#64748b", linestyle="--", linewidth=1.6, label=r"$x_0+x_1\leq4$")
    ax.plot(line_x, 5.0 - 2.0 * line_x, color="#f59e0b", linestyle="--", linewidth=1.6, label=r"$2x_0+x_1\leq5$")
    ax.axvline(3.0, color="#64748b", linestyle=":", linewidth=1.4, label=r"$x_0\leq3$")
    ax.axhline(3.0, color="#64748b", linestyle="-.", linewidth=1.4, label=r"$x_1\leq3$")
    ax.scatter([result.x[0]], [result.x[1]], marker="*", s=260, color="#16a34a", edgecolor="#111827", linewidth=0.6, zorder=6, label="linprog optimum")
    ax.set_xlim(0.0, 3.35)
    ax.set_ylim(0.0, 3.35)
    ax.set_title("Linear programming optimizes over a polytope")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.text(
        0.05,
        0.07,
        rf"maximize $3x_0+2x_1$"
        + "\n"
        + rf"$x^\star=({result.x[0]:.2f}, {result.x[1]:.2f})$"
        + "\n"
        + rf"value ${-result.fun:.2f}$",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "structured_linprog_geometry")
    plt.close(fig)


def medley_cost_matrix() -> tuple[np.ndarray, list[str], list[str]]:
    cost = np.array(
        [
            [43.5, 45.5, 43.4, 46.5, 46.3],
            [47.1, 42.1, 39.1, 44.1, 47.8],
            [48.4, 49.6, 42.1, 44.5, 50.4],
            [38.2, 36.8, 43.2, 41.2, 37.2],
        ]
    )
    styles = ["backstroke", "breaststroke", "butterfly", "freestyle"]
    students = ["A", "B", "C", "D", "E"]
    return cost, styles, students


def plot_assignment_matrix() -> None:
    cost, styles, students = medley_cost_matrix()
    row_ind, col_ind = linear_sum_assignment(cost)
    total = cost[row_ind, col_ind].sum()

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    im = ax.imshow(cost, cmap="YlGnBu_r")
    ax.set_title("Linear sum assignment chooses one cell per style")
    ax.set_xticks(np.arange(len(students)), labels=students)
    ax.set_yticks(np.arange(len(styles)), labels=styles)
    ax.set_xlabel("student")
    ax.set_ylabel("swimming style")
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            ax.text(j, i, f"{cost[i, j]:.1f}", ha="center", va="center", fontsize=9, color="#111827")
    for row, col in zip(row_ind, col_ind):
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#16a34a", linewidth=3.0))
        ax.text(col, row + 0.33, "chosen", ha="center", va="center", fontsize=7, color="#065f46", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("relay time")
    ax.text(
        0.98,
        0.06,
        f"optimal total\n{total:.1f} seconds",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "structured_assignment_matrix")
    plt.close(fig)


def solve_knapsack() -> tuple[object, np.ndarray, np.ndarray, int]:
    sizes = np.array([21, 11, 15, 9, 34, 25, 41, 52])
    values = np.array([22, 12, 16, 10, 35, 26, 42, 53])
    capacity = 100
    constraints = LinearConstraint(A=sizes, lb=0, ub=capacity)
    result = milp(c=-values, constraints=constraints, integrality=np.ones_like(values), bounds=Bounds(0, 1))
    return result, sizes, values, capacity


def plot_milp_knapsack() -> None:
    result, sizes, values, capacity = solve_knapsack()
    selected = result.x > 0.5
    value_density = values / sizes
    total_size = int(sizes[selected].sum())
    total_value = int(values[selected].sum())

    fig, (ax_items, ax_capacity) = plt.subplots(1, 2, figsize=(10.4, 4.85), gridspec_kw={"width_ratios": [1.35, 1.0]})
    colors = np.where(selected, "#16a34a", "#94a3b8")
    ax_items.scatter(sizes, values, s=210, c=colors, edgecolor="#111827", linewidth=0.65)
    for idx, (size, value) in enumerate(zip(sizes, values)):
        ax_items.text(size, value, str(idx + 1), ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax_items.set_title("MILP keeps binary decisions binary")
    ax_items.set_xlabel("item size")
    ax_items.set_ylabel("item value")
    ax_items.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_items.text(
        0.04,
        0.92,
        "green = selected",
        transform=ax_items.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )

    order = np.argsort(-value_density)
    cumulative = 0
    for idx in order:
        if selected[idx]:
            ax_capacity.barh(["capacity"], [sizes[idx]], left=cumulative, color="#16a34a", edgecolor="white", height=0.42)
            ax_capacity.text(cumulative + sizes[idx] / 2, 0, str(idx + 1), ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            cumulative += sizes[idx]
    ax_capacity.barh(["capacity"], [capacity - total_size], left=total_size, color="#e2e8f0", edgecolor="white", height=0.42)
    ax_capacity.axvline(capacity, color="#111827", linewidth=1.1)
    ax_capacity.set_xlim(0, capacity + 8)
    ax_capacity.set_title("Selected size stays within capacity")
    ax_capacity.set_xlabel("size used")
    ax_capacity.grid(True, axis="x", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax_capacity.text(
        0.05,
        0.18,
        f"total size {total_size}/{capacity}\ntotal value {total_value}",
        transform=ax_capacity.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "structured_milp_knapsack")
    plt.close(fig)


def main() -> None:
    plot_linprog_geometry()
    plot_assignment_matrix()
    plot_milp_knapsack()
    print(f"Wrote structured-optimization visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
