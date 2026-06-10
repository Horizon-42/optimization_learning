from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.special import j1


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


def scalar_objective(x: float) -> float:
    return (x - 2.0) * (x + 1.0) ** 2


def record_minimize_scalar(
    func: Callable[[float], float],
    *,
    method: str,
    **kwargs: object,
) -> tuple[OptimizeResult, np.ndarray]:
    history: list[tuple[float, float]] = []

    def wrapped(x: float) -> float:
        value = float(func(float(x)))
        history.append((float(x), value))
        return value

    result = minimize_scalar(wrapped, method=method, **kwargs)
    return result, np.array(history)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ASSET_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def plot_brent_golden_search() -> None:
    bracket = (-1.5, 0.0, 2.2)
    methods = [
        ("brent", "#2563eb", "Brent"),
        ("golden", "#f59e0b", "Golden section"),
    ]
    xs = np.linspace(-1.72, 2.42, 560)
    ys = scalar_objective(xs)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharex=True, sharey=True)
    for ax, (method, color, label) in zip(axes, methods):
        result, history = record_minimize_scalar(
            scalar_objective,
            method=method,
            bracket=bracket,
            options={"xtol": 1.0e-10},
        )
        order = np.arange(1, history.shape[0] + 1)
        ax.plot(xs, ys, color="#111827", linewidth=2.0, label="$f(x)$")
        ax.scatter(
            [bracket[0], bracket[1], bracket[2]],
            [scalar_objective(v) for v in bracket],
            s=74,
            marker="^",
            color="#64748b",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label="initial bracket",
        )
        sc = ax.scatter(
            history[:, 0],
            history[:, 1],
            c=order,
            cmap="viridis",
            s=48,
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
            label="function probes",
        )
        ax.scatter(
            [result.x],
            [result.fun],
            marker="*",
            s=210,
            color=color,
            edgecolor="#111827",
            linewidth=0.55,
            zorder=6,
            label="reported minimizer",
        )
        ax.axvline(1.0, color="#16a34a", linewidth=1.2, linestyle="--", alpha=0.85)
        ax.set_title(f"{label}: {result.nfev} evaluations")
        ax.set_xlabel("$x$")
        ax.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
        ax.text(
            0.04,
            0.08,
            rf"$x={result.x:.8f}$" + "\n" + rf"$f(x)={result.fun:.3f}$",
            transform=ax.transAxes,
            fontsize=9,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
        )
    axes[0].set_ylabel("$f(x)=(x-2)(x+1)^2$")
    axes[0].legend(frameon=False, loc="upper left")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.024, pad=0.02)
    cbar.set_label("evaluation order")
    fig.suptitle("Bracketed scalar minimization probes the one-dimensional objective", y=1.02)
    save_figure(fig, "univariate_brent_golden_search")
    plt.close(fig)


def plot_brent_golden_convergence() -> None:
    bracket = (-1.5, 0.0, 2.2)
    methods = [
        ("brent", "#2563eb", "Brent"),
        ("golden", "#f59e0b", "Golden section"),
    ]
    f_star = scalar_objective(1.0)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for method, color, label in methods:
        result, history = record_minimize_scalar(
            scalar_objective,
            method=method,
            bracket=bracket,
            options={"xtol": 1.0e-10},
        )
        best_so_far = np.minimum.accumulate(history[:, 1])
        gap = np.maximum(best_so_far - f_star, 1.0e-15)
        ax.semilogy(
            np.arange(1, gap.size + 1),
            gap,
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=4.2,
            label=f"{label} ({result.nfev} evaluations)",
        )

    ax.set_title("Brent usually reaches the scalar minimum with fewer probes")
    ax.set_xlabel("function evaluation count")
    ax.set_ylabel(r"best objective gap $f_\mathrm{best}-f^\star$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.04,
        0.09,
        r"true minimizer: $x^\star=1$, $f^\star=-4$",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "univariate_brent_golden_convergence")
    plt.close(fig)


def plot_bounded_bessel() -> None:
    bounds = (4.0, 7.0)
    result, history = record_minimize_scalar(
        j1,
        method="bounded",
        bounds=bounds,
        options={"xatol": 1.0e-10},
    )
    xs = np.linspace(3.2, 7.7, 620)
    ys = j1(xs)
    order = np.arange(1, history.shape[0] + 1)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(xs, ys, color="#111827", linewidth=2.0, label=r"$J_1(x)$")
    ax.axvspan(bounds[0], bounds[1], color="#dbeafe", alpha=0.72, label="feasible interval")
    ax.axvline(bounds[0], color="#2563eb", linewidth=1.2, linestyle="--")
    ax.axvline(bounds[1], color="#2563eb", linewidth=1.2, linestyle="--")
    sc = ax.scatter(
        history[:, 0],
        history[:, 1],
        c=order,
        cmap="plasma",
        s=56,
        edgecolor="white",
        linewidth=0.55,
        zorder=5,
        label="bounded probes",
    )
    ax.scatter(
        [result.x],
        [result.fun],
        marker="*",
        s=240,
        color="#16a34a",
        edgecolor="#111827",
        linewidth=0.6,
        zorder=6,
        label="bounded minimizer",
    )
    ax.set_title("Bounded scalar minimization respects a fixed interval")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$J_1(x)$")
    ax.grid(True, linestyle=":", linewidth=0.7, color="#cbd5e1")
    ax.legend(frameon=False, loc="upper right")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("evaluation order")
    ax.text(
        0.05,
        0.08,
        rf"bounds $[{bounds[0]:.0f}, {bounds[1]:.0f}]$"
        + "\n"
        + rf"$x_\mathrm{{min}}={result.x:.6f}$"
        + "\n"
        + f"{result.nfev} function evaluations",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    save_figure(fig, "univariate_bounded_bessel")
    plt.close(fig)


def main() -> None:
    plot_brent_golden_search()
    plot_brent_golden_convergence()
    plot_bounded_bessel()
    print(f"Wrote univariate scalar-minimizer visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
