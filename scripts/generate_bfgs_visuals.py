from __future__ import annotations

import math
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection
    from scipy.optimize import line_search, minimize
except ModuleNotFoundError as exc:
    mpl = None
    plt = None
    np = None
    LineCollection = None
    line_search = None
    minimize = None
    NUMERIC_IMPORT_ERROR = exc
else:
    NUMERIC_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)

if mpl is not None:
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


def dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def add_vec(a: list[float], b: list[float]) -> list[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def sub_vec(a: list[float], b: list[float]) -> list[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def scale_vec(a: list[float], scalar: float) -> list[float]:
    return [scalar * ai for ai in a]


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[aij + bij for aij, bij in zip(ai, bi)] for ai, bi in zip(a, b)]


def mat_sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[aij - bij for aij, bij in zip(ai, bi)] for ai, bi in zip(a, b)]


def mat_scale(a: list[list[float]], scalar: float) -> list[list[float]]:
    return [[scalar * aij for aij in ai] for ai in a]


def outer(a: list[float], b: list[float]) -> list[list[float]]:
    return [[ai * bj for bj in b] for ai in a]


def eye2() -> list[list[float]]:
    return [[1.0, 0.0], [0.0, 1.0]]


def rosen2_plain(x: list[float]) -> float:
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def rosen2_grad_plain(x: list[float]) -> list[float]:
    return [
        -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
        200.0 * (x[1] - x[0] ** 2),
    ]


def armijo_backtracking_plain(x: list[float], p: list[float], g: list[float], alpha0: float = 1.0) -> float:
    alpha = alpha0
    fx = rosen2_plain(x)
    slope = dot(g, p)
    while rosen2_plain(add_vec(x, scale_vec(p, alpha))) > fx + 1e-4 * alpha * slope:
        alpha *= 0.5
        if alpha < 1e-10:
            break
    return alpha


def bfgs_history_plain(x0: list[float], max_iter: int = 18) -> list[dict[str, object]]:
    """Small dependency-free BFGS trace used for the tutorial SVG."""
    x = [float(x0[0]), float(x0[1])]
    h_inv = eye2()
    history: list[dict[str, object]] = [
        {"x": x[:], "grad": rosen2_grad_plain(x), "h_inv": [row[:] for row in h_inv], "alpha": 0.0}
    ]

    for _ in range(max_iter):
        g = rosen2_grad_plain(x)
        if norm(g) < 1e-8:
            break

        p = scale_vec(mat_vec(h_inv, g), -1.0)
        alpha = armijo_backtracking_plain(x, p, g)
        x_next = add_vec(x, scale_vec(p, alpha))
        g_next = rosen2_grad_plain(x_next)
        s = sub_vec(x_next, x)
        y = sub_vec(g_next, g)
        ys = dot(y, s)
        if ys <= 1e-12:
            break

        rho = 1.0 / ys
        left = mat_sub(eye2(), mat_scale(outer(s, y), rho))
        right = mat_sub(eye2(), mat_scale(outer(y, s), rho))
        h_inv = mat_add(matmul(matmul(left, h_inv), right), mat_scale(outer(s, s), rho))
        x = x_next
        history.append({"x": x[:], "grad": g_next[:], "h_inv": [row[:] for row in h_inv], "alpha": alpha})

    return history


def svg_arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    label: str,
    label_dx: float,
    label_dy: float,
    dashed: bool = False,
    width: float = 5.0,
) -> str:
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    return f"""
      <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" stroke-linecap="round" marker-end="url(#arrow)"{dash}/>
      <text x="{x2 + label_dx:.1f}" y="{y2 + label_dy:.1f}" class="label" fill="{color}">{label}</text>
    """


def scaled_delta(vector: list[float], length: float) -> tuple[float, float]:
    vector_norm = max(norm(vector), 1e-12)
    return length * vector[0] / vector_norm, -length * vector[1] / vector_norm


def plot_secant_update_svg() -> None:
    history = bfgs_history_plain([-1.2, 1.0])
    step_index = min(4, len(history) - 2)
    current = history[step_index]
    following = history[step_index + 1]

    s = sub_vec(following["x"], current["x"])
    y = sub_vec(following["grad"], current["grad"])
    h_next = following["h_inv"]
    mapped = mat_vec(h_next, y)
    secant_error = norm(sub_vec(mapped, s)) / max(norm(s), 1e-12)
    cosine = max(min(dot(s, y) / max(norm(s) * norm(y), 1e-12), 1.0), -1.0)
    angle = math.degrees(math.acos(cosine))

    left_origin = (225.0, 335.0)
    right_origin = (705.0, 360.0)
    s_left = scaled_delta(s, 125.0)
    y_left = scaled_delta(y, 115.0)
    s_right = scaled_delta(s, 108.0)
    y_right = scaled_delta(y, 92.0)
    mapped_right = scaled_delta(mapped, 108.0)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" viewBox="0 0 960 520" role="img" aria-labelledby="title desc">
  <title id="title">BFGS secant update visualization</title>
  <desc id="desc">A Python-generated diagram showing a step vector, a gradient-change vector, and the BFGS inverse Hessian mapping the gradient change back to the step.</desc>
  <defs>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="context-stroke"/>
    </marker>
    <style>
      .title {{ font: 700 25px Inter, Arial, sans-serif; fill: #172033; }}
      .panel-title {{ font: 700 17px Inter, Arial, sans-serif; fill: #172033; }}
      .body {{ font: 14px Inter, Arial, sans-serif; fill: #5f6b7a; }}
      .label {{ font: 700 15px Inter, Arial, sans-serif; }}
      .small {{ font: 12px Inter, Arial, sans-serif; fill: #5f6b7a; }}
      .formula {{ font: 700 18px Inter, Arial, sans-serif; fill: #172033; }}
    </style>
  </defs>
  <rect width="960" height="520" fill="#ffffff"/>
  <text x="36" y="48" class="title">One BFGS update is one curvature lesson</text>
  <text x="36" y="78" class="body">The step s_k says where the parameters moved. The gradient change y_k says how the slope reacted.</text>

  <rect x="36" y="110" width="412" height="330" rx="8" fill="#f8fafc" stroke="#d9e1ea"/>
  <text x="62" y="145" class="panel-title">1. Measure what happened</text>
  <text x="62" y="172" class="body">On this Rosenbrock step, s_k and y_k point in different directions.</text>
  <line x1="84" y1="335" x2="404" y2="335" stroke="#d9e1ea" stroke-width="1"/>
  <line x1="225" y1="404" x2="225" y2="206" stroke="#d9e1ea" stroke-width="1"/>
  <circle cx="{left_origin[0]:.1f}" cy="{left_origin[1]:.1f}" r="5" fill="#172033"/>
  {svg_arrow(left_origin[0], left_origin[1], left_origin[0] + s_left[0], left_origin[1] + s_left[1], "#0f766e", "s_k", 9, -8)}
  {svg_arrow(left_origin[0], left_origin[1], left_origin[0] + y_left[0], left_origin[1] + y_left[1], "#b45309", "y_k", 9, 18, dashed=True)}
  <text x="62" y="414" class="small">Angle between s_k and y_k: {angle:.0f} degrees. The y_k arrow is scaled to fit.</text>

  <rect x="512" y="110" width="412" height="330" rx="8" fill="#f8fafc" stroke="#d9e1ea"/>
  <text x="538" y="145" class="panel-title">2. Store the lesson in H</text>
  <text x="538" y="172" class="body">After the rank-two update, the new inverse-Hessian</text>
  <text x="538" y="192" class="body">model remembers one measured direction:</text>
  <text x="538" y="226" class="formula">H_{{k+1}} y_k = s_k</text>
  <line x1="550" y1="360" x2="886" y2="360" stroke="#d9e1ea" stroke-width="1"/>
  <line x1="705" y1="414" x2="705" y2="238" stroke="#d9e1ea" stroke-width="1"/>
  <circle cx="{right_origin[0]:.1f}" cy="{right_origin[1]:.1f}" r="5" fill="#172033"/>
  {svg_arrow(right_origin[0], right_origin[1], right_origin[0] + y_right[0], right_origin[1] + y_right[1], "#b45309", "input y_k", 11, 18, dashed=True, width=4.0)}
  {svg_arrow(right_origin[0], right_origin[1] + 8.0, right_origin[0] + mapped_right[0], right_origin[1] + 8.0 + mapped_right[1], "#2563eb", "H y_k", -80, -10, dashed=True, width=4.0)}
  {svg_arrow(right_origin[0], right_origin[1], right_origin[0] + s_right[0], right_origin[1] + s_right[1], "#0f766e", "s_k", 11, 16)}
  <text x="538" y="414" class="small">Relative secant error in this generated trace: {secant_error:.1e}.</text>
</svg>
"""
    (ASSET_DIR / "bfgs_secant_update.svg").write_text(svg, encoding="utf-8")


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
    plot_secant_update_svg()
    if NUMERIC_IMPORT_ERROR is None:
        plot_bfgs_path()
        plot_evaluation_efficiency()
    else:
        print(f"Skipped Matplotlib/SciPy visuals: {NUMERIC_IMPORT_ERROR}")
    print(f"Wrote BFGS visuals to {ASSET_DIR}")


if __name__ == "__main__":
    main()
