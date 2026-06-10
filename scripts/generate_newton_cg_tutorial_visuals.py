from __future__ import annotations

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)


def rosen2(x: float, y: float) -> float:
    return 100.0 * (y - x * x) ** 2 + (1.0 - x) ** 2


def rosen2_grad(x: float, y: float) -> tuple[float, float]:
    return (
        -400.0 * x * (y - x * x) - 2.0 * (1.0 - x),
        200.0 * (y - x * x),
    )


def rosen2_hess_entries(x: float, y: float) -> tuple[float, float, float]:
    return (1200.0 * x * x - 400.0 * y + 2.0, -400.0 * x, 200.0)


def solve_symmetric_2x2(
    a00: float,
    a01: float,
    a11: float,
    b0: float,
    b1: float,
) -> tuple[float, float]:
    det = a00 * a11 - a01 * a01
    if abs(det) < 1e-14:
        raise ValueError("singular 2x2 system")
    return ((b0 * a11 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det)


def golden_section_minimize(
    f,
    lower: float,
    upper: float,
    iterations: int = 80,
) -> tuple[float, float]:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    f_left = f(left)
    f_right = f(right)

    for _ in range(iterations):
        if f_left < f_right:
            upper = right
            right = left
            f_right = f_left
            left = upper - ratio * (upper - lower)
            f_left = f(left)
        else:
            lower = left
            left = right
            f_left = f_right
            right = lower + ratio * (upper - lower)
            f_right = f(right)

    alpha = 0.5 * (lower + upper)
    return alpha, f(alpha)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def svg_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    start = points[0]
    rest = " ".join(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return f"M {start[0]:.2f} {start[1]:.2f} {rest}"


def make_line_search_anatomy() -> None:
    x0, y0 = -1.1, 1.2
    gx, gy = rosen2_grad(x0, y0)
    h00, h01, h11 = rosen2_hess_entries(x0, y0)
    px, py = solve_symmetric_2x2(h00, h01, h11, -gx, -gy)

    f0 = rosen2(x0, y0)
    g_dot_p = gx * px + gy * py
    p_h_p = h00 * px * px + 2.0 * h01 * px * py + h11 * py * py

    def phi(alpha: float) -> float:
        return rosen2(x0 + alpha * px, y0 + alpha * py)

    def model(alpha: float) -> float:
        return f0 + alpha * g_dot_p + 0.5 * alpha * alpha * p_h_p

    accepted_alpha, accepted_value = golden_section_minimize(phi, 0.0, 1.2)
    full_value = phi(1.0)
    model_full = model(1.0)

    alphas = [1.2 * i / 160 for i in range(161)]
    true_values = [phi(alpha) for alpha in alphas]
    model_values = [model(alpha) for alpha in alphas]
    y_min = min(min(true_values), min(model_values))
    y_max = max(max(true_values), min(full_value, 28.0))

    width = 980
    height = 520
    plot_x = 72
    plot_y = 72
    plot_w = 610
    plot_h = 330

    def sx(alpha: float) -> float:
        return plot_x + alpha / 1.2 * plot_w

    def sy(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        return plot_y + (y_max - clipped) / (y_max - y_min) * plot_h

    true_points = [(sx(alpha), sy(value)) for alpha, value in zip(alphas, true_values)]
    model_points = [(sx(alpha), sy(value)) for alpha, value in zip(alphas, model_values)]

    ticks = [0.0, accepted_alpha, 1.0]
    y_ticks = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    grid_lines = []
    for value in y_ticks:
        y = sy(value)
        grid_lines.append(
            f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" '
            'stroke="#e2e8f0" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{plot_x - 12}" y="{y + 4:.2f}" text-anchor="end" '
            'font-size="13" fill="#64748b">'
            f"{value:g}</text>"
        )

    x_tick_markup = []
    for alpha in ticks:
        x = sx(alpha)
        x_tick_markup.append(
            f'<line x1="{x:.2f}" y1="{plot_y + plot_h}" x2="{x:.2f}" '
            f'y2="{plot_y + plot_h + 7}" stroke="#64748b" stroke-width="1" />'
        )
        label = "accepted" if abs(alpha - accepted_alpha) < 1e-8 else f"{alpha:g}"
        x_tick_markup.append(
            f'<text x="{x:.2f}" y="{plot_y + plot_h + 27}" text-anchor="middle" '
            f'font-size="13" fill="#475569">{label}</text>'
        )

    def marker(alpha: float, value: float, color: str, label: str, dy: float) -> str:
        x = sx(alpha)
        y = sy(value)
        return (
            f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" '
            f'y2="{plot_y + plot_h}" stroke="{color}" stroke-width="1.4" '
            'stroke-dasharray="5 6" />'
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{color}" '
            'stroke="white" stroke-width="2" />'
            f'<text x="{x + 8:.2f}" y="{y + dy:.2f}" font-size="13" '
            f'fill="{color}" font-weight="700">{label}</text>'
        )

    step_start = (760, 172)
    step_end = (900, 172)
    accepted = (
        step_start[0] + accepted_alpha * (step_end[0] - step_start[0]),
        step_start[1],
    )

    anatomy = f"""
      <rect x="720" y="84" width="220" height="318" rx="10" fill="#f8fafc" stroke="#d9e1ea" />
      <text x="740" y="118" font-size="19" fill="#172033" font-weight="800">One outer iteration</text>
      <text x="740" y="147" font-size="14" fill="#475569">1. Build a quadratic model at x_k.</text>
      <line x1="{step_start[0]}" y1="{step_start[1]}" x2="{step_end[0]}" y2="{step_end[1]}"
        stroke="#94a3b8" stroke-width="5" stroke-linecap="round" />
      <circle cx="{step_start[0]}" cy="{step_start[1]}" r="8" fill="#172033" />
      <circle cx="{step_end[0]}" cy="{step_end[1]}" r="8" fill="#dc2626" />
      <circle cx="{accepted[0]:.2f}" cy="{accepted[1]:.2f}" r="9" fill="#16a34a" stroke="white" stroke-width="2" />
      <text x="748" y="203" font-size="13" fill="#172033">x_k</text>
      <text x="842" y="203" font-size="13" fill="#16a34a">line-search step</text>
      <text x="842" y="158" font-size="13" fill="#dc2626">full Newton step</text>
      <text x="740" y="244" font-size="14" fill="#475569">2. CG solves H_k p approx -g_k</text>
      <text x="740" y="270" font-size="14" fill="#475569">   using products H_k v.</text>
      <text x="740" y="307" font-size="14" fill="#475569">3. Line search tries alpha values.</text>
      <text x="740" y="333" font-size="14" fill="#475569">4. Accept x_k + alpha p only</text>
      <text x="740" y="359" font-size="14" fill="#475569">   after real decrease appears.</text>
    """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Newton-CG line search anatomy</title>
  <desc id="desc">A scripted plot comparing the true Rosenbrock objective along a Newton direction with the local quadratic model, plus a small diagram of one outer iteration.</desc>
  <rect width="100%" height="100%" fill="white" />
  <text x="54" y="38" font-size="24" fill="#172033" font-weight="800">Newton-CG: direction first, step length second</text>
  <text x="54" y="61" font-size="14" fill="#64748b">At x_k = (-1.1, 1.2), the model wants alpha = 1, but the true objective prefers a shorter line-search step.</text>

  <rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d9e1ea" />
  {"".join(grid_lines)}
  <path d="{svg_path(model_points)}" fill="none" stroke="#f59e0b" stroke-width="3" stroke-dasharray="8 7" />
  <path d="{svg_path(true_points)}" fill="none" stroke="#2563eb" stroke-width="3" />
  {marker(0.0, f0, "#172033", "start", -10)}
  {marker(accepted_alpha, accepted_value, "#16a34a", f"accepted alpha {accepted_alpha:.2f}", -12)}
  {marker(1.0, full_value, "#dc2626", "full step overshoots", 16)}
  {"".join(x_tick_markup)}
  <text x="{plot_x + plot_w / 2:.2f}" y="{plot_y + plot_h + 54}" text-anchor="middle" font-size="14" fill="#475569">alpha along the Newton-CG direction p</text>
  <text x="18" y="{plot_y + plot_h / 2:.2f}" transform="rotate(-90 18 {plot_y + plot_h / 2:.2f})" text-anchor="middle" font-size="14" fill="#475569">objective value</text>
  <line x1="500" y1="96" x2="535" y2="96" stroke="#2563eb" stroke-width="3" />
  <text x="543" y="100" font-size="14" fill="#475569">true f(x_k + alpha p)</text>
  <line x1="500" y1="119" x2="535" y2="119" stroke="#f59e0b" stroke-width="3" stroke-dasharray="8 7" />
  <text x="543" y="123" font-size="14" fill="#475569">quadratic model</text>
  <text x="{sx(1.0) - 78:.2f}" y="{sy(model_full) - 12:.2f}" font-size="13" fill="#b45309">model minimum</text>
  {anatomy}
</svg>
"""
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"

    output = ASSET_DIR / "newton_cg_line_search_anatomy.svg"
    output.write_text(svg, encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    make_line_search_anatomy()


if __name__ == "__main__":
    main()
