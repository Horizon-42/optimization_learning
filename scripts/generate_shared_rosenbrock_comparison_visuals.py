from __future__ import annotations

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)

WIDTH = 980
HEIGHT = 660
PLOT_X = 82.0
PLOT_Y = 74.0
PLOT_W = 770.0
PLOT_H = 500.0

X_MIN, X_MAX = -1.55, 1.55
Y_MIN, Y_MAX = -0.35, 2.45
GRID_NX = 150
GRID_NY = 150
LEVELS = [0.1 * (9000.0 ** (i / 23.0)) for i in range(24)]


def rosen2(point: list[float] | tuple[float, float]) -> float:
    x, y = point
    return 100.0 * (y - x * x) ** 2 + (1.0 - x) ** 2


def rosen2_grad(point: list[float]) -> list[float]:
    x, y = point
    return [
        -400.0 * x * (y - x * x) - 2.0 * (1.0 - x),
        200.0 * (y - x * x),
    ]


def rosen2_hess_entries(point: list[float]) -> tuple[float, float, float]:
    x, y = point
    return 1200.0 * x * x - 400.0 * y + 2.0, -400.0 * x, 200.0


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


def outer(a: list[float], b: list[float]) -> list[list[float]]:
    return [[ai * bj for bj in b] for ai in a]


def eye2() -> list[list[float]]:
    return [[1.0, 0.0], [0.0, 1.0]]


def solve_symmetric_2x2(
    a00: float,
    a01: float,
    a11: float,
    b0: float,
    b1: float,
) -> list[float]:
    det = a00 * a11 - a01 * a01
    if abs(det) < 1e-14:
        raise ValueError("singular 2x2 system")
    return [(b0 * a11 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det]


def armijo_backtracking(x: list[float], p: list[float], g: list[float]) -> float:
    alpha = 1.0
    fx = rosen2(x)
    slope = dot(g, p)
    while rosen2(add_vec(x, scale_vec(p, alpha))) > fx + 1e-4 * alpha * slope:
        alpha *= 0.5
        if alpha < 1e-12:
            break
    return alpha


def bfgs_history(x0: list[float], max_iter: int = 80) -> list[list[float]]:
    x = [float(x0[0]), float(x0[1])]
    h_inv = eye2()
    history = [x[:]]

    for _ in range(max_iter):
        g = rosen2_grad(x)
        if norm(g) < 1e-8:
            break

        p = scale_vec(mat_vec(h_inv, g), -1.0)
        alpha = armijo_backtracking(x, p, g)
        x_next = add_vec(x, scale_vec(p, alpha))
        g_next = rosen2_grad(x_next)
        s = sub_vec(x_next, x)
        y = sub_vec(g_next, g)
        ys = dot(y, s)
        if ys <= 1e-12:
            break

        rho = 1.0 / ys
        left = [[eye2()[i][j] - rho * s[i] * y[j] for j in range(2)] for i in range(2)]
        right = [[eye2()[i][j] - rho * y[i] * s[j] for j in range(2)] for i in range(2)]
        updated = matmul(matmul(left, h_inv), right)
        ss = outer(s, s)
        h_inv = [[updated[i][j] + rho * ss[i][j] for j in range(2)] for i in range(2)]

        x = x_next
        history.append(x[:])

    return history


def newton_cg_history(x0: list[float], max_iter: int = 50) -> list[list[float]]:
    x = [float(x0[0]), float(x0[1])]
    history = [x[:]]

    for _ in range(max_iter):
        g = rosen2_grad(x)
        if norm(g) < 1e-8:
            break

        h00, h01, h11 = rosen2_hess_entries(x)
        try:
            p = solve_symmetric_2x2(h00, h01, h11, -g[0], -g[1])
        except ValueError:
            p = scale_vec(g, -1.0)

        if dot(g, p) >= 0:
            p = scale_vec(g, -1.0)

        alpha = armijo_backtracking(x, p, g)
        x = add_vec(x, scale_vec(p, alpha))
        history.append(x[:])

    return history


def hess_mat_vec(hessian: tuple[float, float, float], vector: list[float]) -> list[float]:
    h00, h01, h11 = hessian
    return [h00 * vector[0] + h01 * vector[1], h01 * vector[0] + h11 * vector[1]]


def quadratic_step_model(
    gradient: list[float],
    hessian: tuple[float, float, float],
    step: list[float],
) -> float:
    return dot(gradient, step) + 0.5 * dot(step, hess_mat_vec(hessian, step))


def smallest_eigenvalue_2x2(hessian: tuple[float, float, float]) -> float:
    h00, h01, h11 = hessian
    return 0.5 * (h00 + h11 - math.sqrt((h00 - h11) ** 2 + 4.0 * h01 * h01))


def step_to_boundary(step: list[float], direction: list[float], radius: float) -> float:
    a = dot(direction, direction)
    b = 2.0 * dot(step, direction)
    c = dot(step, step) - radius * radius
    discriminant = max(0.0, b * b - 4.0 * a * c)
    return (-b + math.sqrt(discriminant)) / (2.0 * a)


def exact_trust_region_step(
    gradient: list[float],
    hessian: tuple[float, float, float],
    radius: float,
) -> list[float]:
    h00, h01, h11 = hessian
    lower = max(0.0, -smallest_eigenvalue_2x2(hessian) + 1e-10)

    if lower == 0.0:
        try:
            newton_step = solve_symmetric_2x2(h00, h01, h11, -gradient[0], -gradient[1])
            if norm(newton_step) <= radius:
                return newton_step
        except ValueError:
            pass

    def shifted_step(lam: float) -> list[float]:
        return solve_symmetric_2x2(h00 + lam, h01, h11 + lam, -gradient[0], -gradient[1])

    lo = lower
    hi = max(1.0, 2.0 * lower + 1.0)
    while norm(shifted_step(hi)) > radius:
        hi *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if norm(shifted_step(mid)) > radius:
            lo = mid
        else:
            hi = mid
    return shifted_step(hi)


def truncated_cg_step(
    gradient: list[float],
    hessian: tuple[float, float, float],
    radius: float,
    max_cg_iters: int,
) -> list[float]:
    step = [0.0, 0.0]
    residual = scale_vec(gradient, -1.0)
    direction = residual[:]
    residual_norm_sq = dot(residual, residual)

    if math.sqrt(residual_norm_sq) < 1e-12:
        return step

    for _ in range(max_cg_iters):
        h_direction = hess_mat_vec(hessian, direction)
        curvature = dot(direction, h_direction)
        if curvature <= 0.0:
            tau = step_to_boundary(step, direction, radius)
            return add_vec(step, scale_vec(direction, tau))

        alpha = residual_norm_sq / curvature
        candidate = add_vec(step, scale_vec(direction, alpha))
        if norm(candidate) >= radius:
            tau = step_to_boundary(step, direction, radius)
            return add_vec(step, scale_vec(direction, tau))

        next_residual = sub_vec(residual, scale_vec(h_direction, alpha))
        next_residual_norm_sq = dot(next_residual, next_residual)
        step = candidate
        if math.sqrt(next_residual_norm_sq) < 1e-10:
            return step

        beta = next_residual_norm_sq / residual_norm_sq
        direction = add_vec(next_residual, scale_vec(direction, beta))
        residual = next_residual
        residual_norm_sq = next_residual_norm_sq

    return step


def trust_region_history(
    x0: list[float],
    step_rule,
    initial_radius: float,
    acceptance_threshold: float = 0.15,
    max_radius: float = 2.0,
    max_iter: int = 120,
) -> list[list[float]]:
    x = [float(x0[0]), float(x0[1])]
    radius = initial_radius
    history = [x[:]]

    for _ in range(max_iter):
        gradient = rosen2_grad(x)
        if norm(gradient) < 1e-8:
            break

        hessian = rosen2_hess_entries(x)
        step = step_rule(x, gradient, hessian, radius)
        if norm(step) < 1e-12:
            step = scale_vec(gradient, -radius / norm(gradient))

        predicted_reduction = -quadratic_step_model(gradient, hessian, step)
        actual_reduction = rosen2(x) - rosen2(add_vec(x, step))
        ratio = actual_reduction / predicted_reduction if predicted_reduction > 0.0 else -math.inf

        if ratio > acceptance_threshold and actual_reduction > 0.0:
            x = add_vec(x, step)
            history.append(x[:])

        if ratio < 0.25:
            radius = max(0.25 * radius, 1e-8)
        elif ratio > 0.75 and norm(step) > 0.8 * radius:
            radius = min(2.0 * radius, max_radius)

    return history


def trust_ncg_history(x0: list[float]) -> list[list[float]]:
    def step_rule(
        x: list[float],
        gradient: list[float],
        hessian: tuple[float, float, float],
        radius: float,
    ) -> list[float]:
        max_cg_iters = 1 if rosen2(x) > 5.0 else 2
        return truncated_cg_step(gradient, hessian, radius, max_cg_iters)

    return trust_region_history(x0, step_rule, initial_radius=0.25)


def trust_krylov_history(x0: list[float]) -> list[list[float]]:
    def step_rule(
        _x: list[float],
        gradient: list[float],
        hessian: tuple[float, float, float],
        radius: float,
    ) -> list[float]:
        return truncated_cg_step(gradient, hessian, radius, max_cg_iters=2)

    return trust_region_history(x0, step_rule, initial_radius=0.35, acceptance_threshold=0.1)


def trust_exact_history(x0: list[float]) -> list[list[float]]:
    def step_rule(
        _x: list[float],
        gradient: list[float],
        hessian: tuple[float, float, float],
        radius: float,
    ) -> list[float]:
        return exact_trust_region_step(gradient, hessian, radius)

    return trust_region_history(x0, step_rule, initial_radius=0.75, acceptance_threshold=0.1)


def nelder_mead_history(
    initial_simplex: list[list[float]],
    max_iter: int = 95,
    tol: float = 1e-8,
) -> list[list[list[float]]]:
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    simplex = [[float(x), float(y)] for x, y in initial_simplex]
    history = [[point[:] for point in simplex]]

    for _ in range(max_iter):
        simplex.sort(key=rosen2)
        values = [rosen2(point) for point in simplex]
        mean = sum(values) / len(values)
        if math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) < tol:
            break

        best, second_worst, worst = simplex
        centroid = [(best[0] + second_worst[0]) / 2.0, (best[1] + second_worst[1]) / 2.0]
        reflected = add_vec(centroid, scale_vec(sub_vec(centroid, worst), alpha))
        reflected_value = rosen2(reflected)

        if values[0] <= reflected_value < values[1]:
            simplex[-1] = reflected
        elif reflected_value < values[0]:
            expanded = add_vec(centroid, scale_vec(sub_vec(reflected, centroid), gamma))
            simplex[-1] = expanded if rosen2(expanded) < reflected_value else reflected
        else:
            if reflected_value < values[-1]:
                contracted = add_vec(centroid, scale_vec(sub_vec(reflected, centroid), rho))
                if rosen2(contracted) <= reflected_value:
                    simplex[-1] = contracted
                else:
                    simplex[1:] = [add_vec(best, scale_vec(sub_vec(point, best), sigma)) for point in simplex[1:]]
            else:
                contracted = add_vec(centroid, scale_vec(sub_vec(worst, centroid), rho))
                if rosen2(contracted) < values[-1]:
                    simplex[-1] = contracted
                else:
                    simplex[1:] = [add_vec(best, scale_vec(sub_vec(point, best), sigma)) for point in simplex[1:]]

        history.append([[point[0], point[1]] for point in simplex])

    return history


def sx(x: float) -> float:
    return PLOT_X + (x - X_MIN) / (X_MAX - X_MIN) * PLOT_W


def sy(y: float) -> float:
    return PLOT_Y + (Y_MAX - y) / (Y_MAX - Y_MIN) * PLOT_H


def point_to_svg(point: list[float]) -> tuple[float, float]:
    return sx(point[0]), sy(point[1])


def path_from_points(points: list[list[float]]) -> str:
    if not points:
        return ""
    coords = [point_to_svg(point) for point in points]
    return " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(coords)
    )


def grid_values() -> tuple[list[float], list[float], list[list[float]]]:
    xs = [X_MIN + (X_MAX - X_MIN) * i / (GRID_NX - 1) for i in range(GRID_NX)]
    ys = [Y_MIN + (Y_MAX - Y_MIN) * j / (GRID_NY - 1) for j in range(GRID_NY)]
    values = [[rosen2((x, y)) for x in xs] for y in ys]
    return xs, ys, values


def interpolate(
    p0: tuple[float, float],
    p1: tuple[float, float],
    v0: float,
    v1: float,
    level: float,
) -> tuple[float, float]:
    if abs(v1 - v0) < 1e-14:
        return p0
    t = (level - v0) / (v1 - v0)
    return p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])


def contour_path(xs: list[float], ys: list[float], values: list[list[float]], level: float) -> str:
    segments = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            corners = [
                ((xs[i], ys[j]), values[j][i]),
                ((xs[i + 1], ys[j]), values[j][i + 1]),
                ((xs[i + 1], ys[j + 1]), values[j + 1][i + 1]),
                ((xs[i], ys[j + 1]), values[j + 1][i]),
            ]
            points = []
            for first, second in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                p0, v0 = corners[first]
                p1, v1 = corners[second]
                if (v0 < level <= v1) or (v1 < level <= v0):
                    points.append(interpolate(p0, p1, v0, v1, level))

            if len(points) == 2:
                segments.append(points)
            elif len(points) == 4:
                segments.append(points[:2])
                segments.append(points[2:])

    commands = []
    for segment in segments:
        (x0, y0), (x1, y1) = segment
        commands.append(f"M {sx(x0):.2f} {sy(y0):.2f} L {sx(x1):.2f} {sy(y1):.2f}")
    return " ".join(commands)


def contour_markup() -> str:
    xs, ys, values = grid_values()
    palette = [
        "#e0f2fe",
        "#bae6fd",
        "#7dd3fc",
        "#38bdf8",
        "#22c55e",
        "#84cc16",
        "#eab308",
        "#f97316",
        "#ef4444",
        "#7c3aed",
    ]
    parts = []
    for index, level in enumerate(LEVELS):
        color = palette[min(index * len(palette) // len(LEVELS), len(palette) - 1)]
        width = 0.75 if index % 3 else 1.05
        opacity = 0.38 if index % 3 else 0.58
        parts.append(
            f'<path d="{contour_path(xs, ys, values, level)}" fill="none" '
            f'stroke="{color}" stroke-width="{width}" opacity="{opacity}" />'
        )
    return "\n      ".join(parts)


CONTOURS = contour_markup()


def axis_markup() -> str:
    x_ticks = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    y_ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
    parts = [
        f'<rect x="{PLOT_X}" y="{PLOT_Y}" width="{PLOT_W}" height="{PLOT_H}" fill="#ffffff" stroke="#d9e1ea" />'
    ]
    for tick in x_ticks:
        x = sx(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{PLOT_Y}" x2="{x:.2f}" y2="{PLOT_Y + PLOT_H}" stroke="#e2e8f0" />')
        parts.append(f'<text x="{x:.2f}" y="{PLOT_Y + PLOT_H + 24:.2f}" text-anchor="middle" class="tick">{tick:g}</text>')
    for tick in y_ticks:
        y = sy(tick)
        parts.append(f'<line x1="{PLOT_X}" y1="{y:.2f}" x2="{PLOT_X + PLOT_W}" y2="{y:.2f}" stroke="#e2e8f0" />')
        parts.append(f'<text x="{PLOT_X - 13:.2f}" y="{y + 4:.2f}" text-anchor="end" class="tick">{tick:g}</text>')
    parts.append(f'<text x="{PLOT_X + PLOT_W / 2:.2f}" y="{PLOT_Y + PLOT_H + 54:.2f}" text-anchor="middle" class="axis">x0</text>')
    parts.append(f'<text x="26" y="{PLOT_Y + PLOT_H / 2:.2f}" text-anchor="middle" transform="rotate(-90 26 {PLOT_Y + PLOT_H / 2:.2f})" class="axis">x1</text>')
    return "\n      ".join(parts)


def marker(point: list[float], radius: float, fill: str, label: str, dx: float, dy: float) -> str:
    x, y = point_to_svg(point)
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" '
        'stroke="white" stroke-width="2" />'
        f'<text x="{x + dx:.2f}" y="{y + dy:.2f}" class="label" fill="{fill}">{label}</text>'
    )


def legend_item(x: float, y: float, color: str, text: str) -> str:
    return (
        f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="4" stroke-linecap="round" />'
        f'<text x="{x + 38}" y="{y + 5}" class="small">{text}</text>'
    )


def progress_values_from_points(points: list[list[float]]) -> list[float]:
    best = math.inf
    values = []
    for point in points:
        best = min(best, rosen2(point))
        values.append(best)
    return values


def progress_values_from_simplexes(history: list[list[list[float]]]) -> list[float]:
    best = math.inf
    values = []
    for simplex in history:
        best = min(best, min(rosen2(point) for point in simplex))
        values.append(best)
    return values


def render_progress_svg(series: list[tuple[str, list[float], str]]) -> None:
    width = 980
    height = 690
    plot_x = 84.0
    plot_y = 80.0
    plot_w = 770.0
    plot_h = 330.0
    floor = 1e-16
    x_max = max(len(values) - 1 for _, values, _ in series)
    logs = [math.log10(max(value, floor)) for _, values, _ in series for value in values]
    y_min = math.floor(min(logs))
    y_max = math.ceil(max(logs))

    def px(step: int) -> float:
        return plot_x + step / x_max * plot_w

    def py(value: float) -> float:
        log_value = math.log10(max(value, floor))
        return plot_y + (y_max - log_value) / (y_max - y_min) * plot_h

    grid = []
    for tick in range(int(y_min), int(y_max) + 1):
        y = plot_y + (y_max - tick) / (y_max - y_min) * plot_h
        grid.append(f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" stroke="#e2e8f0" />')
        grid.append(f'<text x="{plot_x - 12:.2f}" y="{y + 4:.2f}" text-anchor="end" class="tick">1e{tick}</text>')

    for tick in [0, 10, 20, 30, 40, 50, 75, x_max]:
        if tick > x_max:
            continue
        x = px(tick)
        grid.append(f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_h}" stroke="#f1f5f9" />')
        grid.append(f'<text x="{x:.2f}" y="{plot_y + plot_h + 24:.2f}" text-anchor="middle" class="tick">{tick}</text>')

    paths = []
    legend = []
    final_rows = []
    for index, (name, values, color) in enumerate(series):
        commands = []
        for step, value in enumerate(values):
            command = "M" if step == 0 else "L"
            commands.append(f"{command} {px(step):.2f} {py(value):.2f}")
        paths.append(
            f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" '
            'stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round" />'
        )
        final_x = px(len(values) - 1)
        final_y = py(values[-1])
        paths.append(f'<circle cx="{final_x:.2f}" cy="{final_y:.2f}" r="5" fill="{color}" stroke="white" stroke-width="2" />')
        legend_y = 106 + index * 28
        legend.append(legend_item(638, legend_y, color, name))
        final_rows.append(
            f'<text x="606" y="{514 + index * 22}" class="small" fill="{color}">{name}: {len(values) - 1} steps, final best f={values[-1]:.1e}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Rosenbrock progress comparison</title>
  <desc id="desc">A Python-generated log-scale comparison of best Rosenbrock objective value by accepted step for Nelder-Mead, BFGS, Newton-CG, trust-ncg, trust-krylov, and trust-exact.</desc>
  <defs>
    <clipPath id="progress-clip">
      <rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" />
    </clipPath>
    <style>
      .title {{ font: 700 24px Inter, Arial, sans-serif; fill: #172033; }}
      .body {{ font: 14px Inter, Arial, sans-serif; fill: #5f6b7a; }}
      .axis {{ font: 700 14px Inter, Arial, sans-serif; fill: #172033; }}
      .tick {{ font: 12px Inter, Arial, sans-serif; fill: #64748b; }}
      .small {{ font: 13px Inter, Arial, sans-serif; fill: #475569; }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="48" y="38" class="title">Rosenbrock objective progress</text>
  <text x="48" y="61" class="body">Best objective value so far on a log scale. The x-axis is accepted outer step or simplex update, not equal compute cost.</text>
  <rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d9e1ea" />
  {"".join(grid)}
  <g clip-path="url(#progress-clip)">
      {"".join(paths)}
  </g>
  <text x="{plot_x + plot_w / 2:.2f}" y="{plot_y + plot_h + 54:.2f}" text-anchor="middle" class="axis">accepted outer step / simplex update</text>
  <text x="27" y="{plot_y + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 27 {plot_y + plot_h / 2:.2f})" class="axis">best f(x) so far</text>
  <rect x="618" y="84" width="230" height="188" rx="8" fill="#ffffff" opacity="0.94" stroke="#d9e1ea" />
  {"".join(legend)}
  <rect x="588" y="486" width="344" height="156" rx="8" fill="#ffffff" opacity="0.94" stroke="#d9e1ea" />
  {"".join(final_rows)}
  <text x="86" y="508" class="small">Read this as convergence shape, not a stopwatch.</text>
  <text x="86" y="531" class="small">Second-order methods spend Hessian or Hessian-vector work;</text>
  <text x="86" y="554" class="small">BFGS spends gradients; Nelder-Mead spends only objective calls.</text>
</svg>
"""
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    output = ASSET_DIR / "rosenbrock_progress_comparison.svg"
    output.write_text(svg, encoding="utf-8")
    print(f"Wrote {output}")


def render_path_svg(
    filename: str,
    title: str,
    subtitle: str,
    path_points: list[list[float]],
    path_color: str,
    finish_label: str,
    info: str,
    extra_markup: str = "",
) -> None:
    path_d = path_from_points(path_points)
    dots = []
    for index, point in enumerate(path_points):
        if index % max(1, len(path_points) // 28) == 0 or index == len(path_points) - 1:
            x, y = point_to_svg(point)
            dots.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.3" fill="{path_color}" opacity="0.82" />')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">{subtitle}</desc>
  <defs>
    <clipPath id="plot-clip">
      <rect x="{PLOT_X}" y="{PLOT_Y}" width="{PLOT_W}" height="{PLOT_H}" />
    </clipPath>
    <style>
      .title {{ font: 700 24px Inter, Arial, sans-serif; fill: #172033; }}
      .body {{ font: 14px Inter, Arial, sans-serif; fill: #5f6b7a; }}
      .axis {{ font: 700 14px Inter, Arial, sans-serif; fill: #172033; }}
      .tick {{ font: 12px Inter, Arial, sans-serif; fill: #64748b; }}
      .label {{ font: 700 13px Inter, Arial, sans-serif; }}
      .small {{ font: 13px Inter, Arial, sans-serif; fill: #475569; }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="48" y="36" class="title">{title}</text>
  <text x="48" y="59" class="body">{subtitle}</text>
  {axis_markup()}
  <g clip-path="url(#plot-clip)">
      {CONTOURS}
      {extra_markup}
      <path d="{path_d}" fill="none" stroke="{path_color}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round" />
      {"".join(dots)}
  </g>
  {marker([1.0, 1.0], 8.5, "#f59e0b", "minimum", 10, -8)}
  {marker(path_points[0], 7.0, "#dc2626", "start", 10, -9)}
  {marker(path_points[-1], 7.0, "#16a34a", finish_label, 10, 17)}
  <rect x="618" y="94" width="218" height="82" rx="8" fill="#ffffff" opacity="0.92" stroke="#d9e1ea" />
  {legend_item(638, 124, path_color, "optimization path")}
  {legend_item(638, 154, "#7dd3fc", "shared Rosenbrock contours")}
  <rect x="96" y="516" width="408" height="42" rx="8" fill="#ffffff" opacity="0.92" stroke="#d9e1ea" />
  <text x="116" y="542" class="small">{info}</text>
</svg>
"""
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    (ASSET_DIR / filename).write_text(svg, encoding="utf-8")
    print(f"Wrote {ASSET_DIR / filename}")


def simplex_markup(history: list[list[list[float]]]) -> str:
    selected = [0, 1, 2, 4, 8, 16, 32, len(history) - 1]
    parts = []
    for index in selected:
        if index >= len(history):
            continue
        simplex = history[index]
        closed = simplex + [simplex[0]]
        path_d = path_from_points(closed)
        opacity = 0.18 if index != len(history) - 1 else 0.34
        parts.append(f'<path d="{path_d}" fill="#38bdf8" fill-opacity="{opacity}" stroke="#1f2937" stroke-width="1.1" opacity="0.72" />')
    return "\n      ".join(parts)


def main() -> None:
    start = [-1.2, 1.0]
    bfgs_points = bfgs_history(start)
    newton_points = newton_cg_history(start)
    trust_ncg_points = trust_ncg_history(start)
    trust_krylov_points = trust_krylov_history(start)
    trust_exact_points = trust_exact_history(start)

    initial_simplex = [[-1.35, 1.65], [-1.05, 2.25], [-0.55, 1.55]]
    nm_history = nelder_mead_history(initial_simplex)
    nm_best_points = [min(simplex, key=rosen2) for simplex in nm_history]
    render_progress_svg(
        [
            ("Newton-CG", progress_values_from_points(newton_points), "#2563eb"),
            ("BFGS", progress_values_from_points(bfgs_points), "#7c3aed"),
            ("Nelder-Mead", progress_values_from_simplexes(nm_history), "#be123c"),
            ("trust-ncg", progress_values_from_points(trust_ncg_points), "#ea580c"),
            ("trust-krylov", progress_values_from_points(trust_krylov_points), "#0f766e"),
            ("trust-exact", progress_values_from_points(trust_exact_points), "#0891b2"),
        ]
    )

    render_path_svg(
        "bfgs_rosenbrock_path.svg",
        "BFGS on the shared Rosenbrock valley",
        "Same contour window used for Nelder-Mead and Newton-CG comparison.",
        bfgs_points,
        "#7c3aed",
        "BFGS finish",
        f"{len(bfgs_points) - 1} BFGS iterations; final f={rosen2(bfgs_points[-1]):.1e}.",
    )
    render_path_svg(
        "newton_cg_rosenbrock_path.svg",
        "Newton-CG on the shared Rosenbrock valley",
        "Same contour window and start as the BFGS comparison figure.",
        newton_points,
        "#2563eb",
        "Newton-CG finish",
        f"{len(newton_points) - 1} Newton-CG outer iterations; final f={rosen2(newton_points[-1]):.1e}.",
    )
    render_path_svg(
        "nelder_mead_rosenbrock_path.svg",
        "Nelder-Mead on the shared Rosenbrock valley",
        "Same contour window used for BFGS and Newton-CG comparison.",
        nm_best_points,
        "#be123c",
        "best vertex finish",
        f"{len(nm_history) - 1} simplex iterations; final best f={rosen2(nm_best_points[-1]):.1e}.",
        extra_markup=simplex_markup(nm_history),
    )
    render_path_svg(
        "trust_ncg_rosenbrock_path.svg",
        "trust-ncg on the shared Rosenbrock valley",
        "Same contour window and start as the Newton-CG comparison figure.",
        trust_ncg_points,
        "#ea580c",
        "trust-ncg finish",
        f"{len(trust_ncg_points) - 1} accepted trust-region steps; final f={rosen2(trust_ncg_points[-1]):.1e}.",
    )
    render_path_svg(
        "trust_krylov_rosenbrock_path.svg",
        "trust-krylov on the shared Rosenbrock valley",
        "Same contour window and start as the trust-ncg comparison figure.",
        trust_krylov_points,
        "#0f766e",
        "trust-krylov finish",
        f"{len(trust_krylov_points) - 1} accepted trust-region steps; final f={rosen2(trust_krylov_points[-1]):.1e}.",
    )
    render_path_svg(
        "trust_exact_rosenbrock_path.svg",
        "trust-exact on the shared Rosenbrock valley",
        "Same contour window and start as the other local minimizer figures.",
        trust_exact_points,
        "#0891b2",
        "trust-exact finish",
        f"{len(trust_exact_points) - 1} accepted trust-region steps; final f={rosen2(trust_exact_points[-1]):.1e}.",
    )


if __name__ == "__main__":
    main()
