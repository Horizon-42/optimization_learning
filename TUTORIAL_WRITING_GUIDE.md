# Tutorial Writing Guide

This project is for learning optimization algorithms, so every algorithm page
should teach the idea as if the reader is meeting it for the first time. Do not
write a page that only lists equations and assumes the reader already knows why
the equations exist.

## Core Method

### 1. Start from the reader's current model

Before introducing the target algorithm, explain the simpler idea it grows from.

For example, a BFGS tutorial should not jump straight to BFGS. It should first
explain Newton's method, why Newton uses curvature, and why computing or
inverting the true Hessian can be too expensive. Only then does BFGS have a
reason to exist.

Good pattern:

1. What problem are we solving?
2. What would the ideal method do?
3. Why is the ideal method expensive or unavailable?
4. What approximation does this algorithm make?
5. What information does one iteration actually learn?

### 2. Name every object before using it

Do not introduce notation and formula at the same time. First say what each
symbol means in plain language.

Bad:

```text
Let phi(alpha)=f(x_k+alpha p_k), so phi'(0)=g_k^T p_k.
```

Better:

```text
The direction p_k is fixed. The current point x_k is fixed. The only thing we
are allowed to vary during line search is the scalar alpha. So we create a
one-dimensional slice of the original objective:

phi(alpha)=f(x_k+alpha p_k).
```

Then derive the derivative.

### 3. Teach the need before the formula

A formula should answer a question the reader already cares about. If the reader
does not know why the formula is needed, the formula feels like a magic spell.

Use this rhythm:

1. State the problem in words.
2. Try the simplest tempting fix.
3. Check whether the fix works.
4. Show what breaks.
5. Add the smallest extra idea needed.
6. Only then show the final formula.
7. Verify the formula does the promised job.

This is the pattern that worked well for the rank-one explanation:

```text
We need Delta_k y_k = s_k - q_k.
If that were the only requirement, a rank-one correction could do it.
Check it directly.
But this correction is usually not symmetric.
BFGS needs symmetry, so we need a more structured correction.
```

The same style must continue through the hardest part. Do not switch back to
textbook mode when the formula becomes complicated.

### 4. Never skip the step that explains a scalar coefficient

Scalars like `rho_k`, `c1`, or `beta` are often the most confusing part.
Do not define them and move on. Explain why they exist.

For BFGS:

```text
r_k = y_k^T s_k is a scalar curvature measurement.
rho_k = 1 / r_k is a normalization trick.
It makes rho_k s_k^T y_k = 1.
```

For inverse BFGS, do not force the expanded correction into the main teaching
path. Teach the compact update first:

```text
rho_k normalizes s_k^T y_k.
E_k = I - rho_k y_k s_k^T erases y_k because E_k y_k = 0.
E_k^T H_k E_k preserves symmetry while making the old model contribute zero on
y_k.
rho_k s_k s_k^T inserts s_k.
```

Only discuss expanded coefficients if the page specifically needs the expanded
rank-two form.

### 5. Distinguish similar-looking objects

If two objects look similar but are not the same, stop and explain the
difference.

Examples:

- `f(x)` is the original objective. `phi(alpha)` is a one-dimensional slice of
  that objective along one search line.
- `nabla f(x)` is a vector derivative with respect to the full input vector.
  `phi'(alpha)` is a scalar derivative with respect to one scalar.
- `g_k = nabla f(x_k)` is the gradient at the current point. It becomes
  `nabla f(z(0))` only because `z(0)=x_k`.
- `H_{k+1} y_k = s_k` is a matrix-vector condition. In multiple dimensions it
  does not mean "divide s by y".

When the reader asks "why are these equal?", do not answer with a slogan. Expand
the definitions until the equality is unavoidable.

### 6. Show the failed shortcut

Readers naturally try shortcuts. A good tutorial respects that instinct and
checks the shortcut.

Examples:

- "Can we optimize alpha directly because alpha is scalar?"
- "Is phi just f with a different variable?"
- "Why not divide s_k by y_k to get H?"
- "Why not use a rank-one correction?"

These questions should appear in the tutorial. They are not distractions; they
are the path to understanding.

### 7. Derive, then summarize

A derivation should end with a short meaning statement.

Example:

```text
So H_{k+1} y_k = s_k.

Meaning: after the update, the new inverse-Hessian approximation gives the
right answer for the latest measured gradient change.
```

The summary is not a replacement for the derivation. It is the reward after the
reader follows the steps.

## Visualization Rules

### Use Python-generated visuals only

For this project, do not manually draw tutorial visuals. If a new visualization
is needed, add or update a Python script under `scripts/`, generate the asset,
and reference the generated file from the HTML.

Good:

```text
scripts/generate_bfgs_visuals.py -> assets/bfgs_rank_two_update.svg
```

Bad:

```text
Manually editing SVG geometry in the asset without updating the generator.
```

### Visuals must answer a teaching question

Do not add decoration. A visualization should clarify one specific confusion.

Examples:

- Line search plot: "Is this curve phi(alpha) or the full f surface?"
- Armijo table: "What exact search method is used, and why is it not binary
  search?"
- Rank-two update diagram: "What old matrix answer is being corrected?"

### Verify the rendered asset

Always render the page after generating assets. Check:

- The image loads with nonzero natural width and height.
- Labels do not overlap.
- Arrowheads do not hide direction.
- Text is readable at desktop and mobile widths.
- The page has no horizontal overflow.
- Equation blocks do not overflow on mobile.

## Mistakes From The BFGS Rewrite

These are concrete mistakes that happened and should be avoided.

### Mistake: jumped to BFGS too quickly

The tutorial originally started as if the reader already knew Newton's method.
That made BFGS feel unmotivated.

Advice: explain Newton first, then explain why BFGS is a quasi-Newton method.

### Mistake: treated `phi(alpha)` as obvious

The line-search section used `phi(alpha)` without making clear that it is a
one-dimensional slice:

```text
phi(alpha)=f(x_k+alpha p_k)
```

Advice: explicitly say `x_k` and `p_k` are frozen, and only `alpha` moves.

### Mistake: skipped chain-rule steps

The derivative of the slice was stated too compactly.

Advice: introduce `z(alpha)=x_k+alpha p_k`, then show component by component:

```text
z_i(alpha)=x_{k,i}+alpha p_{k,i}
dz_i/dalpha=p_{k,i}
phi'(alpha)=sum_i partial f / partial z_i * dz_i/dalpha
```

Only after that should you write:

```text
phi'(alpha)=nabla f(z(alpha))^T p_k
```

### Mistake: blurred `g_k` and `nabla f(z(0))`

The tutorial implied they were automatically the same.

Advice: spell out the equality:

```text
z(0)=x_k
g_k=nabla f(x_k)
therefore nabla f(z(0))=nabla f(x_k)=g_k
```

### Mistake: introduced Armijo without explaining why

Armijo was presented as a condition rather than a reasoned test.

Advice: first explain why "any decrease" is too weak. Then explain sufficient
decrease as requiring progress proportional to the initial downhill slope.

### Mistake: did not name the exact line-search method

The page said line search but did not clearly say which search was being used.

Advice: state the algorithm:

```text
Backtracking line search:
start alpha = 1
if Armijo fails, set alpha = beta alpha
here beta = 0.5
this is not binary search
```

### Mistake: treated the secant condition too abstractly

The secant condition was correct but not enough:

```text
H_{k+1} y_k = s_k
```

Advice: read it in words: the new inverse-Hessian approximation should map the
observed gradient change back to the step that caused it.

### Mistake: did not explain why `s_k / y_k` is invalid

The matrix-vector equation looked like scalar division.

Advice: show the 2D case. A 2x2 matrix has four unknowns, but `Hy=s` gives only
two equations. Therefore many matrices can satisfy the same secant condition.

### Mistake: explained rank one well, then dropped equations for rank two

The "Why A Rank-One Fix Is Tempting" section had a good teaching rhythm:
tempting idea, direct check, then what breaks. The next section lost that rhythm
and jumped to the expanded BFGS formula.

Advice: keep the same teaching rhythm for the hard part. Explain:

1. The naive rank-one fix solves only `Delta_k y_k = s_k - q_k`.
2. It is usually not symmetric.
3. Symmetry requires paired transpose terms.
4. Those terms involve two directions, `s_k` and `q_k = H_k y_k`.
5. That is why the correction is rank two.

### Mistake: forced an expanded coefficient into the main path

This was the biggest teaching failure in the rank-two section. The most
complicated coefficient was introduced right when the reader needed the most
support. The expanded helper coefficient was mathematically valid, but it made
the tutorial feel like symbol bookkeeping.

Advice: teach the compact update first:

- `rho_k` normalizes `s_k^T y_k` so a later product becomes one.
- `E_k = I - rho_k y_k s_k^T` erases `y_k`.
- `E_k^T H_k E_k` keeps symmetry and prevents the old model from answering on
  `y_k`.
- `rho_k s_k s_k^T` inserts the required answer `s_k`.

Only after that should you mention that expanding the compact formula reveals a
rank-two correction.

### Mistake: trusted generated SVGs without rendering them

One SVG contained raw `<` text and did not render as an image. Another visual had
arrowheads so large that direction became unreadable.

Advice: after every generated visual, open the page in a browser and check the
actual rendered result.

### Mistake: ignored mobile equation width until late

Some MathJax blocks used long text inside equations and overflowed on mobile.

Advice: keep equations symbolic and put explanatory prose outside the equation
block.

## Reusable Section Template

Use this template for difficult algorithm sections.

```text
<h3>The question this formula answers</h3>
Plain-language setup.

<h3>The tempting shortcut</h3>
Show the simple idea.
Check it with algebra.

<h3>What breaks</h3>
Explain the failed assumption.

<h3>The extra structure the real method needs</h3>
Explain constraints such as symmetry, positive definiteness, or line-search
safety.

<h3>The final formula</h3>
Define each coefficient by job, not just by symbol.
Show the formula.

<h3>Check the promise</h3>
Apply the formula to the important input and show it gives the required output.

<h3>Meaning</h3>
Summarize what the formula does in one or two sentences.
```

## Pre-Commit Checklist For Tutorials

Before considering a tutorial edit done, verify:

- The page starts with motivation, not the final algorithm.
- Every new symbol is named before being used.
- Every scalar coefficient has a purpose explained before or immediately after
  its definition.
- The hardest formula is preceded by an intuitive problem that it solves.
- At least one tempting but wrong shortcut is addressed when readers are likely
  to ask it.
- Similar objects are explicitly distinguished.
- There is no "therefore" where the previous line does not actually prove it.
- Visualizations are generated by Python scripts, not manually drawn.
- Generated assets are referenced from the page.
- Desktop render works.
- Mobile render has no page overflow.
- Equation blocks have no mobile overflow.
- SVG labels and arrows are readable.

## Tone Standard

Write like a patient tutor, not like a formula sheet.

Good tutorial writing says:

```text
Here is the simple thing you would try.
It almost works.
Here is the exact place it breaks.
The real algorithm adds one more idea to fix that break.
Now the formula has a job you can read.
```

Bad tutorial writing says:

```text
Define rho and A as follows. The BFGS rank-two update is...
```

If a section suddenly becomes dense, slow down exactly there. The hardest step
needs more teaching, not less.
