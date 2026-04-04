# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scikit-learn",
# ]
# ///
"""
Linear Regression — companion notebook to linear-regression.tex

Covers the same material with code and plots instead of proofs.
"""

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def title():
    import marimo as mo

    mo.md(r"""
    # Linear Regression

    Companion to `notes/detailed-notes/linear-regression.tex`.
    Same material, code and plots instead of proofs.

    1. Problem Setup
    2. Choosing the Loss
    3. OLS (univariate & matrix)
    4. Residual Properties
    5. Geometric Interpretation
    6. Numerical Methods (GD & SGD)
    7. Bias-Variance Decomposition
    8. Ridge Regression ($L_2$)
    9. Lasso Regression ($L_1$)
    10. $R^2$ and Model Selection
    11. Cross-Validation
    """)
    return (mo,)


@app.cell
def imports():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import KFold

    matplotlib.rcParams.update({"figure.dpi": 120, "axes.titlesize": 12})
    return KFold, np, plt


@app.cell
def problem_setup_header(mo):
    mo.md(r"""
    ---
    ## 1. Problem Setup

    Let $y$ be an **explained variable** and $x$ an **explanatory variable** related by $y = f(x)$.
    The linear model assumes $f$ is affine:

    $$Y = a + bX + \varepsilon$$

    - $a$ = intercept, $b$ = slope — the parameters we want to estimate
    - $\varepsilon$ = noise term, capturing randomness we don't model

    The noise $\varepsilon$ is not an assumption of linearity — it just acknowledges that real
    data is never perfectly linear. The goal is to recover $a, b$ from observed data
    $\{(x_i, y_i)\}_{i=1}^n$ despite this noise.

    **Univariate** regression: $x \in \mathbb{R}$. **Multivariate**: $x \in \mathbb{R}^d$.
    """)
    return


@app.cell
def problem_setup_synthetic(np):
    _rng = np.random.default_rng(42)
    _n = 50
    x = np.linspace(0, 5, _n)
    y = 2.0 + 3.0 * x + _rng.normal(0, 2, _n)
    return x, y


@app.cell
def ohm_example_header(mo):
    mo.md(r"""
    ### Example: Ohm's Law

    $V = IR$ — fit line through origin. Slope = resistance $\hat{R}$.
    """)
    return


@app.cell
def ohm_example_viz(np, plt):
    def _():
        _rng = np.random.default_rng(0)
        _I = np.linspace(0.5, 5, 20)
        _V = 10.0 * _I + _rng.normal(0, 2, len(_I))
        _R_hat = np.sum(_I * _V) / np.sum(_I**2)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(_I, _V, s=60, color="#2ecc71", edgecolor="white", label="Measurements", zorder=3)
        _xl = np.array([0, 5.5])
        ax.plot(_xl, _R_hat * _xl, "r-", lw=2, label=f"Fitted: $\\hat{{R}} = {_R_hat:.1f}\\,\\Omega$")
        ax.set(xlabel="Current $I$ (A)", ylabel="Voltage $V$ (V)", title="Ohm's Law: Slope = Resistance", xlim=(0, 5.5), ylim=(0, 60))
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def loss_header(mo):
    mo.md(r"""
    ---
    ## 2. Choosing the Loss

    To find $\hat{a}, \hat{b}$ we need a **goodness-of-fit** measure — a loss function.

    Given a point $(x_i, y_i)$ and a candidate line $\hat{y} = a + bx$, there are three
    natural ways to measure the gap between the point and the line:

    - **Vertical distance** $|y_i - (a + bx_i)|$ — difference in $y$-coordinates
    - **Horizontal distance** — difference in $x$-coordinates
    - **Euclidean distance** — perpendicular distance to the line

    We use **vertical distance** because we're trying to predict $y$ from $x$:
    the error is how far our prediction $\hat{y}_i$ is from the observed $y_i$.

    Squaring the vertical gap gives the **squared loss**:
    $$\ell(y, \hat{y}) = (y - \hat{y})^2$$

    Summing over all observations gives the **empirical risk**:
    $$\hat{\mathcal{R}}_n(a, b) = \frac{1}{n}\sum_{i=1}^n (y_i - a - bx_i)^2$$
    """)
    return


@app.cell
def loss_comparison_viz(np, plt):
    def _():
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        _xi, _yi = 3.5, 6.0
        _a, _b = 1.0, 1.0
        _x_line = np.array([0, 7])
        _y_line = _a + _b * _x_line
        _yp = _a + _b * _xi  # line y at xi = 4.5

        for ax, _dtype, _color, _label in zip(
            axes,
            ["vertical", "horizontal", "euclidean"],
            ["#e74c3c", "#3498db", "#2ecc71"],
            ["Vertical (OLS)", "Horizontal", "Euclidean"],
        ):
            ax.plot(_x_line, _y_line, "k-", lw=2, label="Line")
            ax.scatter([_xi], [_yi], s=150, color=_color, zorder=5, label="Point")

            if _dtype == "vertical":
                ax.plot([_xi, _xi], [_yi, _yp], color=_color, lw=3, label="Error")
                ax.annotate("", xy=(_xi, _yp), xytext=(_xi, _yi),
                            arrowprops=dict(arrowstyle="->", color=_color, lw=2))
            elif _dtype == "horizontal":
                _xp = (_yi - _a) / _b
                ax.plot([_xi, _xp], [_yi, _yi], color=_color, lw=3, label="Error")
                ax.annotate("", xy=(_xp, _yi), xytext=(_xi, _yi),
                            arrowprops=dict(arrowstyle="->", color=_color, lw=2))
            else:
                _t = (_xi + _b * (_yi - _a)) / (1 + _b**2)
                _xt, _yt = _t, _a + _b * _t
                ax.plot([_xi, _xt], [_yi, _yt], color=_color, lw=3, label="Error")
                ax.annotate("", xy=(_xt, _yt), xytext=(_xi, _yi),
                            arrowprops=dict(arrowstyle="->", color=_color, lw=2))

            ax.set(xlim=(0, 7), ylim=(0, 9), xlabel="$x$", ylabel="$y$",
                   title=f"Distance: {_label}")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def bayes_loss_header(mo):
    mo.md(r"""
    ### Why Squared Loss? — The Bayes Optimality Argument

    Squared loss has a deeper justification than convenience.

    The **Bayes optimal predictor** — the function minimizing expected risk over the true
    data distribution — is the **conditional expectation**:
    $$f_*(x) = \mathbb{E}[y \mid x]$$

    Intuitively: the best answer to "what do I expect $y$ to be, given I observe $x$?" is
    the conditional mean. No other predictor can beat it in squared loss.

    The **irreducible error** — the best possible risk, even with $f_*$ — is:
    $$\mathcal{R}^* = \mathbb{E}_x\bigl[\text{Var}(y \mid x)\bigr]$$

    This is pure noise: even knowing $x$ exactly, there's inherent randomness in $y$.
    No model can go below this floor.

    When we fit $\hat{y} = a + bx$, we're finding the **best linear approximation** to
    $\mathbb{E}[y \mid x]$. If the true relationship is linear, we get $f_*$ exactly.
    Otherwise, we incur some approximation error on top of the irreducible noise.

    The visualization below shows how $\mathbb{E}[y \mid x]$ is the center of the
    conditional distribution at each $x$.
    """)
    return


@app.cell
def bayes_loss_viz(np, plt):
    def _():
        _rng = np.random.default_rng(123)
        _x_vals = np.array([1.0, 2.0, 3.0, 4.0])
        _n_reps = 500
        _a, _b, _sigma = 2.0, 3.0, 1.5

        fig, ax = plt.subplots(figsize=(7, 4))
        _means = []
        for _xv in _x_vals:
            _ys = _a + _b * _xv + _rng.normal(0, _sigma, _n_reps)
            _means.append(np.mean(_ys))
            ax.scatter(
                np.full(_n_reps, _xv) + 0.05 * _rng.standard_normal(_n_reps),
                _ys, alpha=0.1, s=5, c="#3498db",
            )

        ax.scatter(_x_vals, _means, s=200, c="#e74c3c", marker="s", label=r"$\mathbb{E}[y|x]$ (sample mean)")
        _xf = np.linspace(0.5, 4.5, 100)
        ax.plot(_xf, _a + _b * _xf, "g-", lw=2, label="True $y = a + bx$")
        ax.set(xlabel="$x$", ylabel="$y$", title=r"Conditional mean $\mathbb{E}[y|x]$ across samples")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def ols_header(mo):
    mo.md(r"""
    ---
    ## 3. OLS Estimators (Univariate)

    Differentiating $\hat{\mathcal{R}}_n$ with respect to $a$ and $b$ and setting the
    derivatives to zero gives the **Ordinary Least Squares** (OLS) closed-form solution:

    $$\boxed{\hat{b} = \frac{\sum_{i}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i}(x_i - \bar{x})^2} = \frac{\widehat{\text{Cov}}(X,Y)}{\widehat{\text{Var}}(X)}}$$
    $$\boxed{\hat{a} = \bar{y} - \hat{b}\bar{x}}$$

    **Reading these formulas:**

    - The slope $\hat{b}$ is the ratio of how much $x$ and $y$ vary *together*
      (sample covariance) to how much $x$ varies *alone* (sample variance).
      If $x$ and $y$ always move in the same direction, $\hat{b}$ is large and positive.
    - The intercept $\hat{a}$ pins the line to pass through the centroid $(\bar{x}, \bar{y})$.
      This is a consequence of the first-order condition — OLS always passes through the
      center of mass of the data.

    In **matrix notation** (multivariate case), the design matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$
    (each row is one observation, first column all-ones for the intercept) gives:
    $$\hat{\boldsymbol{\theta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

    The matrix $\mathbf{X}^\top\mathbf{X}$ must be invertible — this requires $\text{rank}(\mathbf{X}) = d$,
    i.e., no column is a linear combination of others (**no perfect multicollinearity**) and $n \geq d$.
    """)
    return


@app.cell
def ols_from_scratch(np, x, y):
    _x_mean = np.mean(x)
    _y_mean = np.mean(y)
    b_hat = np.sum((x - _x_mean) * (y - _y_mean)) / np.sum((x - _x_mean) ** 2)
    a_hat = _y_mean - b_hat * _x_mean
    print(f"â = {a_hat:.4f},  b̂ = {b_hat:.4f}")
    return a_hat, b_hat


@app.cell
def ols_verify_viz(a_hat, b_hat, np, plt, x, y):
    def _():
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, alpha=0.6, s=40, color="#3498db", label="Data")
        _xl = np.array([x.min(), x.max()])
        ax.plot(_xl, a_hat + b_hat * _xl, "r-", lw=2, label=f"OLS: $\\hat{{y}} = {a_hat:.2f} + {b_hat:.2f}x$")
        ax.set(xlabel="$x$", ylabel="$y$", title="OLS Fit (from formulas)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def ols_matrix_header(mo):
    mo.md(r"""
    ---
    ## 3b. OLS in Matrix Form

    **Model:** $\mathbf{y} = \mathbf{X}\boldsymbol{\theta} + \boldsymbol{\varepsilon}$

    **Design matrix** $\mathbf{X}$ has rows $[1, x_i]$ (first column = 1 for intercept).

    **Normal equations:** $\mathbf{X}^\top\mathbf{X}\hat{\boldsymbol{\theta}} = \mathbf{X}^\top\mathbf{y}$

    **Closed form:** $\hat{\boldsymbol{\theta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$
    """)
    return


@app.cell
def ols_matrix_impl(np, x, y):
    X = np.column_stack([np.ones(len(x)), x])
    theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    a_mat, b_mat = theta_hat[0], theta_hat[1]
    return X, a_mat, b_mat


@app.cell
def ols_matrix_check(a_hat, a_mat, b_hat, b_mat, mo):
    mo.md(f"""
    Scalar vs matrix form:
    - $\\hat{{a}}$: {a_hat:.4f} (scalar) vs {a_mat:.4f} (matrix) ✓
    - $\\hat{{b}}$: {b_hat:.4f} (scalar) vs {b_mat:.4f} (matrix) ✓
    """)
    return


@app.cell
def residual_props_header(mo):
    mo.md(r"""
    ---
    ## 4. Properties of Residuals

    The **residuals** are $\hat{\varepsilon}_i = y_i - \hat{a} - \hat{b}x_i$.

    Two properties follow directly from the first-order conditions of the OLS minimization —
    they are **not assumptions**, but algebraic consequences of minimizing squared loss:

    1. $\bar{\varepsilon} = 0$ — residuals have **zero mean**
    2. $\widehat{\text{Cov}}(X, \hat{\varepsilon}) = 0$ — residuals are **uncorrelated with the predictor**

    **Intuition:** The first condition says the model doesn't systematically over- or
    under-predict on average. The second says the *pattern* of errors is unrelated to $x$
    — if there were still correlation, we'd be leaving signal on the table.

    In the matrix formulation, both conditions collapse into the **normal equations**:
    $$\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\theta}}) = \mathbf{0}$$

    This says the residual vector $\hat{\boldsymbol{\varepsilon}}$ is orthogonal to every column
    of $\mathbf{X}$ — exactly the geometric picture in section 5.
    """)
    return


@app.cell
def residual_props_verify(a_hat, b_hat, np, x, y):
    resid = y - a_hat - b_hat * x
    print(f"Mean of residuals:      {np.mean(resid):.2e}  (should be ≈ 0)")
    print(f"Cov(x, residuals):      {np.mean((x - np.mean(x)) * resid):.2e}  (should be ≈ 0)")
    return (resid,)


@app.cell
def residual_props_viz(plt, resid, x, y):
    def _():
        _fitted = y - resid
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        ax.scatter(x, resid, alpha=0.6, s=40, color="#3498db")
        ax.axhline(0, color="red", ls="--", lw=2, label=r"$\bar{\varepsilon} \approx 0$")
        ax.set(xlabel="$x$", ylabel=r"Residual $\hat{\varepsilon}$", title="Residuals vs $x$ (centered at 0)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(_fitted, resid, alpha=0.6, s=40, color="#2ecc71")
        ax.axhline(0, color="red", ls="--", lw=2)
        ax.set(xlabel=r"Fitted $\hat{y}$", ylabel=r"Residual $\hat{\varepsilon}$", title="Residuals vs Fitted")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def geometry_header(mo):
    mo.md(r"""
    ---
    ## 5. Geometric Interpretation

    Think of OLS in observation space $\mathbb{R}^n$ (not feature space):

    - $\mathbf{y} \in \mathbb{R}^n$ is a single vector of all $n$ responses
    - $\text{Im}(\mathbf{X}) = \{\mathbf{X}\boldsymbol{\theta} : \boldsymbol{\theta} \in \mathbb{R}^d\}$ is the
      **column space** of the design matrix — a $d$-dimensional subspace of $\mathbb{R}^n$
    - Since $\mathbf{y}$ generally does not lie in this subspace, OLS finds the closest point in it

    The closest point in a subspace is always the **orthogonal projection**:
    $$\hat{\mathbf{y}} = \mathbf{P}_{\mathbf{X}}\mathbf{y}, \qquad \mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$$

    The matrix $\mathbf{P}_{\mathbf{X}}$ is called the **hat matrix** (it puts a hat on $\mathbf{y}$).
    It is symmetric and idempotent: $\mathbf{P}_{\mathbf{X}}^2 = \mathbf{P}_{\mathbf{X}}$.

    The residual vector $\hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \hat{\mathbf{y}}$ is
    **perpendicular to every column of $\mathbf{X}$** — this is exactly the normal equations.

    **Minimizing Euclidean distance in $\mathbb{R}^n$** is the same as minimizing the sum of
    squared vertical residuals:
    $$\|\mathbf{y} - \hat{\mathbf{y}}\|_2^2 = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

    Below: with 3 observations and $\mathbf{X}$ having 2 columns, $\text{Im}(\mathbf{X})$ is a
    **plane** in $\mathbb{R}^3$. Watch how $\hat{\mathbf{y}}$ (green) lands on the plane and
    $\hat{\boldsymbol{\varepsilon}}$ (dashed) rises perpendicular to it.
    """)
    return


@app.cell
def geometry_viz_3d(np, plt):
    def _():
        # 3 observations, 2 predictors → Im(X) is a plane in R^3
        _x1 = np.array([1.0, 0.5, 0.0])
        _x2 = np.array([0.0, 0.5, 1.0])
        _y  = np.array([0.5, 1.8, 0.8])

        _X = np.column_stack([_x1, _x2])
        _P = _X @ np.linalg.solve(_X.T @ _X, _X.T)
        _y_hat = _P @ _y
        _resid  = _y - _y_hat

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        # column-space plane
        _s = np.linspace(-0.2, 1.3, 6)
        _t = np.linspace(-0.2, 1.3, 6)
        _S, _T = np.meshgrid(_s, _t)
        _Px = _x1[0] * _S + _x2[0] * _T
        _Py = _x1[1] * _S + _x2[1] * _T
        _Pz = _x1[2] * _S + _x2[2] * _T
        ax.plot_surface(_Px, _Py, _Pz, alpha=0.18, color="#3498db", zorder=0)

        def _vec(ax, start, end, color, lw=2.5, ls="-", label=None):
            ax.plot(*zip(start, end), color=color, lw=lw, linestyle=ls,
                    label=label, zorder=5)
            ax.quiver(*end, *(0.001 * (np.array(end) - np.array(start))),
                      color=color, arrow_length_ratio=50, lw=lw, zorder=6)

        O = [0, 0, 0]
        _vec(ax, O, _x1, "#e74c3c", label=r"$\mathbf{x}_1$")
        _vec(ax, O, _x2, "#f39c12", label=r"$\mathbf{x}_2$")
        _vec(ax, O, _y,  "#3498db", lw=3, label=r"$\mathbf{y}$")
        _vec(ax, O, _y_hat, "#2ecc71", lw=3, label=r"$\hat{\mathbf{y}} = P_X\mathbf{y}$")
        _vec(ax, _y_hat, _y, "#9b59b6", lw=2.5, ls="--", label=r"$\hat{\boldsymbol{\varepsilon}}$")

        # right-angle marker at y_hat
        _r = 0.07
        _n = _resid / np.linalg.norm(_resid)
        _u = _x1 / np.linalg.norm(_x1)
        _corner = _y_hat + _r * (_n + _u)
        ax.scatter(*_corner, s=20, color="#9b59b6", zorder=7)

        ax.set_xlabel("$e_1$", labelpad=6)
        ax.set_ylabel("$e_2$", labelpad=6)
        ax.set_zlabel("$e_3$", labelpad=6)
        ax.set_title(
            r"$\hat{\mathbf{y}}$ = projection of $\mathbf{y}$ onto $\mathrm{Im}(\mathbf{X})$",
            fontsize=12, pad=12,
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.view_init(elev=22, azim=-55)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def numerics_header(mo):
    mo.md(r"""
    ---
    ## 6. Numerical Methods: GD and SGD

    The closed-form $\hat{\boldsymbol{\theta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$
    requires forming $\mathbf{X}^\top\mathbf{X}$ in $O(nd^2)$ and inverting it in $O(d^3)$.
    When $d$ is large (millions of features), this is infeasible.

    **Gradient descent (GD)** avoids matrix inversion by taking small steps downhill:
    $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\,\nabla\hat{\mathcal{R}}_n(\boldsymbol{\theta}_t), \qquad \nabla\hat{\mathcal{R}}_n = \frac{2}{n}(\mathbf{X}^\top\mathbf{X}\boldsymbol{\theta} - \mathbf{X}^\top\mathbf{y})$$

    At convergence $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t$, which requires
    $\mathbf{X}^\top\mathbf{X}\hat{\boldsymbol{\theta}} = \mathbf{X}^\top\mathbf{y}$ — the normal equations.
    So **GD converges to the exact OLS solution**.

    But computing the full gradient requires summing over all $n$ observations — expensive
    when $n$ is large. **Stochastic gradient descent (SGD)** approximates it with a single
    random sample $i$ per step:
    $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t\,\mathbf{x}_i(\mathbf{x}_i^\top\boldsymbol{\theta}_t - y_i)$$

    This is an **unbiased** estimate of the gradient:
    $\mathbb{E}_i[\nabla_i(\boldsymbol{\theta})] = \nabla\hat{\mathcal{R}}_n(\boldsymbol{\theta})$.
    SGD converges with a decaying learning rate satisfying
    $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ (e.g., $\eta_t \propto 1/t$).

    **The tradeoff:** GD takes smooth, precise steps. SGD takes noisy steps but each one
    costs $O(d)$ instead of $O(nd)$ — crucial when $n$ is in the millions.
    """)
    return


@app.cell
def gradient_descent_demo(X, np, y):
    def _ols_gradient(_X, _y, _theta):
        _n = len(_y)
        return (2 / _n) * (_X.T @ _X @ _theta - _X.T @ _y)

    _theta = np.zeros(2)
    gd_path = [_theta.copy()]
    for _ in range(200):
        _theta = _theta - 0.01 * _ols_gradient(X, y, _theta)
        gd_path.append(_theta.copy())
    gd_path = np.array(gd_path)

    theta_ols = np.linalg.solve(X.T @ X, X.T @ y)
    return gd_path, theta_ols


@app.cell
def sgd_demo(X, np, y):
    _rng = np.random.default_rng(42)
    _n = len(y)
    _theta = np.zeros(2)
    sgd_path = [_theta.copy()]
    for _t in range(200):
        _i = _rng.integers(0, _n)
        _xi = X[_i : _i + 1]
        _yi = y[_i : _i + 1]
        _eta = 0.05 / (1 + 0.01 * _t)
        _grad = 2 * _xi.T @ (_xi @ _theta - _yi)
        _theta = _theta - _eta * _grad
        sgd_path.append(_theta.copy())
    sgd_path = np.array(sgd_path)
    return (sgd_path,)


@app.cell
def gd_sgd_comparison_plot(gd_path, np, plt, sgd_path, theta_ols):
    def _():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        ax = axes[0]
        ax.plot(gd_path[:, 0], gd_path[:, 1], "b.-", markersize=2, alpha=0.7, label="GD")
        ax.plot(sgd_path[:, 0], sgd_path[:, 1], "r.-", markersize=2, alpha=0.4, label="SGD")
        ax.scatter([theta_ols[0]], [theta_ols[1]], s=150, c="gold", marker="*", zorder=5, label="OLS solution")
        ax.set(xlabel=r"$\hat{a}$", ylabel=r"$\hat{b}$", title="Parameter space trajectories")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _gd_err = np.linalg.norm(gd_path - theta_ols, axis=1)
        _sgd_err = np.linalg.norm(sgd_path - theta_ols, axis=1)
        ax.semilogy(_gd_err, "b-", lw=1.5, label="GD")
        ax.semilogy(_sgd_err, "r-", lw=1, alpha=0.7, label="SGD")
        ax.set(xlabel="Iteration", ylabel=r"$\|\boldsymbol{\theta}_t - \hat{\boldsymbol{\theta}}_{\mathrm{OLS}}\|$", title="Convergence to OLS")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        _gd_steps = np.linalg.norm(np.diff(gd_path, axis=0), axis=1)
        _sgd_steps = np.linalg.norm(np.diff(sgd_path, axis=0), axis=1)
        ax.semilogy(_gd_steps, "b-", lw=1.5, label="GD step size")
        ax.semilogy(_sgd_steps, "r-", lw=1, alpha=0.7, label="SGD step size")
        ax.set(xlabel="Iteration", ylabel="Step size", title="Step size per iteration")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def bias_variance_header(mo):
    mo.md(r"""
    ---
    ## 7. Bias-Variance Decomposition

    The **mean squared error** of any estimator decomposes as:
    $$\text{MSE}(\hat{\theta}) = \mathbb{E}\bigl[(\hat{\theta} - \theta)^2\bigr] = \underbrace{\text{Bias}(\hat{\theta})^2}_{\text{systematic error}} + \underbrace{\text{Var}(\hat{\theta})}_{\text{sampling noise}}$$

    For **prediction** at a new point $x_0$ with response $y_0 = f(x_0) + \varepsilon_0$:
    $$\mathbb{E}\bigl[(y_0 - \hat{y}_0)^2\bigr] = \underbrace{\sigma^2}_{\text{irreducible}} + \underbrace{\text{Bias}(\hat{y}_0)^2}_{\text{misspecification}} + \underbrace{\text{Var}(\hat{y}_0)}_{\text{overfitting}}$$

    The three terms:
    - **$\sigma^2$** — irreducible noise, inherent randomness in $y$ given $x$. No model can beat this.
    - **Bias²** — how far the model's average prediction is from the truth.
      High bias = underfitting (model too simple to capture the true shape).
    - **Variance** — how much the model's prediction changes across different training sets.
      High variance = overfitting (model fits the training noise).

    **The tradeoff:** increasing model complexity (e.g., polynomial degree) reduces bias but
    inflates variance. The optimal model lives at the sweet spot of their sum.

    Use the slider to see this live — watch how a degree-1 model is too rigid (high bias)
    while a degree-12 model is too wiggly (high variance).
    """)
    return


@app.cell
def bias_variance_slider(mo):
    degree_slider = mo.ui.slider(
        start=1, stop=12, value=1, step=1,
        label="Polynomial degree $d$",
    )
    mo.md(f"**Model complexity:** {degree_slider}")
    return (degree_slider,)


@app.cell
def bias_variance_sim(degree_slider, np, plt):
    def _():
        _rng = np.random.default_rng(42)
        _deg = degree_slider.value
        _n_sims = 200
        _n_train = 30
        _x_train = np.linspace(0, 3, _n_train)
        _x_test = 1.5
        _sigma = 1.0

        # Non-polynomial true function — bias never collapses to 0
        def _f(x):
            return 2 * np.sin(x) + 0.5 * x

        _y_test_true = _f(_x_test)

        _preds = np.empty(_n_sims)
        _sample_curves = []
        for _s in range(_n_sims):
            _y_train = _f(_x_train) + _rng.normal(0, _sigma, _n_train)
            _coefs = np.polyfit(_x_train, _y_train, _deg)
            _preds[_s] = np.polyval(_coefs, _x_test)
            if _s < 20:
                _sample_curves.append(_coefs)

        _bias = np.mean(_preds) - _y_test_true
        _var = np.var(_preds)
        _mse = np.mean((_preds - _y_test_true) ** 2)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        _xp = np.linspace(0, 3, 300)
        for _c in _sample_curves:
            ax.plot(_xp, np.polyval(_c, _xp), "b-", alpha=0.15, lw=0.8)
        ax.plot(_xp, _f(_xp), "r-", lw=2, label=r"True $y = 2\sin(x) + 0.5x$")
        ax.scatter([_x_test], [_y_test_true], s=120, c="red", zorder=5, marker="*")
        ax.set(xlabel="$x$", ylabel="$y$", title=f"20 fitted polynomials (degree {_deg})", ylim=(-3, 5))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.hist(_preds, bins=30, density=True, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(_y_test_true, color="red", lw=2, label=f"True $y_0 = {_y_test_true:.2f}$")
        ax.axvline(np.mean(_preds), color="green", lw=2, ls="--", label=f"$\\mathbb{{E}}[\\hat{{y}}_0] = {np.mean(_preds):.2f}$")
        ax.set(xlabel=r"$\hat{y}_0$", ylabel="Density", title=f"Prediction distribution (deg={_deg})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.bar(
            ["Bias²", "Variance", f"σ² ({_sigma**2:.2f})"],
            [_bias**2, _var, _sigma**2],
            color=["#e74c3c", "#3498db", "#95a5a6"],
        )
        ax.set(
            ylabel="Contribution to MSE",
            title=f"MSE = {_mse:.2f} = Bias²({_bias**2:.2f}) + Var({_var:.2f}) + σ²({_sigma**2:.2f})",
        )
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def ridge_header(mo):
    mo.md(r"""
    ---
    ## 8. Ridge Regression ($L_2$ Regularization)

    When $\mathbf{X}^\top\mathbf{X}$ is nearly singular (correlated features) or $d > n$, OLS
    has enormous variance — small changes in the data can flip coefficients wildly.
    Ridge adds a **penalty on the size of the coefficients**:

    $$\hat{\boldsymbol{\theta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\theta}} \bigl\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \lambda\|\boldsymbol{\theta}\|_2^2 \bigr\}$$

    This has a clean closed form — adding $\lambda\mathbf{I}$ to $\mathbf{X}^\top\mathbf{X}$
    makes it always invertible:
    $$\hat{\boldsymbol{\theta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$$

    **What the penalty does geometrically:** it's equivalent to constraining
    $\|\boldsymbol{\theta}\|_2^2 \leq t$ for some budget $t$. The OLS solution (unconstrained minimum
    of the loss ellipses) gets pulled toward the origin until it hits the $L_2$ ball.

    **Bayesian view:** Ridge is the MAP estimate under a Gaussian prior
    $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$ with $\lambda = \sigma^2/\tau^2$.
    Larger $\lambda$ = stronger prior belief that $\boldsymbol{\theta}$ is near zero.

    **Bias-variance view:** Ridge is *biased* (it shrinks away from the true $\boldsymbol{\theta}$),
    but its variance is strictly lower than OLS for any $\lambda > 0$.
    This can reduce MSE even though it introduces bias — a biased estimator with lower
    variance can beat an unbiased one (this is why Gauss-Markov being "BLUE" doesn't mean
    OLS is always optimal for prediction).

    Use the slider to see how $\lambda$ moves the solution along the coefficient path
    and how the constraint circle tightens.
    """)
    return


@app.cell
def ridge_lambda_slider(mo):
    ridge_slider = mo.ui.slider(
        start=-2.0, stop=4.0, value=0.0, step=0.1,
        label=r"$\log_{10}(\lambda)$",
    )
    mo.md(f"**Ridge regularization:** {ridge_slider}")
    return (ridge_slider,)


@app.cell
def ridge_viz(np, plt, ridge_slider):
    def _():
        _rng = np.random.default_rng(7)
        _n, _d = 80, 6
        _X = _rng.standard_normal((_n, _d))
        _true = np.array([3.0, -2.0, 0.0, 1.5, -0.5, 0.0])
        _y = _X @ _true + _rng.normal(0, 1, _n)

        _lam_current = 10 ** ridge_slider.value

        _alphas = np.logspace(-2, 4, 200)
        _coefs = np.array([np.linalg.solve(_X.T @ _X + _a * np.eye(_d), _X.T @ _y) for _a in _alphas])

        _colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        for _j in range(_d):
            ax.plot(_alphas, _coefs[:, _j], lw=2, color=_colors[_j], label=f"$\\theta_{_j+1}$")
        ax.axvline(_lam_current, color="red", ls="--", lw=2, label=f"$\\lambda = {_lam_current:.2f}$")
        ax.set(xscale="log", xlabel=r"$\lambda$", ylabel="Coefficient value", title="Ridge coefficient path")
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        # 2D illustration system — OLS at (2, 1), nicely centered
        _X2 = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        _y2 = np.array([2.0, 1.0, 3.0])
        _ols2 = np.linalg.solve(_X2.T @ _X2, _X2.T @ _y2)   # = [2, 1]
        _ridge2 = np.linalg.solve(_X2.T @ _X2 + _lam_current * np.eye(2), _X2.T @ _y2)

        # Loss contours centered on OLS
        _cx, _cy = _ols2
        _t1 = np.linspace(_cx - 3.5, _cx + 3.5, 300)
        _t2 = np.linspace(_cy - 3.5, _cy + 3.5, 300)
        _T1, _T2 = np.meshgrid(_t1, _t2)
        _L = sum((_y2[i] - _X2[i, 0] * _T1 - _X2[i, 1] * _T2) ** 2 for i in range(3))
        ax.contour(_T1, _T2, _L, levels=12, cmap="Blues", alpha=0.7)

        # L2 ball with radius = norm of ridge solution (active constraint)
        _circ = np.linspace(0, 2 * np.pi, 300)
        _r = np.linalg.norm(_ridge2)
        ax.plot(_r * np.cos(_circ), _r * np.sin(_circ), color="#e74c3c", lw=2)
        ax.fill(_r * np.cos(_circ), _r * np.sin(_circ), color="#e74c3c", alpha=0.08)

        ax.scatter(*_ols2, s=100, c="#3498db", zorder=5, label=f"OLS ({_ols2[0]:.1f}, {_ols2[1]:.1f})")
        ax.scatter(*_ridge2, s=100, c="#e74c3c", marker="D", zorder=5, label=f"Ridge ({_ridge2[0]:.2f}, {_ridge2[1]:.2f})")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$",
               title=f"$L_2$ ball (λ={_lam_current:.2f}) + loss contours", aspect="equal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def lasso_header(mo):
    mo.md(r"""
    ---
    ## 9. Lasso Regression ($L_1$ Regularization)

    Lasso replaces the $L_2$ penalty with an $L_1$ penalty on the coefficients:

    $$\hat{\boldsymbol{\theta}}_{\text{lasso}} = \arg\min_{\boldsymbol{\theta}} \left\{ \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \lambda\|\boldsymbol{\theta}\|_1 \right\}, \qquad \|\boldsymbol{\theta}\|_1 = \sum_j |\theta_j|$$

    Unlike Ridge, Lasso has **no closed form** — the $|\cdot|$ is non-differentiable at zero.
    It is solved iteratively (coordinate descent, proximal gradient methods, LARS).

    **The key difference from Ridge: Lasso produces exact zeros.**

    Why? The constraint region $\{\boldsymbol{\theta} : \|\boldsymbol{\theta}\|_1 \leq t\}$ is a
    **diamond** (cross-polytope) with sharp corners on the axes. When the loss ellipses
    from OLS expand outward, they typically hit a corner first — pinning some $\theta_j = 0$ exactly.

    The $L_2$ ball is smooth with no corners, so the first contact point almost never falls
    on an axis — Ridge shrinks everything but zeros nothing.

    **Lasso as variable selection:** useful when $d$ is large and you believe only a few
    features truly matter (sparsity assumption). The solution is sparse by construction.

    The coordinate-wise solution to the Lasso sub-problem is **soft thresholding** — see below.
    """)
    return


@app.cell
def soft_threshold_header(mo):
    mo.md(r"""
    ### Soft Thresholding — The Lasso's Coordinate Solution

    For the simplest Lasso problem in one dimension:
    $$\min_\theta \; \tfrac{1}{2}(z - \theta)^2 + \lambda|\theta|$$

    the solution is the **soft thresholding operator**:
    $$S_\lambda(z) = \text{sign}(z)\,(|z| - \lambda)_+ = \begin{cases} z - \lambda & \text{if } z > \lambda \\ 0 & \text{if } |z| \leq \lambda \\ z + \lambda & \text{if } z < -\lambda \end{cases}$$

    **Reading this:** if the OLS estimate $z$ is small (within $\lambda$ of zero), the penalty
    dominates and the coefficient is zeroed. Otherwise, the coefficient is shrunk toward zero
    by exactly $\lambda$. This "dead zone" around the origin is what makes Lasso sparse.

    **Contrast with hard thresholding** (used in best subset selection):
    $$H_\lambda(z) = z \cdot \mathbf{1}[|z| > \lambda]$$

    Hard thresholding keeps coefficients *unchanged* above the threshold and zeros them below it.
    It's discontinuous — a tiny change in $z$ near $\lambda$ can flip a coefficient on or off entirely.
    This makes best subset selection NP-hard to optimize globally.

    Soft thresholding is **continuous** and is the proximal operator of the $L_1$ norm,
    making coordinate descent on Lasso tractable and globally convergent.
    """)
    return


@app.cell
def soft_threshold_viz(np, plt):
    def _():
        _z = np.linspace(-4, 4, 500)
        _lams = [0.5, 1.0, 2.0]
        _colors = ["#3498db", "#e74c3c", "#2ecc71"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.plot(_z, _z, "k--", lw=1, alpha=0.4, label="Identity (no shrinkage)")
        for _lam, _c in zip(_lams, _colors):
            _soft = np.sign(_z) * np.maximum(np.abs(_z) - _lam, 0)
            ax.plot(_z, _soft, color=_c, lw=2, label=f"$S_{{{_lam}}}(z)$")
        ax.set(xlabel="$z$ (OLS estimate)", ylabel="$S_\\lambda(z)$ (Lasso estimate)", title="Soft Thresholding")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        ax = axes[1]
        _lam_h = 1.0
        _soft = np.sign(_z) * np.maximum(np.abs(_z) - _lam_h, 0)
        _hard = np.where(np.abs(_z) > _lam_h, _z, 0)
        ax.plot(_z, _z, "k--", lw=1, alpha=0.4, label="Identity")
        ax.plot(_z, _soft, color="#e74c3c", lw=2, label=f"Soft ($\\lambda={_lam_h}$)")
        ax.plot(_z, _hard, color="#3498db", lw=2, ls="--", label=f"Hard ($\\lambda={_lam_h}$)")
        ax.set(xlabel="$z$", ylabel="Thresholded $z$", title="Soft vs Hard Thresholding")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def lasso_lambda_slider(mo):
    lasso_slider = mo.ui.slider(
        start=-3.0, stop=1.5, value=-1.0, step=0.1,
        label=r"$\log_{10}(\lambda)$",
    )
    mo.md(f"**Lasso regularization:** {lasso_slider}")
    return (lasso_slider,)


@app.cell
def lasso_viz(lasso_slider, np, plt):
    def _():
        _rng = np.random.default_rng(7)
        _n, _d = 80, 6
        _X = _rng.standard_normal((_n, _d))
        _true = np.array([3.0, -2.0, 0.0, 1.5, -0.5, 0.0])
        _y = _X @ _true + _rng.normal(0, 1, _n)
        _X_std = (_X - _X.mean(axis=0)) / _X.std(axis=0)
        _y_c = _y - _y.mean()

        def _soft(z, lam):
            return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

        def _coord_descent(X, y, lam, n_iter=500):
            n, d = X.shape
            theta = np.zeros(d)
            for _ in range(n_iter):
                for j in range(d):
                    _r = y - X @ theta + X[:, j] * theta[j]
                    _z = X[:, j] @ _r / n
                    theta[j] = _soft(_z, lam) / (np.sum(X[:, j] ** 2) / n)
            return theta

        _lam_current = 10 ** lasso_slider.value
        _alphas = np.logspace(-3, 1.5, 100)
        _coefs = np.array([_coord_descent(_X_std, _y_c, _a) for _a in _alphas])
        _coef_current = _coord_descent(_X_std, _y_c, _lam_current)
        _n_zero = np.sum(_coef_current == 0)

        _colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        for _j in range(_d):
            ax.plot(_alphas, _coefs[:, _j], lw=2, color=_colors[_j], label=f"$\\theta_{_j+1}$")
        ax.axvline(_lam_current, color="red", ls="--", lw=2, label=f"$\\lambda = {_lam_current:.3f}$")
        ax.set(xscale="log", xlabel=r"$\lambda$", ylabel="Coefficient", title=f"Lasso path — {_n_zero} coefficients zeroed at current λ")
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        # Same 2D illustration system — OLS at (2, 1)
        _X2 = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        _y2 = np.array([2.0, 1.0, 3.0])
        _ols2 = np.linalg.solve(_X2.T @ _X2, _X2.T @ _y2)  # = [2, 1]

        # Lasso on the 2D system via coordinate descent (unnormalized)
        def _lasso2d(X, y, lam, n_iter=1000):
            theta = np.zeros(2)
            for _ in range(n_iter):
                for j in range(2):
                    r = y - X @ theta + X[:, j] * theta[j]
                    z = X[:, j] @ r
                    s = X[:, j] @ X[:, j]
                    theta[j] = np.sign(z) * max(abs(z) - lam, 0) / s
            return theta

        _lasso2 = _lasso2d(_X2, _y2, _lam_current)

        # Loss contours centered on OLS
        _cx, _cy = _ols2
        _t1 = np.linspace(_cx - 3.5, _cx + 3.5, 300)
        _t2 = np.linspace(_cy - 3.5, _cy + 3.5, 300)
        _T1, _T2 = np.meshgrid(_t1, _t2)
        _L = sum((_y2[i] - _X2[i, 0] * _T1 - _X2[i, 1] * _T2) ** 2 for i in range(3))
        ax.contour(_T1, _T2, _L, levels=12, cmap="Blues", alpha=0.7)

        # L1 diamond with radius = L1 norm of Lasso solution (active constraint)
        _r = np.sum(np.abs(_lasso2))
        _r = min(max(_r, 0.05), 4.5)
        _diamond = np.array([[_r, 0], [0, _r], [-_r, 0], [0, -_r], [_r, 0]])
        ax.plot(_diamond[:, 0], _diamond[:, 1], color="#2ecc71", lw=2)
        ax.fill(_diamond[:, 0], _diamond[:, 1], color="#2ecc71", alpha=0.08)

        _n_zero2 = np.sum(_lasso2 == 0)
        ax.scatter(*_ols2, s=100, c="#3498db", zorder=5, label=f"OLS ({_ols2[0]:.1f}, {_ols2[1]:.1f})")
        ax.scatter(*_lasso2, s=100, c="#2ecc71", marker="D", zorder=5,
                   label=f"Lasso ({_lasso2[0]:.2f}, {_lasso2[1]:.2f}){' — sparse' if _n_zero2 > 0 else ''}")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$",
               title=f"$L_1$ diamond (λ={_lam_current:.3f}) + loss contours", aspect="equal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def r2_header(mo):
    mo.md(r"""
    ---
    ## 10. $R^2$ and Adjusted $R^2$

    The **coefficient of determination** measures the proportion of variance in $y$ explained
    by the model. It decomposes total variation (TSS) into explained (ESS) and residual (RSS):

    $$\text{TSS} = \text{ESS} + \text{RSS} \quad \Longrightarrow \quad R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{ESS}}{\text{TSS}}$$

    where:
    - $\text{TSS} = \sum_i (y_i - \bar{y})^2$ — total variance of $y$ (baseline: always predict $\bar{y}$)
    - $\text{RSS} = \sum_i (y_i - \hat{y}_i)^2$ — residual variance after the model
    - $\text{ESS} = \sum_i (\hat{y}_i - \bar{y})^2$ — variance explained by predictions

    So $R^2 = 0$ means the model is no better than always predicting the mean.
    $R^2 = 1$ means perfect fit. For OLS with an intercept, $R^2 = \text{Corr}(y, \hat{y})^2$.

    **The problem:** adding any feature, even pure noise, can only decrease RSS (OLS minimizes it),
    so $R^2$ always increases or stays flat. It cannot detect overfitting.

    **Adjusted $R^2$** corrects for this by penalizing the number of parameters $d$:
    $$\bar{R}^2 = 1 - \frac{n-1}{n-d}(1 - R^2)$$

    The factor $\frac{n-1}{n-d}$ grows as you add parameters, so $\bar{R}^2$ can actually
    *decrease* when you add a feature that doesn't improve fit enough to justify its cost.
    """)
    return


@app.cell
def r2_demo(np):
    _rng = np.random.default_rng(0)
    _n = 100
    _y = _rng.standard_normal(_n)
    r2_vals, adj_r2_vals = [], []
    for _d in range(1, 21):
        _X = _rng.standard_normal((_n, _d))
        _X = np.column_stack([np.ones(_n), _X])
        _theta = np.linalg.solve(_X.T @ _X, _X.T @ _y)
        _rss = np.sum((_y - _X @ _theta) ** 2)
        _tss = np.sum((_y - np.mean(_y)) ** 2)
        _r2 = 1 - _rss / _tss
        r2_vals.append(_r2)
        adj_r2_vals.append(1 - (_n - 1) / (_n - _d - 1) * (1 - _r2))
    return adj_r2_vals, r2_vals


@app.cell
def r2_plot(adj_r2_vals, plt, r2_vals):
    def _():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, 21), r2_vals, "b-o", markersize=5, label="$R^2$")
        ax.plot(range(1, 21), adj_r2_vals, "r-s", markersize=5, label=r"$\bar{R}^2$")
        ax.set(xlabel="Number of (random noise) features", ylabel="Score", title="$R^2$ vs Adjusted $R^2$ — adding noise features")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def cv_header(mo):
    mo.md(r"""
    ---
    ## 11. Cross-Validation for Model Selection

    $R^2$ on training data is **optimistically biased** — the model was fit to that data, so
    of course it looks good. We need an honest estimate of **out-of-sample** error.

    **$k$-fold cross-validation** provides this without throwing away data:

    1. Split the data into $k$ equal folds
    2. For each fold $i$: train on all other $k-1$ folds, evaluate on fold $i$
    3. Average the $k$ test errors:

    $$\text{CV}_k = \frac{1}{k}\sum_{i=1}^k \text{MSE}_i$$

    This estimates how the model generalizes to unseen data.

    **Choosing $k$:**
    - **Large $k$ (e.g., leave-one-out, $k=n$):** nearly unbiased estimate of test error,
      but high variance and expensive to compute
    - **$k = 5$ or $k = 10$:** the practical standard — good bias-variance balance

    **Using CV to select $\lambda$:** fit the model for many $\lambda$ values on a log scale,
    pick the $\lambda$ minimizing CV error. The plot below shows how training MSE always
    decreases with $\lambda \to 0$ (OLS), while CV error has a U-shape — too small and
    we overfit, too large and we underfit.
    """)
    return


@app.cell
def cv_lambda_demo(KFold, X, np, plt, y):
    def _():
        _alphas = np.logspace(-3, 3, 50)
        _kf = KFold(n_splits=5, shuffle=True, random_state=42)
        _mse_cv = []
        for _lam in _alphas:
            _fold_mse = []
            for _tr, _va in _kf.split(X):
                _theta = np.linalg.solve(X[_tr].T @ X[_tr] + _lam * np.eye(2), X[_tr].T @ y[_tr])
                _fold_mse.append(np.mean((y[_va] - X[_va] @ _theta) ** 2))
            _mse_cv.append(np.mean(_fold_mse))
        _mse_cv = np.array(_mse_cv)
        _best = _alphas[np.argmin(_mse_cv)]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogx(_alphas, _mse_cv, "b-", lw=2)
        ax.axvline(_best, color="red", ls="--", label=f"Best $\\lambda = {_best:.3f}$")
        ax.set(xlabel=r"$\lambda$", ylabel="5-fold CV MSE", title=r"Selecting $\lambda$ for Ridge via Cross-Validation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


if __name__ == "__main__":
    app.run()
