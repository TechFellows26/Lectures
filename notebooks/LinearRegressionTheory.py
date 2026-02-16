"""
Linear Regression -- companion notebook to linear-regression.tex

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
    6. Numerical Methods
    7. Bias-Variance Decomposition
    8. Regularization (Ridge, Lasso)
    9. Model Selection
    """)
    return (mo,)


@app.cell
def imports():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import KFold

    np.random.seed(42)
    plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 12})
    return KFold, np, plt


@app.cell
def problem_setup_header(mo):
    mo.md(r"""
    ---
    ## 1. Problem Setup

    **Model:** $Y = a + bX + \varepsilon$

    - $y$ = explained variable
    - $x$ = explanatory variable
    - $a, b \in \mathbb{R}$ = parameters to estimate
    - $\varepsilon$ = noise

    **Goal:** Find estimators $\hat{a}, \hat{b}$ from data $\{(x_i, y_i)\}_{i=1}^n$.
    """)
    return


@app.cell
def problem_setup_synthetic(np):
    _n = 50
    x = np.linspace(0, 5, _n)
    true_a, true_b = 2.0, 3.0
    y = true_a + true_b * x + np.random.normal(0, 2, _n)
    return x, y


@app.cell
def ohm_example_header(mo):
    mo.md(r"""
    ### Example: Ohm's Law

    $V = IR$ - fit line through origin. Slope = resistance $\hat{R}$.
    """)
    return


@app.cell
def ohm_example_viz(np, plt):
    def _():
        _I = np.linspace(0.5, 5, 20)
        _R_true = 10.0
        _V = _R_true * _I + np.random.normal(0, 2, len(_I))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(
            _I,
            _V,
            s=60,
            color="#2ecc71",
            edgecolor="white",
            label="Measurements",
            zorder=3,
        )
        slope_R = np.sum(_I * _V) / np.sum(_I**2)
        _x_line = np.array([0, 5.5])
        ax.plot(
            _x_line,
            slope_R * _x_line,
            "r-",
            lw=2,
            label=f"Fitted: $\\hat{{R}} = {slope_R:.1f}\\,\\Omega$",
        )
        ax.set_xlabel("Current $I$ (A)")
        ax.set_ylabel("Voltage $V$ (V)")
        ax.set_title("Ohm's Law: Slope = Resistance")
        ax.legend()
        ax.set_xlim(0, 5.5)
        ax.set_ylim(0, 60)
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

    **Squared loss:** $\ell(y, \hat{y}) = (y - \hat{y})^2$

    Why vertical distance? We predict $y$ from $x$, so error = vertical gap.

    **Three distance types** to a line:
    - **Vertical** (used in regression)
    - **Horizontal**
    - **Euclidean** (perpendicular)
    """)
    return


@app.cell
def loss_comparison_viz(np, plt):
    def _():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        _xi, _yi = 2.0, 8.0
        _a, _b = 1.5, 3.0
        _x_line = np.array([0, 3])
        _y_line = _a + _b * _x_line

        for ax, _dist_type, _color, _title_suffix in zip(
            axes,
            ["vertical", "horizontal", "euclidean"],
            ["#e74c3c", "#3498db", "#2ecc71"],
            ["Vertical (OLS)", "Horizontal", "Euclidean"],
        ):
            ax.plot(_x_line, _y_line, "k-", lw=2, label="Line")
            ax.scatter([_xi], [_yi], s=120, color=_color, zorder=3, label="Point")
            _y_pred = _a + _b * _xi

            if _dist_type == "vertical":
                ax.plot([_xi, _xi], [_yi, _y_pred], color=_color, lw=3, label="Error")
                ax.axhline(_y_pred, color="gray", ls="--", alpha=0.5)
            elif _dist_type == "horizontal":
                _x_proj = (_yi - _a) / _b
                ax.plot([_xi, _x_proj], [_yi, _yi], color=_color, lw=3, label="Error")
                ax.axvline(_x_proj, color="gray", ls="--", alpha=0.5)
            else:
                _t = (_xi + _b * (_yi - _a)) / (1 + _b**2)
                _xf, _yf = _t, _a + _b * _t
                ax.plot([_xi, _xf], [_yi, _yf], color=_color, lw=3, label="Error")

            ax.set_xlim(0, 3)
            ax.set_ylim(0, 12)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_title(f"Distance: {_title_suffix}")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def bayes_loss_header(mo):
    mo.md(r"""
    ### Why Squared Loss?

    Under squared loss, the optimal predictor is $\mathbb{E}[y|x]$ (conditional expectation).
    We approximate it with a linear function $\hat{y} = a + bx$.
    """)
    return


@app.cell
def bayes_loss_viz(np, plt):
    def _():
        np.random.seed(123)
        _x_vals = np.array([1.0, 2.0, 3.0, 4.0])
        _n_reps = 500
        _a, _b, _sigma = 2.0, 3.0, 1.5
        _means = []
        for _xv in _x_vals:
            _ys = _a + _b * _xv + np.random.normal(0, _sigma, _n_reps)
            _means.append(np.mean(_ys))
        _means = np.array(_means)

        fig, ax = plt.subplots(figsize=(7, 4))
        for _xv in _x_vals:
            _ys = _a + _b * _xv + np.random.normal(0, _sigma, _n_reps)
            ax.scatter(
                np.full(_n_reps, _xv) + 0.05 * np.random.randn(_n_reps),
                _ys,
                alpha=0.1,
                s=5,
                c="#3498db",
            )
        ax.scatter(
            _x_vals,
            _means,
            s=200,
            c="#e74c3c",
            marker="s",
            label="$\\mathbb{E}[y|x]$ (sample mean)",
        )
        _x_fine = np.linspace(0.5, 4.5, 100)
        ax.plot(_x_fine, _a + _b * _x_fine, "g-", lw=2, label="True $y = a + bx$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title("Conditional mean $\\mathbb{E}[y|x]$ across samples")
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

    Minimize empirical risk: $\hat{\mathcal{R}}_n = \frac{1}{n}\sum_{i=1}^n (y_i - a - bx_i)^2$

    **Formulas:**
    $$\hat{b} = \frac{\sum_{i}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i}(x_i - \bar{x})^2} = \frac{\widehat{\text{Cov}}(X,Y)}{\widehat{\text{Var}}(X)}$$
    $$\hat{a} = \bar{y} - \hat{b}\bar{x}$$
    """)
    return


@app.cell
def ols_derivation():
    return


@app.cell
def ols_from_scratch(np, x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov_xy = np.sum((x - x_mean) * (y - y_mean))
    var_x = np.sum((x - x_mean) ** 2)
    b_hat = cov_xy / var_x
    a_hat = y_mean - b_hat * x_mean
    return a_hat, b_hat


@app.cell
def ols_verify_viz(a_hat, b_hat, np, plt, x, y):
    def _():
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, alpha=0.6, s=40, color="#3498db", label="Data")
        _x_line = np.array([x.min(), x.max()])
        ax.plot(
            _x_line,
            a_hat + b_hat * _x_line,
            "r-",
            lw=2,
            label=f"OLS: $\\hat{{y}} = {a_hat:.2f} + {b_hat:.2f}x$",
        )
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title("OLS Fit (implemented from formulas)")
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
    XtX = X.T @ X
    Xty = X.T @ y
    theta_hat = np.linalg.solve(XtX, Xty)
    a_mat, b_mat = theta_hat[0], theta_hat[1]
    return X, a_mat, b_mat


@app.cell
def ols_matrix_check(a_hat, a_mat, b_hat, b_mat, mo):
    mo.md(f"""
    Scalar vs matrix form:
    - $\\hat{{a}}$: {a_hat:.4f} (scalar) vs {a_mat:.4f} (matrix)
    - $\\hat{{b}}$: {b_hat:.4f} (scalar) vs {b_mat:.4f} (matrix)
    """)
    return


@app.cell
def residual_props_header(mo):
    mo.md(r"""
    ---
    ## 4. Properties of Residuals

    Residuals: $\hat{\varepsilon}_i = y_i - \hat{a} - \hat{b}x_i$

    **From first-order conditions:**
    1. $\bar{\varepsilon} = 0$ (zero mean)
    2. $\text{Cov}(X, \varepsilon) = 0$ (uncorrelated with predictor)
    """)
    return


@app.cell
def residual_props_verify(a_hat, b_hat, np, x, y):
    resid = y - a_hat - b_hat * x
    resid_mean = np.mean(resid)
    cov_x_resid = np.mean((x - np.mean(x)) * (resid - np.mean(resid)))
    return (resid,)


@app.cell
def residual_props_viz(plt, resid, x, y):
    def _():
        _fitted = y - resid
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        ax.scatter(x, resid, alpha=0.6, s=40, color="#3498db")
        ax.axhline(
            0, color="red", ls="--", lw=2, label="$\\bar{\\varepsilon} \\approx 0$"
        )
        ax.set_xlabel("$x$")
        ax.set_ylabel("Residual $\\hat{\\varepsilon}$")
        ax.set_title("Residuals vs $x$ (centered at 0)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(_fitted, resid, alpha=0.6, s=40, color="#2ecc71")
        ax.axhline(0, color="red", ls="--", lw=2)
        ax.set_xlabel("Fitted $\\hat{y}$")
        ax.set_ylabel("Residual $\\hat{\\varepsilon}$")
        ax.set_title("Residuals vs Fitted")
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

    - $\mathbf{y} \in \mathbb{R}^n$: response vector
    - $\text{Im}(\mathbf{X})$: column space of design matrix
    - OLS finds the **orthogonal projection** of $\mathbf{y}$ onto $\text{Im}(\mathbf{X})$

    $$\hat{\mathbf{y}} = \mathbf{P}_{\mathbf{X}}\mathbf{y} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

    Residual $\hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to every column of $\mathbf{X}$.
    """)
    return


@app.cell
def geometry_viz_2d(np, plt):
    def _():
        _X_vec = np.array([[1], [1]])
        _y_vec = np.array([3.0, 1.0])
        _P = _X_vec @ np.linalg.solve(_X_vec.T @ _X_vec, _X_vec.T)
        _y_hat_vec = _P @ _y_vec
        _resid_vec = _y_vec - _y_hat_vec

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.quiver(
            0,
            0,
            _X_vec[0],
            _X_vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#e74c3c",
            label="$\\mathbf{X}$ (col)",
        )
        ax.quiver(
            0,
            0,
            _y_vec[0],
            _y_vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#3498db",
            label="$\\mathbf{y}$",
        )
        ax.quiver(
            0,
            0,
            _y_hat_vec[0],
            _y_hat_vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#2ecc71",
            label="$\\hat{\\mathbf{y}} = P_X \\mathbf{y}$",
        )
        ax.quiver(
            _y_hat_vec[0],
            _y_hat_vec[1],
            _resid_vec[0],
            _resid_vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#9b59b6",
            label="$\\hat{\\boldsymbol{\\varepsilon}}$",
        )
        ax.set_xlim(-0.5, 4)
        ax.set_ylim(-0.5, 3)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(
            "Projection: $\\hat{\\mathbf{y}}$ is closest point in $\\text{Im}(\\mathbf{X})$"
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def numerics_header(mo):
    mo.md(r"""
    ---
    ## 6. Numerical Methods

    **Closed form** requires $O(nd^2 + d^3)$. For large $d$, use **gradient descent**.

    Gradient: $\nabla\hat{\mathcal{R}}_n = \frac{2}{n}(\mathbf{X}^\top\mathbf{X}\boldsymbol{\theta} - \mathbf{X}^\top\mathbf{y})$

    Update: $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla\hat{\mathcal{R}}_n$
    """)
    return


@app.cell
def gradient_descent_demo(X, np, y):
    def _ols_gradient(_X, _y, _theta):
        _n = len(_y)
        return (2 / _n) * (_X.T @ _X @ _theta - _X.T @ _y)

    _theta_init = np.zeros(2)
    _eta = 0.01
    _n_iters = 200
    path = [_theta_init.copy()]
    _theta = _theta_init.copy()
    for _ in range(_n_iters):
        _theta = _theta - _eta * _ols_gradient(X, y, _theta)
        path.append(_theta.copy())
    path = np.array(path)
    theta_ols = np.linalg.solve(X.T @ X, X.T @ y)
    return path, theta_ols


@app.cell
def gd_convergence_plot(np, path, plt, theta_ols):
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax.plot(path[:, 0], path[:, 1], "b.-", markersize=2, alpha=0.7)
        ax.scatter(
            [theta_ols[0]],
            [theta_ols[1]],
            s=150,
            c="red",
            marker="*",
            label="OLS solution",
        )
        ax.set_xlabel("$\\hat{a}$")
        ax.set_ylabel("$\\hat{b}$")
        ax.set_title("Gradient descent path in parameter space")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _err = np.linalg.norm(path - theta_ols, axis=1)
        ax.semilogy(_err, "b-", lw=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(
            "$\\|\\boldsymbol{\\theta}_t - \\hat{\\boldsymbol{\\theta}}_{\\text{OLS}}\\|$"
        )
        ax.set_title("Convergence to OLS")
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

    $\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2] = \text{Bias}^2 + \text{Var}$

    Prediction error: $\mathbb{E}[(y_0 - \hat{y}_0)^2] = \sigma^2 + \text{Bias}^2 + \text{Var}$

    - $\sigma^2$: irreducible noise
    - Bias: model misspecification
    - Variance: sensitivity to training data
    """)
    return


@app.cell
def bias_variance_sim(np):
    np.random.seed(42)
    _n_sims = 100
    _n_train = 30
    _x_train = np.linspace(0, 3, _n_train)
    _x_test = np.array([1.5])
    y_test_true = _x_test[0] ** 2
    _biases, _variances = [], []
    for _ in range(_n_sims):
        _y_train = _x_train**2 + np.random.normal(0, 1, _n_train)
        _X_train = np.column_stack([np.ones(_n_train), _x_train])
        _theta = np.linalg.solve(_X_train.T @ _X_train, _X_train.T @ _y_train)
        _X_test = np.column_stack([np.ones(1), _x_test])
        _y_pred = (_X_test @ _theta)[0]
        _biases.append(_y_pred - y_test_true)
        _variances.append(_y_pred)
    preds = np.array(_variances)
    mean_bias = np.mean(_biases)
    var_pred = np.var(preds)
    mse_emp = np.mean((preds - y_test_true) ** 2)
    return mean_bias, mse_emp, preds, var_pred, y_test_true


@app.cell
def bias_variance_viz(
    mean_bias,
    mse_emp,
    np,
    plt,
    preds,
    var_pred,
    y_test_true,
):
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        ax.hist(
            preds, bins=25, density=True, color="#3498db", alpha=0.7, edgecolor="white"
        )
        ax.axvline(y_test_true, color="red", lw=2, label=f"True $y_0 = {y_test_true}$")
        ax.axvline(
            np.mean(preds),
            color="green",
            lw=2,
            ls="--",
            label=f"$\\mathbb{{E}}[\\hat{{y}}_0] = {np.mean(preds):.2f}$",
        )
        ax.set_xlabel("$\\hat{y}_0$")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of predictions (linear fit to quadratic data)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _bias_sq = mean_bias**2
        ax.bar(
            ["Bias²", "Variance", "σ² (noise)"],
            [_bias_sq, var_pred, 1.0],
            color=["#e74c3c", "#3498db", "#95a5a6"],
        )
        ax.set_ylabel("Contribution to MSE")
        ax.set_title(f"MSE decomposition (empirical MSE = {mse_emp:.2f})")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def ridge_header(mo):
    mo.md(r"""
    ---
    ## 8. Ridge Regression ($L_2$)

    $$\hat{\boldsymbol{\theta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$$

    Shrinks all coefficients toward zero. Always invertible for $\lambda > 0$.
    """)
    return


@app.cell
def ridge_viz(np, plt):
    def _():
        np.random.seed(7)
        _n, _d = 80, 6
        _X_r = np.random.randn(_n, _d)
        _true_theta = np.array([3.0, -2.0, 0.0, 1.5, -0.5, 0.0])
        _y_r = _X_r @ _true_theta + np.random.normal(0, 1, _n)

        _alphas = np.logspace(-2, 4, 200)
        _coefs = []
        for _lam in _alphas:
            _theta_r = np.linalg.solve(_X_r.T @ _X_r + _lam * np.eye(_d), _X_r.T @ _y_r)
            _coefs.append(_theta_r)
        _coefs = np.array(_coefs)

        _colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        for _j in range(_d):
            ax.plot(
                _alphas,
                _coefs[:, _j],
                lw=2,
                color=_colors[_j],
                label=f"$\\theta_{_j + 1}$",
            )
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("Coefficient value")
        ax.set_title("Ridge coefficient path (6 features)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _theta1 = np.linspace(-4, 4, 200)
        _theta2 = np.linspace(-4, 4, 200)
        _T1, _T2 = np.meshgrid(_theta1, _theta2)

        _X2 = np.array([[1, 0], [0, 1], [1, 1]])
        _y2 = np.array([3.0, -2.0, 1.0])
        _Loss = np.zeros_like(_T1)
        for _i in range(len(_y2)):
            _Loss += (_y2[_i] - _X2[_i, 0] * _T1 - _X2[_i, 1] * _T2) ** 2

        ax.contour(_T1, _T2, _Loss, levels=15, cmap="Blues", alpha=0.7)
        _circle_t = np.linspace(0, 2 * np.pi, 200)
        for _r, _alpha in [(1.0, 0.8), (2.0, 0.5), (3.0, 0.3)]:
            ax.plot(
                _r * np.cos(_circle_t),
                _r * np.sin(_circle_t),
                color="#e74c3c",
                lw=1.5,
                alpha=_alpha,
            )
        ax.fill(
            1.5 * np.cos(_circle_t),
            1.5 * np.sin(_circle_t),
            color="#e74c3c",
            alpha=0.08,
        )

        _theta_ols_2d = np.linalg.solve(_X2.T @ _X2, _X2.T @ _y2)
        ax.scatter(*_theta_ols_2d, s=80, c="#3498db", zorder=5, label="OLS")
        _lam_ex = 2.0
        _theta_ridge_2d = np.linalg.solve(
            _X2.T @ _X2 + _lam_ex * np.eye(2), _X2.T @ _y2
        )
        ax.scatter(
            *_theta_ridge_2d, s=80, c="#e74c3c", marker="D", zorder=5, label="Ridge"
        )

        ax.set_xlabel("$\\theta_1$")
        ax.set_ylabel("$\\theta_2$")
        ax.set_title("$L_2$ ball + loss contours")
        ax.set_aspect("equal")
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
    ## 8b. Lasso ($L_1$)

    $$\hat{\boldsymbol{\theta}}_{\text{lasso}} = \arg\min_{\boldsymbol{\theta}} \left\{ \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \lambda\|\boldsymbol{\theta}\|_1 \right\}$$

    Produces **exact zeros** -- automatic variable selection. No closed form.
    """)
    return


@app.cell
def lasso_soft_threshold_header(mo):
    mo.md(r"""
    **Soft thresholding** (univariate Lasso): $\hat{\theta} = S_\lambda(y) = \text{sign}(y)(|y|-\lambda)_+$

    The $L_1$ penalty creates a diamond-shaped constraint region whose corners
    lie on the axes -- this is why the Lasso produces exact zeros.
    """)
    return


@app.cell
def lasso_viz(np, plt):
    def _():
        np.random.seed(7)
        _n, _d = 80, 6
        _X_l = np.random.randn(_n, _d)
        _true_theta = np.array([3.0, -2.0, 0.0, 1.5, -0.5, 0.0])
        _y_l = _X_l @ _true_theta + np.random.normal(0, 1, _n)

        _X_std = (_X_l - _X_l.mean(axis=0)) / _X_l.std(axis=0)
        _y_c = _y_l - _y_l.mean()

        def _soft_threshold(z, lam):
            return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

        def _coord_descent(X, y, lam, n_iter=500):
            n, d = X.shape
            theta = np.zeros(d)
            for _ in range(n_iter):
                for j in range(d):
                    r_j = y - X @ theta + X[:, j] * theta[j]
                    z_j = X[:, j] @ r_j / n
                    theta[j] = _soft_threshold(z_j, lam) / (np.sum(X[:, j] ** 2) / n)
            return theta

        _alphas = np.logspace(-3, 1.5, 100)
        _coefs = []
        for _lam in _alphas:
            _coefs.append(_coord_descent(_X_std, _y_c, _lam))
        _coefs = np.array(_coefs)

        _colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        for _j in range(_d):
            ax.plot(
                _alphas,
                _coefs[:, _j],
                lw=2,
                color=_colors[_j],
                label=f"$\\theta_{_j + 1}$",
            )
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("Coefficient value")
        ax.set_title("Lasso coefficient path (6 features)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _theta1 = np.linspace(-4, 4, 200)
        _theta2 = np.linspace(-4, 4, 200)
        _T1, _T2 = np.meshgrid(_theta1, _theta2)

        _X2 = np.array([[1, 0], [0, 1], [1, 1]])
        _y2 = np.array([3.0, -2.0, 1.0])
        _Loss = np.zeros_like(_T1)
        for _i in range(len(_y2)):
            _Loss += (_y2[_i] - _X2[_i, 0] * _T1 - _X2[_i, 1] * _T2) ** 2

        ax.contour(_T1, _T2, _Loss, levels=15, cmap="Blues", alpha=0.7)
        for _r, _alpha in [(1.0, 0.8), (2.0, 0.5), (3.0, 0.3)]:
            _diamond = np.array([[_r, 0], [0, _r], [-_r, 0], [0, -_r], [_r, 0]])
            ax.plot(
                _diamond[:, 0], _diamond[:, 1], color="#2ecc71", lw=1.5, alpha=_alpha
            )
        _diamond_fill = np.array([[1.5, 0], [0, 1.5], [-1.5, 0], [0, -1.5], [1.5, 0]])
        ax.fill(_diamond_fill[:, 0], _diamond_fill[:, 1], color="#2ecc71", alpha=0.08)

        _theta_ols_2d = np.linalg.solve(_X2.T @ _X2, _X2.T @ _y2)
        ax.scatter(*_theta_ols_2d, s=80, c="#3498db", zorder=5, label="OLS")
        ax.scatter(
            0, -0.5, s=80, c="#2ecc71", marker="D", zorder=5, label="Lasso (sparse)"
        )

        ax.set_xlabel("$\\theta_1$")
        ax.set_ylabel("$\\theta_2$")
        ax.set_title("$L_1$ diamond + loss contours")
        ax.set_aspect("equal")
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
    ## 9. $R^2$ and Adjusted $R^2$

    $$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

    **Adjusted $R^2$** penalizes complexity: $\bar{R}^2 = 1 - \frac{n-1}{n-d}(1 - R^2)$

    $R^2$ always increases with more features; $\bar{R}^2$ may decrease.
    """)
    return


@app.cell
def r2_demo(np):
    _n = 100
    _y_rand = np.random.randn(_n)
    r2_vals, adj_r2_vals = [], []
    for _d in range(1, 21):
        _X_d = np.random.randn(_n, _d)
        _X_d = np.column_stack([np.ones(_n), _X_d])
        _theta = np.linalg.solve(_X_d.T @ _X_d, _X_d.T @ _y_rand)
        _y_hat = _X_d @ _theta
        _rss = np.sum((_y_rand - _y_hat) ** 2)
        _tss = np.sum((_y_rand - np.mean(_y_rand)) ** 2)
        _r2 = 1 - _rss / _tss
        _adj_r2 = 1 - (_n - 1) / (_n - _d - 1) * (1 - _r2)
        r2_vals.append(_r2)
        adj_r2_vals.append(_adj_r2)
    return adj_r2_vals, r2_vals


@app.cell
def r2_plot(adj_r2_vals, plt, r2_vals):
    def _():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, 21), r2_vals, "b-o", markersize=5, label="$R^2$")
        ax.plot(range(1, 21), adj_r2_vals, "r-s", markersize=5, label="$\\bar{R}^2$")
        ax.set_xlabel("Number of (random) features")
        ax.set_ylabel("Score")
        ax.set_title("$R^2$ vs Adjusted $R^2$ on noise features")
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
    ## 9b. Cross-Validation

    **$k$-fold CV:** Split data into $k$ folds; for each fold, train on the rest, evaluate on it.
    $$\text{CV}_k = \frac{1}{k}\sum_{i=1}^k \text{MSE}_i$$

    Use CV to select $\lambda$ (Ridge/Lasso) or model complexity.
    """)
    return


@app.cell
def cv_lambda_demo(KFold, X, np, plt, y):
    def _():
        _alphas = np.logspace(-3, 3, 50)
        _kf = KFold(n_splits=5, shuffle=True, random_state=42)
        _mse_cv = []
        for _lam in _alphas:
            _mse_fold = []
            for _train_idx, _val_idx in _kf.split(X):
                _Xt, _Xv = X[_train_idx], X[_val_idx]
                _yt, _yv = y[_train_idx], y[_val_idx]
                _theta = np.linalg.solve(_Xt.T @ _Xt + _lam * np.eye(2), _Xt.T @ _yt)
                _pred = _Xv @ _theta
                _mse_fold.append(np.mean((_yv - _pred) ** 2))
            _mse_cv.append(np.mean(_mse_fold))
        _mse_cv = np.array(_mse_cv)
        _best_idx = np.argmin(_mse_cv)
        best_lam = _alphas[_best_idx]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogx(_alphas, _mse_cv, "b-", lw=2)
        ax.axvline(
            best_lam, color="red", ls="--", label=f"Best $\\lambda = {best_lam:.3f}$"
        )
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("5-fold CV MSE")
        ax.set_title("Selecting $\\lambda$ for Ridge via Cross-Validation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


if __name__ == "__main__":
    app.run()
