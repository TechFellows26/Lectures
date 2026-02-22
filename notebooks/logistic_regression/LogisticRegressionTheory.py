"""
Logistic Regression — companion notebook to logistic-regression.tex

Covers the same material with code and plots instead of proofs.
"""

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def title():
    import marimo as mo

    mo.md(r"""
    # Logistic Regression

    Companion to `notes/detailed-notes/logistic-regression.tex`.
    Same material, code and plots instead of proofs.

    1. Problem Setup
    2. The Logistic Transformation
    3. Decision Boundaries
    4. Choosing the Loss
    5. Maximum Likelihood
    6. Gradient & Optimization
    7. Regularization (L1 / L2)
    """)
    return (mo,)


@app.cell
def imports():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams.update({"figure.dpi": 120, "axes.titlesize": 12})
    return np, plt


@app.cell
def problem_setup_header(mo):
    mo.md(r"""
    ---
    ## 1. Problem Setup

    Linear regression predicts a continuous target $y \in \mathbb{R}$.
    Now suppose $y \in \{0, 1\}$ — a **binary label** (spam / not spam, disease / healthy, etc.).

    We still have i.i.d. data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $\mathbf{x}_i \in \mathbb{R}^d$,
    and we want to model:
    $$p(\mathbf{x}) = P(Y = 1 \mid X = \mathbf{x})$$

    Why not just use linear regression and threshold at 0.5? Two problems:
    - A linear model $\mathbf{x}^\top\boldsymbol{\beta}$ is **unbounded** — it can predict probabilities outside $[0, 1]$.
    - It has **constant sensitivity**: the model says moving $x$ by $\Delta$ always shifts the prediction by $b\Delta$,
      regardless of whether we're already at $p = 0.99$ or $p = 0.5$.

    We need a model that is bounded and has **diminishing sensitivity** near the extremes.
    """)
    return


@app.cell
def linear_prob_problem_viz(np, plt):
    def _():
        _rng = np.random.default_rng(0)
        _n = 60
        _x0 = _rng.normal(-1.5, 1, _n // 2)
        _x1 = _rng.normal(1.5, 1, _n // 2)
        _x = np.concatenate([_x0, _x1])
        _y = np.array([0] * (_n // 2) + [1] * (_n // 2))

        _xg = np.linspace(-5, 5, 200)
        _lin = 0.5 + 0.2 * _xg  # naive linear fit

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax.scatter(_x[_y == 0], _y[_y == 0], alpha=0.6, s=30, color="#e74c3c", label="$y=0$")
        ax.scatter(_x[_y == 1], _y[_y == 1], alpha=0.6, s=30, color="#3498db", label="$y=1$")
        ax.plot(_xg, _lin, "k-", lw=2, label="Linear fit")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axhline(1, color="gray", lw=0.5, ls="--")
        ax.fill_between(_xg, 1, _lin, where=_lin > 1, alpha=0.2, color="red", label="$p > 1$ impossible")
        ax.fill_between(_xg, 0, _lin, where=_lin < 0, alpha=0.2, color="red", label="$p < 0$ impossible")
        ax.set(xlabel="$x$", ylabel="$P(Y=1|x)$", title="Linear regression: predictions outside [0,1]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _sig = 1 / (1 + np.exp(-_xg))
        ax.scatter(_x[_y == 0], _y[_y == 0], alpha=0.6, s=30, color="#e74c3c", label="$y=0$")
        ax.scatter(_x[_y == 1], _y[_y == 1], alpha=0.6, s=30, color="#3498db", label="$y=1$")
        ax.plot(_xg, _sig, "#2ecc71", lw=2.5, label=r"$\sigma(\beta x)$")
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set(xlabel="$x$", ylabel="$P(Y=1|x)$", ylim=(-0.05, 1.05),
               title="Logistic regression: always in [0, 1]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def logistic_transform_header(mo):
    mo.md(r"""
    ---
    ## 2. The Logistic Transformation

    We want to map a linear score $z = \beta_0 + \mathbf{x}^\top\boldsymbol{\beta} \in \mathbb{R}$
    to a probability in $(0, 1)$.

    **Step 1 — Odds:** Instead of working with $p$ directly, work with the **odds** $\frac{p}{1-p} \in (0, \infty)$.
    The odds are unbounded above (good) but still bounded below at 0 (bad).

    **Step 2 — Log-odds (logit):** Taking the log gives the **logit**:
    $$\text{logit}(p) = \log\frac{p}{1-p} \in (-\infty, +\infty)$$

    Now it's unbounded in both directions. Set it equal to a linear function:
    $$\log\frac{p}{1-p} = \beta_0 + \mathbf{x}^\top\boldsymbol{\beta}$$

    **Step 3 — Solve for $p$:** Inverting the logit gives the **sigmoid** (logistic) function:
    $$\boxed{p(\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \mathbf{x}^\top\boldsymbol{\beta}}$$

    Key properties of $\sigma$:
    - $\sigma(0) = 0.5$ — uncertain at the boundary
    - $\sigma(z) \to 1$ as $z \to +\infty$, $\sigma(z) \to 0$ as $z \to -\infty$
    - Symmetric: $\sigma(-z) = 1 - \sigma(z)$
    - Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ — peaks at $z=0$, vanishes at extremes (diminishing sensitivity ✓)

    Use the slider below to see how $\beta$ controls the steepness and therefore the model's confidence.
    """)
    return


@app.cell
def sigmoid_viz(np, plt):
    def _():
        _z = np.linspace(-7, 7, 500)
        _sig = 1 / (1 + np.exp(-_z))
        _dsig = _sig * (1 - _sig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        ax = axes[0]
        ax.plot(_z, _sig, "#3498db", lw=2.5, label=r"$\sigma(z)$")
        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7, label="$p = 0.5$")
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
        ax.scatter([0], [0.5], s=80, c="#e74c3c", zorder=5)
        ax.set(xlabel="$z$", ylabel=r"$\sigma(z)$", title=r"Sigmoid: $\sigma(z) = \frac{1}{1+e^{-z}}$",
               ylim=(-0.05, 1.05))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(_z, _dsig, "#e74c3c", lw=2.5, label=r"$\sigma'(z) = \sigma(z)(1-\sigma(z))$")
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
        ax.scatter([0], [0.25], s=80, c="#e74c3c", zorder=5, label="Peak: $\\sigma'(0) = 0.25$")
        ax.set(xlabel="$z$", ylabel=r"$\sigma'(z)$", title="Derivative — diminishing sensitivity")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        _p = np.linspace(0.001, 0.999, 500)
        _logit = np.log(_p / (1 - _p))
        ax.plot(_logit, _p, "#2ecc71", lw=2.5)
        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set(xlabel=r"$\log\frac{p}{1-p}$ (logit)", ylabel="$p$",
               title="Logit → probability (inverse sigmoid)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def beta_slider(mo):
    beta_s = mo.ui.slider(
        start=-4.0, stop=4.0, value=1.5, step=0.1,
        label=r"Coefficient $\beta$",
    )
    beta_s
    return (beta_s,)


@app.cell
def sigmoid_slider_viz(beta_s, np, plt):
    def _():
        _b = beta_s.value
        _z = np.linspace(-6, 6, 400)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        for _bv, _alpha in [(-3.0, 0.25), (-1.0, 0.35), (0.5, 0.45), (3.0, 0.55)]:
            ax.plot(_z, 1 / (1 + np.exp(-_bv * _z)), lw=1.2, alpha=_alpha, color="#95a5a6",
                    label=f"$\\beta={_bv}$" if _bv in [-3.0, 3.0] else None)
        ax.plot(_z, 1 / (1 + np.exp(-_b * _z)), "#3498db", lw=3,
                label=f"Current: $\\beta = {_b:.1f}$")
        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.6)
        _steepness = "steep → confident" if abs(_b) > 2 else ("flat → uncertain" if abs(_b) < 0.5 else "moderate")
        ax.set(xlabel="$z = \\mathbf{x}^\\top\\boldsymbol{\\beta}$", ylabel=r"$\sigma(\beta z)$",
               title=f"$\\beta = {_b:.1f}$: {_steepness}", ylim=(-0.05, 1.05))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _z_pts = np.array([-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3], dtype=float)
        _p_pts = 1 / (1 + np.exp(-_b * _z_pts))
        _bar_colors = ["#e74c3c" if p < 0.5 else "#3498db" for p in _p_pts]
        ax.bar(_z_pts, _p_pts, width=0.35, color=_bar_colors, alpha=0.8, edgecolor="k", linewidth=0.5)
        ax.axhline(0.5, color="gray", ls="--", lw=1.5, label="Decision threshold")
        ax.set(xlabel="Score $z$", ylabel="$P(Y=1 \\mid z)$",
               title="Predicted probabilities (red=$\\hat{y}=0$, blue=$\\hat{y}=1$)",
               ylim=(0, 1.05))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def decision_boundary_header(mo):
    mo.md(r"""
    ---
    ## 3. Decision Boundaries

    To classify, we predict $\hat{Y} = 1$ when $p(\mathbf{x}) \geq 0.5$, i.e., when $\sigma(z) \geq 0.5$.
    Since $\sigma$ is monotone and $\sigma(0) = 0.5$, this is equivalent to:
    $$\hat{Y} = \mathbf{1}[\beta_0 + \mathbf{x}^\top\boldsymbol{\beta} \geq 0]$$

    The **decision boundary** is the hyperplane $\{\mathbf{x} : \beta_0 + \mathbf{x}^\top\boldsymbol{\beta} = 0\}$.
    Logistic regression is therefore a **linear classifier** — the boundary is always a hyperplane.

    **Signed distance from the boundary:** For any point $\mathbf{x}$:
    $$\text{dist}(\mathbf{x}, \text{boundary}) = \frac{\beta_0 + \mathbf{x}^\top\boldsymbol{\beta}}{\|\boldsymbol{\beta}\|}$$

    Points far from the boundary (large $|z|$) get predictions close to 0 or 1.
    Points near the boundary (small $|z|$) get predictions close to 0.5 — high uncertainty.

    **Effect of $\|\boldsymbol{\beta}\|$:** Larger coefficients make the sigmoid steeper around the boundary,
    giving more confident predictions at the same distance. Regularization controls this.

    Use the slider below to see how $\beta_0$ (the intercept) shifts the boundary.
    """)
    return


@app.cell
def decision_boundary_viz(np, plt):
    def _():
        _rng = np.random.default_rng(1)
        _n = 80
        _X0 = _rng.multivariate_normal([-1.5, -1.0], [[1, 0.3], [0.3, 1]], _n // 2)
        _X1 = _rng.multivariate_normal([1.5, 1.0], [[1, 0.3], [0.3, 1]], _n // 2)
        _X = np.vstack([_X0, _X1])
        _y = np.array([0] * (_n // 2) + [1] * (_n // 2))

        # True (generative) parameters
        _b0, _b1, _b2 = -0.0, 1.0, 0.7
        _norm = np.sqrt(_b1**2 + _b2**2)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, _scale, _title in zip(
            axes, [1.0, 3.0],
            ["Shallow boundary ($\\|\\boldsymbol{\\beta}\\|$ small)",
             "Sharp boundary ($\\|\\boldsymbol{\\beta}\\|$ large)"]
        ):
            _b = _scale * np.array([_b1, _b2])
            _b0s = _scale * _b0
            _norm_s = np.linalg.norm(_b)

            _xg = np.linspace(-4, 4, 200)
            _yg = np.linspace(-4, 4, 200)
            _Xg, _Yg = np.meshgrid(_xg, _yg)
            _Z = _b0s + _b[0] * _Xg + _b[1] * _Yg
            _P = 1 / (1 + np.exp(-_Z))

            ax.contourf(_Xg, _Yg, _P, levels=20, cmap="RdBu_r", alpha=0.5, vmin=0, vmax=1)
            ax.contour(_Xg, _Yg, _Z, levels=[0], colors="k", linewidths=2)
            ax.scatter(_X[_y == 0, 0], _X[_y == 0, 1], s=30, c="#e74c3c", alpha=0.7, label="$y=0$", zorder=3)
            ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], s=30, c="#3498db", alpha=0.7, label="$y=1$", zorder=3)
            ax.set(xlabel="$x_1$", ylabel="$x_2$", title=_title, xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
            ax.legend(fontsize=9)

        plt.suptitle("Same boundary, different $\\|\\boldsymbol{\\beta}\\|$ — color = predicted probability",
                     fontsize=11, y=1.01)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def bias_slider(mo):
    bias_s = mo.ui.slider(
        start=-3.0, stop=3.0, value=0.0, step=0.1,
        label=r"Intercept $\beta_0$",
    )
    bias_s
    return (bias_s,)


@app.cell
def boundary_slider_viz(bias_s, np, plt):
    def _():
        _rng = np.random.default_rng(1)
        _n = 80
        _X0 = _rng.multivariate_normal([-1.5, -1.0], [[1, 0.3], [0.3, 1]], _n // 2)
        _X1 = _rng.multivariate_normal([1.5, 1.0], [[1, 0.3], [0.3, 1]], _n // 2)
        _Xd = np.vstack([_X0, _X1])
        _yd = np.array([0] * (_n // 2) + [1] * (_n // 2))

        _b0 = bias_s.value
        _b1, _b2 = 1.0, 0.7

        _xg = np.linspace(-4.5, 4.5, 250)
        _yg = np.linspace(-4.5, 4.5, 250)
        _Xg, _Yg = np.meshgrid(_xg, _yg)
        _Z = _b0 + _b1 * _Xg + _b2 * _Yg
        _P = 1 / (1 + np.exp(-_Z))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.contourf(_Xg, _Yg, _P, levels=20, cmap="RdBu_r", alpha=0.45, vmin=0, vmax=1)
        ax.contour(_Xg, _Yg, _Z, levels=[0], colors="k", linewidths=2.5)

        # Annotate decision boundary equation
        _x1_line = np.linspace(-4.5, 4.5, 200)
        _x2_line = -(_b0 + _b1 * _x1_line) / _b2
        _mask = (_x2_line >= -4.5) & (_x2_line <= 4.5)
        ax.plot(_x1_line[_mask], _x2_line[_mask], "k-", lw=2.5,
                label=f"$\\beta_0 + x_1 + 0.7x_2 = 0$ (β₀={_b0:.1f})")

        ax.scatter(_Xd[_yd == 0, 0], _Xd[_yd == 0, 1], s=30, c="#e74c3c", alpha=0.7, label="$y=0$", zorder=3)
        ax.scatter(_Xd[_yd == 1, 0], _Xd[_yd == 1, 1], s=30, c="#3498db", alpha=0.7, label="$y=1$", zorder=3)

        _direction = "left/up" if _b0 < -0.1 else ("right/down" if _b0 > 0.1 else "centered")
        ax.set(xlabel="$x_1$", ylabel="$x_2$", xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), aspect="equal",
               title=f"$\\beta_0 = {_b0:.1f}$: boundary shifted {_direction}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def loss_header(mo):
    mo.md(r"""
    ---
    ## 4. Choosing the Loss

    The **0-1 loss** is the most natural:
    $$\ell_{0\text{-}1}(\hat{y}, y) = \mathbf{1}[y \neq \mathbf{1}[\hat{y} \geq 0]]$$

    It counts mistakes. But it is **neither convex nor differentiable** — minimizing it is NP-hard.
    We need a **convex surrogate**.

    The **logistic loss** (cross-entropy) is:
    $$\ell_{\log}(\hat{y}, y) = y\log(1 + e^{-\hat{y}}) + (1-y)\log(1 + e^{\hat{y}})$$

    where $\hat{y} = \mathbf{x}^\top\boldsymbol{\beta}$ is the raw linear score (not yet passed through sigmoid).

    Rewriting for each class:
    - $y = 1$: $\ell = \log(1 + e^{-\hat{y}})$ — penalizes negative scores (predicting 0 when truth is 1)
    - $y = 0$: $\ell = \log(1 + e^{\hat{y}})$ — penalizes positive scores (predicting 1 when truth is 0)

    In both cases, the loss is **large when the prediction is confidently wrong** and **near 0 when correctly confident**.
    It is strictly convex in $\hat{y}$, so gradient descent converges to a unique minimum.

    The hinge loss (SVM) is another convex surrogate but lacks a probabilistic interpretation.
    """)
    return


@app.cell
def loss_comparison_viz(np, plt):
    def _():
        _yhat = np.linspace(-4, 4, 500)

        # y=1 case: score should be positive
        _loss_01 = (_yhat < 0).astype(float)
        _loss_log = np.log(1 + np.exp(-_yhat))
        _loss_hinge = np.maximum(0, 1 - _yhat)
        _loss_sq = (1 - _yhat) ** 2 / 4  # scaled for readability

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.plot(_yhat, _loss_01, "k-", lw=2, label="0-1 loss (non-convex, NP-hard)", zorder=3)
        ax.plot(_yhat, _loss_log, "#e74c3c", lw=2.5, label=r"Logistic loss $\log(1+e^{-\hat{y}})$")
        ax.plot(_yhat, _loss_hinge, "#3498db", lw=2, ls="--", label=r"Hinge loss $\max(0, 1-\hat{y})$")
        ax.plot(_yhat, _loss_sq, "#2ecc71", lw=2, ls=":", label=r"Squared loss (scaled)")
        ax.set(xlabel=r"Score $\hat{y} = \mathbf{x}^\top\boldsymbol{\beta}$", ylabel="Loss",
               title="Surrogate losses for $y=1$ — score should be $> 0$",
               ylim=(-0.1, 3.5), xlim=(-4, 4))
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.6)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _z = np.linspace(-5, 5, 500)
        _sig = 1 / (1 + np.exp(-_z))
        ax.plot(_z, _loss_log[::-1], "#e74c3c", lw=2.5, label="$y=1$: $\\log(1+e^{-z})$")
        ax.plot(_z, np.log(1 + np.exp(_z))[::-1], "#3498db", lw=2.5, ls="--", label="$y=0$: $\\log(1+e^{z})$")
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.6, label="Decision boundary $z=0$")
        ax.set(xlabel=r"Score $z = \mathbf{x}^\top\boldsymbol{\beta}$", ylabel="Logistic loss",
               title="Logistic loss is large when confidently wrong",
               xlim=(-4, 4), ylim=(-0.1, 4.5))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def mle_header(mo):
    mo.md(r"""
    ---
    ## 5. Maximum Likelihood

    The logistic loss has a clean probabilistic origin: it is the **negative log-likelihood** under a Bernoulli model.

    **Model:** Given $\mathbf{x}$, the label $Y$ is Bernoulli with success probability $\sigma(\boldsymbol{\theta}^\top\mathbf{x})$:
    $$P(Y = y \mid \mathbf{x}) = \sigma(\boldsymbol{\theta}^\top\mathbf{x})^y \cdot [1 - \sigma(\boldsymbol{\theta}^\top\mathbf{x})]^{1-y}$$

    **Log-likelihood** over $n$ i.i.d. observations:
    $$LL(\boldsymbol{\theta}) = \sum_{i=1}^n \bigl[y_i \log \sigma(\boldsymbol{\theta}^\top\mathbf{x}_i) + (1 - y_i)\log(1 - \sigma(\boldsymbol{\theta}^\top\mathbf{x}_i))\bigr]$$

    **Equivalence:** Using $-\log\sigma(z) = \log(1 + e^{-z})$ and $-\log(1-\sigma(z)) = \log(1 + e^z)$:
    $$-LL(\boldsymbol{\theta}) = \sum_{i=1}^n \ell_{\log}(\boldsymbol{\theta}^\top\mathbf{x}_i, y_i)$$

    So **minimizing the empirical logistic loss = maximizing the Bernoulli log-likelihood**. 
    The two optimization problems are identical — the loss function *is* the negative log-likelihood.

    The visualization below shows how the log-likelihood changes as we sweep $\theta$ on a 1D example.
    The MLE is the $\theta$ where the observed data was most likely under the model.
    """)
    return


@app.cell
def mle_viz(np, plt):
    def _():
        _rng = np.random.default_rng(5)
        _n = 30
        _x = _rng.uniform(-3, 3, _n)
        _true_theta = 1.5
        _y = (_rng.uniform(size=_n) < 1 / (1 + np.exp(-_true_theta * _x))).astype(float)

        _thetas = np.linspace(-3, 5, 300)

        def _ll(theta):
            _z = theta * _x
            _p = 1 / (1 + np.exp(-_z))
            _p = np.clip(_p, 1e-10, 1 - 1e-10)
            return np.sum(_y * np.log(_p) + (1 - _y) * np.log(1 - _p))

        _ll_vals = np.array([_ll(t) for t in _thetas])
        _mle_idx = np.argmax(_ll_vals)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.plot(_thetas, _ll_vals, "#3498db", lw=2.5)
        ax.axvline(_thetas[_mle_idx], color="#e74c3c", ls="--", lw=2,
                   label=f"MLE $\\hat{{\\theta}} = {_thetas[_mle_idx]:.2f}$")
        ax.axvline(_true_theta, color="#2ecc71", ls=":", lw=2,
                   label=f"True $\\theta = {_true_theta}$")
        ax.scatter([_thetas[_mle_idx]], [_ll_vals[_mle_idx]], s=100, c="#e74c3c", zorder=5)
        ax.set(xlabel=r"$\theta$", ylabel="Log-likelihood $LL(\theta)$",
               title="Log-likelihood is concave → unique maximum")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _xg = np.linspace(-3.5, 3.5, 200)
        _sig_mle = 1 / (1 + np.exp(-_thetas[_mle_idx] * _xg))
        _sig_true = 1 / (1 + np.exp(-_true_theta * _xg))
        ax.scatter(_x[_y == 0], _y[_y == 0], alpha=0.7, s=40, c="#e74c3c", label="$y=0$", zorder=3)
        ax.scatter(_x[_y == 1], _y[_y == 1], alpha=0.7, s=40, c="#3498db", label="$y=1$", zorder=3)
        ax.plot(_xg, _sig_mle, "#e74c3c", lw=2.5, label=f"MLE fit ($\\hat{{\\theta}}={_thetas[_mle_idx]:.2f}$)")
        ax.plot(_xg, _sig_true, "#2ecc71", lw=2, ls="--", label=f"True ($\\theta={_true_theta}$)")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set(xlabel="$x$", ylabel="$P(Y=1|x)$", ylim=(-0.05, 1.05),
               title="MLE fit vs true sigmoid")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def gradient_header(mo):
    mo.md(r"""
    ---
    ## 6. Gradient & Optimization

    **No closed form.** Unlike OLS, setting $\nabla \hat{\mathcal{R}}_n = \mathbf{0}$ gives a nonlinear system
    with no analytical solution. We must use iterative methods.

    **The gradient is elegantly simple.** For a single point $(y, \mathbf{x})$:
    $$\frac{\partial \ell_{\log}}{\partial \theta_j} = \bigl[\sigma(\boldsymbol{\theta}^\top\mathbf{x}) - y\bigr] x_j$$

    The prediction error $(\hat{p} - y)$ times the feature $x_j$ — identical in form to the linear regression gradient.
    Over all $n$ observations:
    $$\nabla_{\boldsymbol{\theta}} \hat{\mathcal{R}}_n = \frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{p}} - \mathbf{y}), \quad \hat{\mathbf{p}}_i = \sigma(\mathbf{x}_i^\top\boldsymbol{\theta})$$

    **Convexity guarantees a unique global minimum.** The Hessian of the loss is:
    $$H = \frac{1}{n}\mathbf{X}^\top \mathbf{W} \mathbf{X}, \quad \mathbf{W} = \text{diag}(\hat{p}_i(1-\hat{p}_i))$$

    $\mathbf{W}$ is positive definite (weights are all in $(0,1)$), so $H \succ 0$ — the loss is strictly convex
    and **gradient descent converges to the unique global minimum**.

    **Gradient descent update:**
    $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{1}{n}\mathbf{X}^\top(\sigma(\mathbf{X}\boldsymbol{\theta}_t) - \mathbf{y})$$
    """)
    return


@app.cell
def gradient_descent_viz(np, plt):
    def _():
        _rng = np.random.default_rng(3)
        _n = 50
        _x1 = _rng.normal(0, 1, _n)
        _x2 = _rng.normal(0, 1, _n)
        _X = np.column_stack([np.ones(_n), _x1, _x2])
        _true = np.array([0.0, 2.0, -1.5])
        _z_true = _X @ _true
        _y = (_rng.uniform(size=_n) < 1 / (1 + np.exp(-_z_true))).astype(float)

        def _loss(theta):
            _z = _X @ theta
            _p = np.clip(1 / (1 + np.exp(-_z)), 1e-10, 1 - 1e-10)
            return -np.mean(_y * np.log(_p) + (1 - _y) * np.log(1 - _p))

        def _grad(theta):
            _z = _X @ theta
            _p = 1 / (1 + np.exp(-_z))
            return _X.T @ (_p - _y) / _n

        _eta = 0.5
        _theta = np.zeros(3)
        _losses, _thetas = [], []
        for _t in range(200):
            _losses.append(_loss(_theta))
            _thetas.append(_theta.copy())
            _theta -= _eta * _grad(_theta)

        _thetas = np.array(_thetas)
        _losses_arr = np.array(_losses)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        ax.plot(_losses_arr, "#3498db", lw=2)
        ax.axhline(min(_losses_arr), color="red", ls="--", lw=1, alpha=0.7,
                   label=f"Min loss = {min(_losses_arr):.3f}")
        ax.set(xlabel="Iteration", ylabel="Negative log-likelihood", title="Gradient descent convergence")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for _j, (_name, _color) in enumerate(
            zip([r"$\theta_0$ (intercept)", r"$\theta_1$", r"$\theta_2$"],
                ["#95a5a6", "#e74c3c", "#3498db"])
        ):
            ax.plot(_thetas[:, _j], color=_color, lw=2, label=_name)
            ax.axhline(_true[_j], color=_color, ls=":", lw=1.5, alpha=0.6)
        ax.set(xlabel="Iteration", ylabel="Parameter value", title="Parameter convergence\n(dotted = true values)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        _final_theta = _thetas[-1]
        _probs = 1 / (1 + np.exp(-_X @ _final_theta))
        _sort_idx = np.argsort(_probs)
        _colors = ["#e74c3c" if yi == 0 else "#3498db" for yi in _y[_sort_idx]]
        ax.scatter(range(_n), _probs[_sort_idx], c=_colors, s=30, zorder=3)
        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set(xlabel="Sample (sorted by predicted prob)", ylabel="$\\hat{p}$",
               title="Final predicted probabilities\n(red=$y=0$, blue=$y=1$)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def convexity_viz(np, plt):
    def _():
        _rng = np.random.default_rng(7)
        _n = 40
        _x = _rng.standard_normal((_n, 2))
        _y = (_rng.standard_normal(_n) > 0).astype(float)

        def _loss(t1, t2):
            _z = t1 * _x[:, 0] + t2 * _x[:, 1]
            _p = np.clip(1 / (1 + np.exp(-_z)), 1e-10, 1 - 1e-10)
            return -np.mean(_y * np.log(_p) + (1 - _y) * np.log(1 - _p))

        _t1 = np.linspace(-3, 3, 80)
        _t2 = np.linspace(-3, 3, 80)
        _T1, _T2 = np.meshgrid(_t1, _t2)
        _L = np.vectorize(_loss)(_T1, _T2)

        fig, ax = plt.subplots(figsize=(7, 6))
        _cf = ax.contourf(_T1, _T2, _L, levels=25, cmap="Blues")
        ax.contour(_T1, _T2, _L, levels=25, colors="white", alpha=0.3, linewidths=0.5)
        plt.colorbar(_cf, ax=ax, label="Loss")
        _min_idx = np.unravel_index(np.argmin(_L), _L.shape)
        ax.scatter(_T1[_min_idx], _T2[_min_idx], s=150, c="#e74c3c", marker="*", zorder=5,
                   label=f"Minimum ({_T1[_min_idx]:.2f}, {_T2[_min_idx]:.2f})")
        ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$",
               title="Logistic loss surface — convex, single global minimum")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def regularization_header(mo):
    mo.md(r"""
    ---
    ## 7. Regularization

    Logistic regression overfits when $d$ is large relative to $n$, or when features are highly correlated.
    The symptoms are the same as in linear regression: large coefficients, high variance.

    There is an additional pathology unique to classification: **perfect separability**.
    If the data is linearly separable, the MLE does not exist — the log-likelihood increases without bound
    as $\|\boldsymbol{\theta}\| \to \infty$ (the sigmoid gets infinitely steep, certainty increases forever).
    Regularization prevents this by penalizing large coefficients.

    **L2 (Ridge logistic regression):**
    $$\hat{\boldsymbol{\theta}}_{\text{L2}} = \arg\min_{\boldsymbol{\theta}} \left\{ \frac{1}{n}\sum_{i=1}^n \ell_{\log}(\boldsymbol{\theta}^\top\mathbf{x}_i, y_i) + \lambda\|\boldsymbol{\theta}\|_2^2 \right\}$$

    Shrinks all coefficients smoothly toward zero. Still no closed form (gradient still nonlinear),
    but makes the Hessian strictly positive definite even under separability.

    **L1 (Lasso logistic regression):**
    $$\hat{\boldsymbol{\theta}}_{\text{L1}} = \arg\min_{\boldsymbol{\theta}} \left\{ \frac{1}{n}\sum_{i=1}^n \ell_{\log}(\boldsymbol{\theta}^\top\mathbf{x}_i, y_i) + \lambda\|\boldsymbol{\theta}\|_1 \right\}$$

    Same geometric argument as in linear Lasso — the $L_1$ diamond has corners on the axes,
    so solutions are sparse: irrelevant features get exactly zeroed out.

    Note: sklearn's `C = 1/λ` convention — **larger C = less regularization**.

    Use the slider below to see how $\lambda$ affects the decision boundary and coefficient norm.
    """)
    return


@app.cell
def separability_viz(np, plt):
    def _():
        _rng = np.random.default_rng(42)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, _gap, _title in zip(
            axes, [0.2, 2.5],
            ["Non-separable: MLE exists, finite $\\|\\boldsymbol{\\theta}\\|$",
             "Perfectly separable: MLE → $\\infty$, need regularization"]
        ):
            _n = 50
            _x0 = _rng.normal(-_gap, 0.8, _n // 2)
            _x1 = _rng.normal(_gap, 0.8, _n // 2)
            _x = np.concatenate([_x0, _x1])
            _y = np.array([0] * (_n // 2) + [1] * (_n // 2))

            # Fit logistic regression via gradient descent with/without reg
            _X = np.column_stack([np.ones(_n), _x])

            def _fit(lam=0.0, n_iter=2000, lr=0.1):
                theta = np.zeros(2)
                for _ in range(n_iter):
                    p = np.clip(1 / (1 + np.exp(-_X @ theta)), 1e-10, 1 - 1e-10)
                    g = _X.T @ (p - _y) / _n + 2 * lam * theta
                    theta -= lr * g
                return theta

            _theta_unreg = _fit(lam=0.0)
            _theta_reg = _fit(lam=0.1)

            _xg = np.linspace(-5, 5, 300)
            _Xg = np.column_stack([np.ones(300), _xg])
            _p_unreg = 1 / (1 + np.exp(-_Xg @ _theta_unreg))
            _p_reg = 1 / (1 + np.exp(-_Xg @ _theta_reg))

            ax.scatter(_x[_y == 0], np.zeros(_n // 2) + 0.02 * _rng.standard_normal(_n // 2),
                       alpha=0.7, s=40, c="#e74c3c", label="$y=0$", zorder=3)
            ax.scatter(_x[_y == 1], np.ones(_n // 2) + 0.02 * _rng.standard_normal(_n // 2),
                       alpha=0.7, s=40, c="#3498db", label="$y=1$", zorder=3)
            ax.plot(_xg, _p_unreg, "#e74c3c", lw=2.5, label=f"No reg ($\\|\\theta\\|={np.linalg.norm(_theta_unreg):.1f}$)")
            ax.plot(_xg, _p_reg, "#3498db", lw=2.5, ls="--", label=f"L2 reg ($\\|\\theta\\|={np.linalg.norm(_theta_reg):.1f}$)")
            ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
            ax.set(xlabel="$x$", ylabel="$P(Y=1|x)$", title=_title, ylim=(-0.15, 1.15), xlim=(-5, 5))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def lam_log_slider(mo):
    lam_s = mo.ui.slider(
        start=-3.0, stop=2.0, value=0.0, step=0.1,
        label=r"$\log_{10}(\lambda)$",
    )
    lam_s
    return (lam_s,)


@app.cell
def reg_boundary_viz(lam_s, np, plt):
    def _():
        _rng = np.random.default_rng(42)
        _n = 100
        _X0 = _rng.multivariate_normal([-1.2, 0.5], [[0.8, 0], [0, 0.8]], _n // 2)
        _X1 = _rng.multivariate_normal([1.2, -0.5], [[0.8, 0], [0, 0.8]], _n // 2)
        _Xr = np.vstack([_X0, _X1])
        _yr = np.array([0] * (_n // 2) + [1] * (_n // 2))
        _Xf = np.column_stack([np.ones(_n), _Xr])

        _lam = 10 ** lam_s.value

        def _fit_l2(lam, n_iter=600, lr=0.08):
            theta = np.zeros(3)
            for _ in range(n_iter):
                z = _Xf @ theta
                p = np.clip(1 / (1 + np.exp(-z)), 1e-10, 1 - 1e-10)
                g = _Xf.T @ (p - _yr) / _n
                g[1:] += 2 * lam * theta[1:]
                theta -= lr * g
            return theta

        _theta = _fit_l2(_lam)
        _coef_norm = np.linalg.norm(_theta[1:])

        _xg = np.linspace(-4, 4, 200)
        _yg = np.linspace(-4, 4, 200)
        _Xg, _Yg = np.meshgrid(_xg, _yg)
        _Xg_flat = np.column_stack([np.ones(200 * 200), _Xg.ravel(), _Yg.ravel()])
        _P = (1 / (1 + np.exp(-_Xg_flat @ _theta))).reshape(200, 200)

        # Precompute norm path
        _lam_grid = np.logspace(-3, 2, 60)
        _norms = []
        _accs = []
        for _l in _lam_grid:
            _th = _fit_l2(_l)
            _norms.append(np.linalg.norm(_th[1:]))
            _accs.append(np.mean((_Xf @ _th >= 0).astype(float) == _yr))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.contourf(_Xg, _Yg, _P, levels=20, cmap="RdBu_r", alpha=0.45, vmin=0, vmax=1)
        ax.contour(_Xg, _Yg, _P, levels=[0.5], colors="k", linewidths=2.5)
        ax.scatter(_Xr[_yr == 0, 0], _Xr[_yr == 0, 1], s=30, c="#e74c3c", alpha=0.8, label="$y=0$", zorder=3)
        ax.scatter(_Xr[_yr == 1, 0], _Xr[_yr == 1, 1], s=30, c="#3498db", alpha=0.8, label="$y=1$", zorder=3)
        ax.set(xlabel="$x_1$", ylabel="$x_2$", xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
               title=f"$\\lambda = {_lam:.3f}$,  $\\|\\hat{{\\boldsymbol{{\\theta}}}}\\| = {_coef_norm:.2f}$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        ax2r = axes[1].twinx()
        axes[1].plot(_lam_grid, _norms, "#e74c3c", lw=2, label=r"$\|\hat{\theta}\|$ (left)")
        ax2r.plot(_lam_grid, _accs, "#3498db", lw=2, ls="--", label="Train accuracy (right)")
        axes[1].axvline(_lam, color="gray", ls="--", lw=2, alpha=0.8, label=f"Current λ = {_lam:.3f}")
        axes[1].set(xscale="log", xlabel=r"$\lambda$", ylabel=r"$\|\hat{\theta}\|$",
                    title="Regularization path: shrinkage vs accuracy")
        axes[1].legend(loc="upper left", fontsize=8)
        ax2r.legend(loc="upper right", fontsize=8)
        ax2r.set_ylabel("Train accuracy")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def reg_path_viz(np, plt):
    def _():
        _rng = np.random.default_rng(9)
        _n, _d = 100, 8
        _X = _rng.standard_normal((_n, _d))
        _true = np.array([2.0, -1.5, 0.8, 0.0, 0.0, -0.3, 1.2, 0.0])
        _z_true = _X @ _true
        _y = (_rng.uniform(size=_n) < 1 / (1 + np.exp(-_z_true))).astype(float)
        _X_std = (_X - _X.mean(0)) / _X.std(0)

        def _fit_reg(lam, penalty="l2", n_iter=2000, lr=0.05):
            theta = np.zeros(_d)
            for _ in range(n_iter):
                _z = _X_std @ theta
                p = np.clip(1 / (1 + np.exp(-_z)), 1e-10, 1 - 1e-10)
                grad = _X_std.T @ (p - _y) / _n
                if penalty == "l2":
                    grad += 2 * lam * theta
                else:  # l1 subgradient
                    grad += lam * np.sign(theta)
                theta -= lr * grad
                if penalty == "l1":
                    theta = np.sign(theta) * np.maximum(np.abs(theta) - lr * lam, 0)
            return theta

        _lambdas = np.logspace(-3, 1, 50)
        _coefs_l2 = np.array([_fit_reg(l, "l2") for l in _lambdas])
        _coefs_l1 = np.array([_fit_reg(l, "l1") for l in _lambdas])

        _colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, _coefs, _title, _pen in zip(
            axes,
            [_coefs_l2, _coefs_l1],
            ["Ridge (L2): all shrink, none exactly zero", "Lasso (L1): sparse — some hit zero exactly"],
            ["L2", "L1"],
        ):
            for _j in range(_d):
                _label = f"$\\theta_{_j+1}$" + (" (true=0)" if _true[_j] == 0 else "")
                ax.plot(_lambdas, _coefs[:, _j], lw=2, color=_colors[_j], label=_label)
            ax.set(xscale="log", xlabel=r"$\lambda$", ylabel="Coefficient",
                   title=_title)
            ax.axhline(0, color="k", lw=0.5, alpha=0.5)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Regularization path: as $\\lambda$ increases, coefficients shrink toward zero",
                     fontsize=11, y=1.01)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def summary_header(mo):
    mo.md(r"""
    ---
    ## Summary

    | Concept | Formula | Intuition |
    |---------|---------|-----------|
    | **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ | Squashes $\mathbb{R}$ into $(0,1)$, S-shaped |
    | **Logit** | $\log\frac{p}{1-p} = \mathbf{x}^\top\boldsymbol{\beta}$ | Log-odds is linear in features |
    | **Decision boundary** | $\mathbf{x}^\top\boldsymbol{\beta} = 0$ | A hyperplane — logistic is a linear classifier |
    | **Logistic loss** | $y\log(1+e^{-\hat{y}}) + (1-y)\log(1+e^{\hat{y}})$ | Convex surrogate for 0-1 loss |
    | **MLE equivalence** | $\min \hat{\mathcal{R}}_n = \max LL$ | Loss minimization = likelihood maximization |
    | **Gradient** | $\frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{p}} - \mathbf{y})$ | Prediction error × feature — same form as OLS |
    | **Convexity** | $H = \frac{1}{n}\mathbf{X}^\top\mathbf{W}\mathbf{X} \succ 0$ | Unique global minimum, GD converges |
    | **L2 reg** | $+\lambda\|\boldsymbol{\theta}\|_2^2$ | Shrinks all, prevents separability divergence |
    | **L1 reg** | $+\lambda\|\boldsymbol{\theta}\|_1$ | Shrinks and zeros — feature selection |

    **Connection to linear regression:**
    Both are empirical risk minimization with a linear model $z = \mathbf{x}^\top\boldsymbol{\theta}$.
    The difference is entirely in the loss: squared loss for regression, logistic loss for classification.
    The gradient in both cases has the form **(prediction − truth) × feature**.
    Regularization (Ridge/Lasso) carries over identically.
    """)
    return


if __name__ == "__main__":
    app.run()
