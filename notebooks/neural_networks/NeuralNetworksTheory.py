"""
Neural Networks & Backpropagation — companion notebook to neural-networks.tex

Covers the same material with code and plots instead of proofs.
"""

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def title():
    import marimo as mo

    mo.md(r"""
    # Neural Networks & Backpropagation

    Companion to `notes/detailed-notes/neural-networks.tex`.
    Same material, code and plots instead of proofs.

    1. The Perceptron — linear separation and margin
    2. Need for Nonlinearity — activation functions
    3. Universal Approximation — what one hidden layer can do
    4. Forward Pass — computing predictions layer by layer
    5. Loss Functions — what we're optimizing
    6. Backpropagation — why it's $O(E)$ not $O(WE)$
    7. Vanishing & Exploding Gradients — depth's dark side
    8. SGD and Convergence — from full-batch to stochastic
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
def perceptron_header(mo):
    mo.md(r"""
    ---
    ## 1. The Perceptron

    The Perceptron is the simplest model for binary classification.
    Given input $\mathbf{x} \in \mathbb{R}^d$ and labels $y \in \{-1, +1\}$, it predicts:
    $$\hat{y} = \text{sign}(\langle \mathbf{w}, \mathbf{x} \rangle)$$

    The weight vector $\mathbf{w}$ is **orthogonal to the decision hyperplane** $H = \{\mathbf{x} : \langle \mathbf{w}, \mathbf{x} \rangle = 0\}$.
    The **signed distance** from any point $\mathbf{x}$ to $H$ is:
    $$\text{dist}(\mathbf{x}, H) = \frac{\langle \mathbf{w}, \mathbf{x} \rangle}{\|\mathbf{w}\|}$$

    On a mistake, the perceptron **tilts the hyperplane** toward the misclassified point:
    $$\mathbf{w} \leftarrow \mathbf{w} + y\mathbf{x}$$

    This update strictly improves the margin:
    $y\langle \tilde{\mathbf{w}}, \mathbf{x} \rangle = y\langle \mathbf{w}, \mathbf{x} \rangle + \|\mathbf{x}\|^2 > y\langle \mathbf{w}, \mathbf{x} \rangle$

    **Convergence theorem:** If the data has margin $\gamma = \min_i |\langle \mathbf{w}^*, \mathbf{x}_i \rangle|$
    and $\|\mathbf{x}\| \leq 1$, the perceptron makes at most:
    $$T \leq \frac{1}{\gamma^2} \text{ mistakes}$$

    Use the slider below — watch how the margin controls the mistake bound.
    """)
    return


@app.cell
def margin_slider(mo):
    margin_s = mo.ui.slider(
        start=0.05, stop=0.95, value=0.3, step=0.05,
        label=r"Margin $\gamma$",
    )
    margin_s
    return (margin_s,)


@app.cell
def perceptron_viz(margin_s, np, plt):
    def _():
        _rng = np.random.default_rng(42)
        _gamma = margin_s.value
        _n = 60

        # Generate linearly separable data with given margin
        _w_star = np.array([1.0, 0.0])
        _X = _rng.uniform(-1, 1, (_n, 2))
        _X = _X / np.linalg.norm(_X, axis=1, keepdims=True)
        _margin_vals = _X @ _w_star
        _mask = np.abs(_margin_vals) >= _gamma
        _X = _X[_mask]
        _y = np.sign(_X @ _w_star)

        # Simulate perceptron
        _w = np.zeros(2)
        _mistakes = 0
        for _xi, _yi in zip(_X, _y):
            if _yi * np.dot(_w, _xi) <= 0:
                _w += _yi * _xi
                _mistakes += 1

        _bound = int(np.ceil(1 / _gamma**2))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        _c0 = _y == -1
        _c1 = _y == 1
        ax.scatter(_X[_c0, 0], _X[_c0, 1], s=40, c="#e74c3c", alpha=0.7, label="$y=-1$", zorder=3)
        ax.scatter(_X[_c1, 0], _X[_c1, 1], s=40, c="#3498db", alpha=0.7, label="$y=+1$", zorder=3)
        _t = np.linspace(-1.2, 1.2, 100)
        if abs(_w[1]) > 1e-8:
            ax.plot(_t, -_w[0] / _w[1] * _t, "k-", lw=2, label="Perceptron boundary")
        ax.axvline(0, color="#2ecc71", ls="--", lw=2, label=f"True boundary ($w^*$)")
        ax.fill_betweenx([-1.2, 1.2], -_gamma, _gamma, alpha=0.1, color="#2ecc71", label=f"Margin $\\gamma={_gamma:.2f}$")
        ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal",
               xlabel="$x_1$", ylabel="$x_2$",
               title=f"Mistakes: {_mistakes} (bound: $1/\\gamma^2 = {_bound}$)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _gammas = np.linspace(0.05, 0.95, 200)
        _bounds = 1 / _gammas**2
        ax.plot(_gammas, _bounds, "#3498db", lw=2.5, label=r"$T \leq 1/\gamma^2$")
        ax.axvline(_gamma, color="#e74c3c", ls="--", lw=2,
                   label=f"$\\gamma = {_gamma:.2f}$ → bound = {_bound}")
        ax.scatter([_gamma], [_bound], s=100, c="#e74c3c", zorder=5)
        ax.set(xlabel=r"Margin $\gamma$", ylabel="Max mistakes $T$",
               title="Perceptron mistake bound — tiny margin = exponential mistakes",
               xlim=(0.05, 0.95), ylim=(0, 420))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def nonlinearity_header(mo):
    mo.md(r"""
    ---
    ## 2. Need for Nonlinearity

    The perceptron is limited to linear decision boundaries. But more fundamentally,
    **stacking linear layers doesn't help** — the composition of affine maps is affine:
    $$g(f(\mathbf{x})) = C(A\mathbf{x} + b) + d = \underbrace{(CA)}_{\text{one matrix}}\mathbf{x} + \underbrace{(Cb+d)}_{\text{one bias}}$$

    A 100-layer linear network is exactly equivalent to a single linear layer.

    **Solution:** Apply a nonlinear **activation function** $\sigma: \mathbb{R} \to \mathbb{R}$ element-wise after each layer:
    $$\mathbf{h}^{(l)} = \sigma\bigl(W^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\bigr)$$

    Common choices and their properties:

    | Activation | Formula | Derivative | Vanishes? |
    |-----------|---------|-----------|-----------|
    | Sigmoid | $\frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z)) \leq \frac{1}{4}$ | Yes — saturates |
    | Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z) \leq 1$ | Yes — saturates |
    | ReLU | $\max(0,z)$ | $\mathbf{1}_{z>0}$ | No dead neurons only |
    | Leaky ReLU | $\max(\alpha z, z)$ | $\alpha$ or $1$ | Never fully dead |
    """)
    return


@app.cell
def activation_viz(np, plt):
    def _():
        _z = np.linspace(-4, 4, 500)
        _sig = 1 / (1 + np.exp(-_z))
        _tanh = np.tanh(_z)
        _relu = np.maximum(0, _z)
        _lrelu = np.where(_z > 0, _z, 0.1 * _z)

        _dsig = _sig * (1 - _sig)
        _dtanh = 1 - _tanh**2
        _drelu = (_z > 0).astype(float)
        _dlrelu = np.where(_z > 0, 1.0, 0.1)

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        _activations = [
            (_sig, _dsig, "#3498db", "Sigmoid", r"$\sigma(z) = \frac{1}{1+e^{-z}}$", r"$\sigma'(z) \leq 0.25$"),
            (_tanh, _dtanh, "#e74c3c", "Tanh", r"$\tanh(z)$", r"$\tanh'(z) \leq 1$"),
            (_relu, _drelu, "#2ecc71", "ReLU", r"$\max(0,z)$", r"$\{0, 1\}$ — no shrinking"),
            (_lrelu, _dlrelu, "#f39c12", "Leaky ReLU", r"$\max(0.1z, z)$", r"$\{0.1, 1\}$ — never dead"),
        ]

        for ax, (act, dact, color, name, formula, deriv_note) in zip(axes.flat, _activations):
            ax2 = ax.twinx()
            ax.plot(_z, act, color=color, lw=2.5, label=formula)
            ax2.plot(_z, dact, color=color, lw=1.5, ls="--", alpha=0.6, label=f"Derivative: {deriv_note}")
            ax.axhline(0, color="k", lw=0.4, alpha=0.5)
            ax.axvline(0, color="k", lw=0.4, alpha=0.5)
            ax.set(xlabel="$z$", title=name)
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="lower right", fontsize=8)
            ax2.set_ylabel("Derivative", fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Activation functions (solid) and their derivatives (dashed)", fontsize=11, y=1.01)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def uat_header(mo):
    mo.md(r"""
    ---
    ## 3. Universal Approximation Theorem (UAT)

    Despite having only one hidden layer, a neural network can approximate *any* continuous function.

    **Theorem (Cybenko, 1989):** For any continuous $f: [0,1]^d \to \mathbb{R}$ and $\varepsilon > 0$,
    there exists a single-hidden-layer network $g(x) = \sum_{j=1}^N c_j \sigma(\langle \mathbf{w}_j, \mathbf{x} \rangle + b_j)$
    such that $\sup_{x} |f(x) - g(x)| < \varepsilon$.

    **Constructive proof via ReLU bumps:**

    A "bump" function on $[a,b]$ is built from three ReLU units:
    $$\text{bump}_{[a,b]}(x) = \text{ReLU}(x-a) - 2\cdot\text{ReLU}\!\left(x - \tfrac{a+b}{2}\right) + \text{ReLU}(x-b)$$

    Scaling each bump to match $f$ at that interval gives a piecewise-linear approximation.
    As the number of bumps increases, the approximation improves — by uniform continuity of $f$.

    **What UAT does NOT say:**
    - It doesn't say how wide the network must be (could be exponential)
    - It doesn't say training will find the right weights
    - It doesn't say one hidden layer is *efficient* — depth helps enormously in practice

    Use the slider to see how more bumps improve the approximation of $f(x) = \sin(2\pi x)$.
    """)
    return


@app.cell
def n_bumps_slider(mo):
    n_bumps_s = mo.ui.slider(
        start=1, stop=30, value=4, step=1,
        label="Number of ReLU bumps $N$",
    )
    n_bumps_s
    return (n_bumps_s,)


@app.cell
def uat_viz(n_bumps_s, np, plt):
    def _():
        _n = n_bumps_s.value
        _x = np.linspace(0, 1, 1000)
        _f = np.sin(2 * np.pi * _x)

        def _bump(x, a, b):
            _mid = (a + b) / 2
            return (np.maximum(0, x - a)
                    - 2 * np.maximum(0, x - _mid)
                    + np.maximum(0, x - b))

        _breakpoints = np.linspace(0, 1, _n + 1)
        _midpoints = 0.5 * (_breakpoints[:-1] + _breakpoints[1:])
        _f_at_mid = np.sin(2 * np.pi * _midpoints)
        _approx = np.zeros_like(_x)
        for _k in range(_n):
            _scale = _f_at_mid[_k] * _n
            _approx += _scale * _bump(_x, _breakpoints[_k], _breakpoints[_k + 1])

        _error = np.max(np.abs(_f - _approx))

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        ax.plot(_x, _f, "#e74c3c", lw=2.5, label="True $f(x) = \\sin(2\\pi x)$")
        ax.plot(_x, _approx, "#3498db", lw=2, ls="--", label=f"ReLU approx ($N={_n}$ bumps)")
        for _bp in _breakpoints:
            ax.axvline(_bp, color="gray", lw=0.5, alpha=0.4)
        ax.set(xlabel="$x$", ylabel="$f(x)$",
               title=f"UAT: $N={_n}$ bumps, max error = {_error:.3f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _ns = np.arange(1, 31)
        _errors = []
        for _ni in _ns:
            _bps = np.linspace(0, 1, _ni + 1)
            _mids = 0.5 * (_bps[:-1] + _bps[1:])
            _fmids = np.sin(2 * np.pi * _mids)
            _app = np.zeros_like(_x)
            for _k in range(_ni):
                _app += _fmids[_k] * _ni * _bump(_x, _bps[_k], _bps[_k + 1])
            _errors.append(np.max(np.abs(_f - _app)))
        ax.plot(_ns, _errors, "#3498db", lw=2, marker="o", markersize=4)
        ax.axvline(_n, color="#e74c3c", ls="--", lw=2, label=f"Current $N={_n}$")
        ax.scatter([_n], [_error], s=100, c="#e74c3c", zorder=5)
        ax.set(xlabel="$N$ (bumps = hidden units)", ylabel="Max approximation error",
               title="More units → better approximation (UAT)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def forward_pass_header(mo):
    mo.md(r"""
    ---
    ## 4. Forward Pass — Computing Predictions

    A feedforward network with $L$ layers computes:
    $$\mathbf{h}^{(0)} = \mathbf{x} \quad \text{(input)}$$
    $$\mathbf{u}^{(l)} = W^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)} \quad \text{(pre-activation)}$$
    $$\mathbf{h}^{(l)} = \sigma\bigl(\mathbf{u}^{(l)}\bigr) \quad \text{(activation)}$$
    $$\hat{y} = \mathbf{h}^{(L)} \quad \text{(output)}$$

    Both $\mathbf{u}^{(l)}$ and $\mathbf{h}^{(l)}$ must be **stored** during the forward pass —
    they are needed for computing gradients in the backward pass.

    Below: watch how a simple 2-layer ReLU network builds its function from piecewise linear pieces.
    Each hidden unit contributes one "hinge", and their combination produces a nonlinear map.
    """)
    return


@app.cell
def forward_pass_viz(np, plt):
    def _():
        _rng = np.random.default_rng(7)
        _n_hidden = 8

        # Random small MLP: R -> R with 2 layers
        _W1 = _rng.normal(0, 1, (_n_hidden, 1))
        _b1 = _rng.normal(0, 1, _n_hidden)
        _W2 = _rng.normal(0, 1, (1, _n_hidden)) * 0.5
        _b2 = _rng.normal(0, 0.5, 1)

        _x = np.linspace(-3, 3, 500).reshape(-1, 1)
        _u1 = _x @ _W1.T + _b1
        _h1 = np.maximum(0, _u1)
        _out = (_h1 @ _W2.T + _b2).ravel()

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        ax = axes[0]
        _colors = plt.cm.tab10(np.linspace(0, 1, _n_hidden))
        for _j in range(_n_hidden):
            ax.plot(_x.ravel(), _h1[:, _j], color=_colors[_j], lw=1.5, alpha=0.7,
                    label=f"$h_{_j+1}$" if _j < 4 else None)
        ax.set(xlabel="$x$", ylabel="$h_j^{(1)}$",
               title=f"Hidden layer: {_n_hidden} ReLU units\n(each is a hinge function)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for _j in range(_n_hidden):
            _contrib = _W2[0, _j] * _h1[:, _j]
            ax.plot(_x.ravel(), _contrib, color=_colors[_j], lw=1.2, alpha=0.5)
        ax.set(xlabel="$x$", ylabel=r"$w_j^{(2)} h_j^{(1)}$",
               title="Weighted contributions\n(positive and negative hinges)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(_x.ravel(), _out, "#3498db", lw=2.5)
        ax.set(xlabel="$x$", ylabel=r"$\hat{y}$",
               title="Network output = sum of contributions\n(piecewise linear, 8 breakpoints)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)

        plt.suptitle("Forward pass: from hinges to complex function", fontsize=11, y=1.01)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def loss_header(mo):
    mo.md(r"""
    ---
    ## 5. Loss Functions

    The network is trained by minimizing the **empirical risk** over all parameters $\theta$:
    $$\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(\mathbf{x}_i), y_i)$$

    The choice of loss depends on the task:

    | Task | Output | Loss |
    |------|--------|------|
    | Regression | $\hat{y} \in \mathbb{R}$ | MSE: $\frac{1}{2}\|\hat{y} - y\|^2$ |
    | Binary classification | $\hat{p} = \sigma(z) \in (0,1)$ | Cross-entropy: $-y\log\hat{p} - (1-y)\log(1-\hat{p})$ |
    | Multi-class ($K$ classes) | $\hat{\mathbf{p}} = \text{softmax}(\mathbf{z})$ | $-\log\hat{p}_{y}$ |

    **Softmax** turns raw logits $\mathbf{z} \in \mathbb{R}^K$ into a probability distribution:
    $$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

    Binary cross-entropy is the same loss as logistic regression — a neural network with sigmoid output
    and cross-entropy loss is exactly logistic regression if there are no hidden layers.
    """)
    return


@app.cell
def loss_viz(np, plt):
    def _():
        _p_hat = np.linspace(0.001, 0.999, 500)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        ax = axes[0]
        _y_hat = np.linspace(-3, 3, 500)
        ax.plot(_y_hat, 0.5 * _y_hat**2, "#3498db", lw=2.5, label="MSE: $\\frac{1}{2}(\\hat{y}-y)^2$")
        ax.plot(_y_hat, np.abs(_y_hat), "#e74c3c", lw=2, ls="--", label="MAE: $|\\hat{y}-y|$")
        ax.set(xlabel="Residual $\\hat{y} - y$", ylabel="Loss",
               title="Regression losses\n(centered at truth)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(_p_hat, -np.log(_p_hat), "#3498db", lw=2.5, label="$y=1$: $-\\log(\\hat{p})$")
        ax.plot(_p_hat, -np.log(1 - _p_hat), "#e74c3c", lw=2.5, ls="--", label="$y=0$: $-\\log(1-\\hat{p})$")
        ax.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
        ax.set(xlabel="Predicted probability $\\hat{p}$", ylabel="Loss",
               title="Binary cross-entropy\n(large when confidently wrong)",
               ylim=(0, 5))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        _K = 5
        _z = np.linspace(-3, 3, 200)
        _softmax_rows = []
        for _zi in _z:
            _logits = np.array([_zi, 0.5, -0.5, 1.0, -1.0])
            _sm = np.exp(_logits) / np.exp(_logits).sum()
            _softmax_rows.append(_sm)
        _softmax_rows = np.array(_softmax_rows)
        _colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        for _k in range(_K):
            ax.plot(_z, _softmax_rows[:, _k], color=_colors[_k], lw=2,
                    label=f"Class {_k+1}" + (" (swept)" if _k == 0 else ""))
        ax.set(xlabel="$z_1$ (logit for class 1, others fixed)", ylabel="$\\hat{p}_k = \\text{softmax}(z)_k$",
               title="Softmax: competition between classes", ylim=(0, 1))
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.suptitle("Loss functions for neural networks", fontsize=11, y=1.01)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def backprop_header(mo):
    mo.md(r"""
    ---
    ## 6. Backpropagation

    To train, we need $\frac{\partial \mathcal{L}}{\partial W^{(l)}}$ for every layer $l$.
    **Naive approach (forward mode):** compute the gradient for each weight separately.
    Cost: $O(WE)$ where $W$ = number of weights, $E$ = number of edges.

    **The key insight:** Many terms are shared. For two weights $\theta_i, \theta_k$ in the same layer,
    both gradients involve the same downstream factor $\frac{\partial \mathcal{L}}{\partial v}$ — computed twice in forward mode.

    **Backpropagation** reverses the order: compute $\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{u}^{(l)}}$
    (the "error signal") for all layers in one backward sweep, then read off weight gradients for free:

    $$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(L)}} \odot \sigma'(\mathbf{u}^{(L)})$$
    $$\delta^{(l)} = \bigl(W^{(l+1)}\bigr)^\top \delta^{(l+1)} \odot \sigma'(\mathbf{u}^{(l)})$$
    $$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \bigl(\mathbf{h}^{(l-1)}\bigr)^\top$$

    **Complexity:** Each edge visited once forward, once backward → $O(E)$ total.
    Compare: forward-mode autodiff costs $O(WE)$.
    """)
    return


@app.cell
def backprop_viz(np, plt):
    def _():
        _rng = np.random.default_rng(3)
        _n, _d = 200, 2
        _X = _rng.standard_normal((_n, _d))
        _true_w = np.array([2.0, -1.5])
        _y = (_X @ _true_w + _rng.normal(0, 0.5, _n) > 0).astype(float)

        # 2-layer network: d → 4 → 1 (sigmoid output)
        _W1 = _rng.normal(0, 0.5, (4, _d))
        _b1 = np.zeros(4)
        _W2 = _rng.normal(0, 0.5, (1, 4))
        _b2 = np.zeros(1)

        def _forward(X, W1, b1, W2, b2):
            _u1 = X @ W1.T + b1
            _h1 = np.maximum(0, _u1)
            _u2 = _h1 @ W2.T + b2
            _p = 1 / (1 + np.exp(-_u2.ravel()))
            return _u1, _h1, _u2, _p

        def _loss(p, y):
            _p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.mean(y * np.log(_p) + (1 - y) * np.log(1 - _p))

        def _backward(X, y, u1, h1, p, W2):
            _n = len(y)
            _delta2 = (p - y).reshape(-1, 1) / _n
            _dW2 = _delta2.T @ h1
            _db2 = _delta2.sum(axis=0)
            _delta1 = (_delta2 @ W2) * (u1 > 0)
            _dW1 = _delta1.T @ X
            _db1 = _delta1.sum(axis=0)
            return _dW1, _db1, _dW2, _db2

        _lr = 0.5
        _losses = []
        for _t in range(300):
            _u1, _h1, _u2, _p = _forward(_X, _W1, _b1, _W2, _b2)
            _losses.append(_loss(_p, _y))
            _dW1, _db1, _dW2, _db2 = _backward(_X, _y, _u1, _h1, _p, _W2)
            _W1 -= _lr * _dW1
            _b1 -= _lr * _db1
            _W2 -= _lr * _dW2
            _b2 -= _lr * _db2

        _, _, _, _p_final = _forward(_X, _W1, _b1, _W2, _b2)
        _acc = np.mean((_p_final >= 0.5) == _y)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        ax = axes[0]
        ax.plot(_losses, "#3498db", lw=2)
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss",
               title=f"Backprop training (final acc = {_acc:.1%})")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _xg = np.linspace(-3, 3, 150)
        _yg = np.linspace(-3, 3, 150)
        _Xg, _Yg = np.meshgrid(_xg, _yg)
        _Xgf = np.column_stack([_Xg.ravel(), _Yg.ravel()])
        _, _, _, _Pg = _forward(_Xgf, _W1, _b1, _W2, _b2)
        _Pg = _Pg.reshape(150, 150)
        ax.contourf(_Xg, _Yg, _Pg, levels=20, cmap="RdBu_r", alpha=0.5, vmin=0, vmax=1)
        ax.contour(_Xg, _Yg, _Pg, levels=[0.5], colors="k", linewidths=2)
        ax.scatter(_X[_y == 0, 0], _X[_y == 0, 1], s=15, c="#e74c3c", alpha=0.5)
        ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], s=15, c="#3498db", alpha=0.5)
        ax.set(xlabel="$x_1$", ylabel="$x_2$", title="Learned decision boundary\n(nonlinear — hidden layer helps)",
               aspect="equal", xlim=(-3, 3), ylim=(-3, 3))
        ax.grid(True, alpha=0.2)

        ax = axes[2]
        _methods = ["Numerical\ndiff", "Forward\nmode", "Backprop"]
        _costs = [100, 100, 1]
        _bar_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        ax.bar(_methods, _costs, color=_bar_colors, edgecolor="k", linewidth=0.5)
        ax.set(ylabel="Relative cost (× number of edges $E$)",
               title="Gradient computation cost\n(backprop is $O(E)$, others $O(WE)$)")
        ax.text(2, 1 + 2, "1×E\n(backprop!)", ha="center", va="bottom", fontweight="bold", color="#2ecc71")
        ax.text(0, 100 + 2, "W×E", ha="center", va="bottom", fontsize=9)
        ax.text(1, 100 + 2, "W×E", ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Backpropagation: efficient gradient computation via the chain rule", fontsize=11, y=1.01)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def vanishing_header(mo):
    mo.md(r"""
    ---
    ## 7. Vanishing & Exploding Gradients

    During backprop, the gradient at layer $l$ involves a **product of terms** from all deeper layers:
    $$\frac{\partial \mathcal{L}}{\partial W^{(1)}} \propto \prod_{k=2}^{L} \sigma'(\mathbf{u}^{(k)}) \cdot W^{(k)}$$

    **Vanishing gradients (sigmoid/tanh):**
    Since $\sigma'(z) \leq \frac{1}{4}$ for sigmoid, after $L$ layers:
    $$\left\|\frac{\partial \mathcal{L}}{\partial W^{(1)}}\right\| \leq \left(\frac{1}{4}\right)^{L-1}$$

    Early layers get essentially zero gradient — they stop learning.

    **Exploding gradients:** If weight matrices have $\|W^{(k)}\| > 1$, gradients grow exponentially.

    **Solutions:**
    - **ReLU:** $\sigma'(z) \in \{0, 1\}$ — no shrinking in active region
    - **Xavier/He initialization:** Scale weights so variance stays constant across layers
      - Xavier: $\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ (sigmoid/tanh)
      - He: $\text{Var}(w) = \frac{2}{n_{\text{in}}}$ (ReLU)
    - **Residual connections:** $\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + F(\mathbf{h}^{(l)})$ — gradient highway bypasses layers
    - **Gradient clipping:** Cap $\|\nabla\|$ to prevent explosion

    Use the slider below to see how sigmoid gradients vanish with depth.
    """)
    return


@app.cell
def depth_slider(mo):
    depth_s = mo.ui.slider(
        start=1, stop=20, value=5, step=1,
        label="Network depth $L$",
    )
    depth_s
    return (depth_s,)


@app.cell
def vanishing_viz(depth_s, np, plt):
    def _():
        _L = depth_s.value
        _layers = np.arange(1, _L + 1)

        # Sigmoid: derivative ≤ 1/4
        _sig_grad = (0.25) ** (_L - _layers)
        # ReLU: derivative = 1 (in active region)
        _relu_grad = np.ones(_L)
        # Exploding: weights with norm 1.5
        _expl_grad = (1.5) ** (_L - _layers)
        _expl_grad = np.clip(_expl_grad, 0, 1e6)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.semilogy(_layers, _sig_grad, "#e74c3c", lw=2.5, marker="o", markersize=5,
                    label=f"Sigmoid ($\\sigma'(z) \\leq 0.25$)")
        ax.semilogy(_layers, _relu_grad, "#2ecc71", lw=2.5, marker="s", markersize=5,
                    label="ReLU ($\\sigma'(z) = 1$, active)")
        ax.semilogy(_layers, _expl_grad, "#3498db", lw=2, marker="^", markersize=5, ls="--",
                    label="Exploding ($\\|W\\| = 1.5$)")
        ax.axhline(1e-4, color="gray", ls=":", lw=1, alpha=0.7, label="Vanishing threshold")
        ax.set(xlabel="Layer (from output ← input)", ylabel="Gradient magnitude (log scale)",
               title=f"Gradient magnitude at each layer (depth $L={_L}$)\nLayer 1 = closest to input")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _depths = np.arange(1, 25)
        _sig_final = (0.25) ** (_depths - 1)
        ax.semilogy(_depths, _sig_final, "#e74c3c", lw=2.5, label="Sigmoid gradient at layer 1")
        ax.axvline(_L, color="gray", ls="--", lw=2, label=f"Current $L = {_L}$")
        ax.scatter([_L], [(0.25) ** (_L - 1)], s=100, c="#e74c3c", zorder=5)
        ax.axhline(1e-4, color="k", ls=":", lw=1, alpha=0.5, label="~0 threshold")
        ax.set(xlabel="Depth $L$", ylabel="Gradient at layer 1 (log scale)",
               title="Sigmoid gradients vanish exponentially with depth")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def sgd_header(mo):
    mo.md(r"""
    ---
    ## 8. SGD and Convergence

    The full gradient over $n$ examples costs $O(n)$ per step — prohibitive for large datasets.

    **Stochastic gradient descent (SGD)** uses one random sample $i$ per step:
    $$\theta_{t+1} = \theta_t - \eta_t \nabla \ell(f_\theta(\mathbf{x}_i), y_i)$$

    The key property: it's an **unbiased estimator**:
    $$\mathbb{E}_i[\nabla \ell_i(\theta)] = \nabla \mathcal{L}(\theta)$$

    **Mini-batch SGD** averages over $B$ samples, reducing variance while staying tractable.

    **Convergence rates (from notes):**
    - **Strongly convex + $L$-smooth:** GD achieves linear convergence $\|\theta_t - \theta^*\|^2 \leq (1 - \mu/L)^t \|\theta_0 - \theta^*\|^2$
    - **Non-convex + $L$-smooth:** SGD with $\eta = 1/(L\sqrt{T})$ finds a near-stationary point at rate $O(1/\sqrt{T})$

    **Neural network losses are non-convex** (multiple local minima, saddle points).
    Yet SGD works well in practice — empirically, local minima found by SGD generalize well.
    """)
    return


@app.cell
def sgd_viz(np, plt):
    def _():
        _rng = np.random.default_rng(99)

        # Simple non-convex loss landscape (1D for visualization)
        def _loss_landscape(theta):
            return (np.sin(3 * theta) + 0.5 * theta**2 + 0.3 * np.cos(7 * theta))

        def _grad_landscape(theta):
            return 3 * np.cos(3 * theta) + theta - 2.1 * np.sin(7 * theta)

        _theta_grid = np.linspace(-3, 3, 500)
        _L_grid = _loss_landscape(_theta_grid)

        # GD trajectory
        _theta_gd = -2.5
        _traj_gd = [_theta_gd]
        for _ in range(50):
            _theta_gd -= 0.05 * _grad_landscape(_theta_gd)
            _traj_gd.append(_theta_gd)

        # SGD trajectory (add noise to gradient)
        _theta_sgd = -2.5
        _traj_sgd = [_theta_sgd]
        for _ in range(200):
            _g_noisy = _grad_landscape(_theta_sgd) + _rng.normal(0, 0.8)
            _theta_sgd -= 0.05 * _g_noisy
            _theta_sgd = np.clip(_theta_sgd, -3, 3)
            _traj_sgd.append(_theta_sgd)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.plot(_theta_grid, _L_grid, "k-", lw=2.5, label="Loss $\\mathcal{L}(\\theta)$")
        _traj_gd_arr = np.array(_traj_gd)
        ax.plot(_traj_gd_arr, _loss_landscape(_traj_gd_arr), "#3498db",
                marker="o", markersize=3, lw=1.5, label="GD (smooth)")
        _traj_sgd_arr = np.array(_traj_sgd[::4])
        ax.scatter(_traj_sgd_arr, _loss_landscape(_traj_sgd_arr),
                   s=15, c="#e74c3c", alpha=0.4, label="SGD (noisy)")
        ax.set(xlabel=r"$\theta$", ylabel=r"$\mathcal{L}(\theta)$",
               title="Non-convex landscape: GD vs SGD\nSGD noise can escape local minima")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _batch_sizes = [1, 4, 16, 64, 256]
        _n_data = 1000
        _true_grad = 1.5

        _var_per_bs = []
        for _B in _batch_sizes:
            _sample_grads = []
            for _ in range(500):
                _noise = _rng.normal(0, 3, _B)
                _sample_grads.append(_true_grad + _noise.mean())
            _var_per_bs.append(np.var(_sample_grads))

        ax.loglog(_batch_sizes, _var_per_bs, "#3498db", lw=2.5, marker="o", markersize=8)
        ax.loglog(_batch_sizes, [9 / _B for _B in _batch_sizes], "#e74c3c", lw=2, ls="--",
                  label=r"$\sigma^2/B$ (theory)")
        ax.set(xlabel="Batch size $B$", ylabel="Gradient variance",
               title="Variance ∝ $1/B$: larger batch = less noisy gradient")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def summary_header(mo):
    mo.md(r"""
    ---
    ## Summary

    | Concept | Key Formula | Insight |
    |---------|------------|---------|
    | **Perceptron** | $T \leq 1/\gamma^2$ mistakes | Margin controls convergence speed |
    | **Activation** | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | Must be nonlinear; ReLU avoids saturation |
    | **UAT** | $g = \sum c_j \sigma(\langle w_j, x\rangle + b_j)$ | One hidden layer suffices (but may be wide) |
    | **Forward pass** | $\mathbf{h}^{(l)} = \sigma(W^{(l)}\mathbf{h}^{(l-1)} + b^{(l)})$ | Store $u^{(l)}, h^{(l)}$ for backprop |
    | **Backprop** | $\delta^{(l)} = (W^{(l+1)})^\top\delta^{(l+1)} \odot \sigma'(u^{(l)})$ | $O(E)$ vs $O(WE)$ for forward mode |
    | **Vanishing** | $\|\nabla W^{(1)}\| \leq (1/4)^{L-1}$ | ReLU + He init + residuals fix this |
    | **SGD** | $\theta_{t+1} = \theta_t - \eta_t \nabla \ell_i$ | Unbiased, $O(1)$ per step, non-convex works |

    **The progression of ideas (from notes):**

    | Model | Key Idea | Limitation |
    |-------|---------|-----------|
    | Perceptron | Linear separator | Linearly separable only |
    | SVM | Max margin + kernel | Fixed feature map |
    | Neural network | Nonlinear activations → UAT | Non-convex training |
    | Deep network | Depth = exponential expressiveness | Vanishing/exploding gradients |
    """)
    return


if __name__ == "__main__":
    app.run()
