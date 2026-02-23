"""
Training Neural Networks — companion notebook to nn-training.tex

Covers the same material with code and plots instead of proofs.
"""

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def title():
    import marimo as mo

    mo.md(r"""
    # Training Neural Networks

    Companion to `notes/detailed-notes/nn-training.tex`.
    Same material, code and plots instead of proofs.

    1. Momentum — accelerating SGD
    2. Adaptive Learning Rates — AdaGrad, RMSProp, Adam
    3. Weight Initialization — Xavier and He
    4. Normalization — BatchNorm, LayerNorm
    5. Regularization — weight decay, dropout
    6. Learning Rate Schedules — warmup + cosine
    7. Bias-Variance and Double Descent
    """)
    return (mo,)


@app.cell
def imports():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams.update({"figure.dpi": 120, "axes.titlesize": 12})
    return np, plt


# ---------------------------------------------------------------------------
# 1. Momentum
# ---------------------------------------------------------------------------


@app.cell
def momentum_header(mo):
    mo.md(r"""
    ---
    ## 1. Momentum

    Plain SGD oscillates in high-curvature directions.
    Momentum smooths the trajectory by accumulating a velocity:

    $$\mathbf{v}_{t+1} = \mu\,\mathbf{v}_t + \mathbf{g}_t, \qquad
      \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\,\mathbf{v}_{t+1}$$

    The velocity $\mathbf{v}_t$ is an exponential moving average of past gradients.
    Unrolling the recursion:
    $$\mathbf{v}_t = \sum_{s=0}^{t-1} \mu^{t-1-s}\,\mathbf{g}_s$$

    Gradients that consistently point in the same direction accumulate;
    gradients that oscillate cancel out.

    **Effective learning rate:** In a direction where the gradient is constant $g$,
    steady-state velocity is $v = g/(1-\mu)$. For $\mu = 0.9$ this is a $10\times$ amplification.

    Use the slider to see how $\mu$ affects the trajectory on an elongated loss surface.
    """)
    return


@app.cell
def momentum_slider(mo):
    mu_s = mo.ui.slider(
        start=0.0, stop=0.99, value=0.9, step=0.01,
        label=r"Momentum $\mu$",
    )
    mu_s
    return (mu_s,)


@app.cell
def momentum_viz(mu_s, np, plt):
    def _():
        _mu = mu_s.value
        _eta = 0.02

        def _loss(t1, t2):
            return 0.5 * t1**2 + 10 * t2**2

        def _grad(t1, t2):
            return np.array([t1, 20 * t2])

        _theta = np.array([-4.0, 3.0])
        _v = np.zeros(2)
        _traj = [_theta.copy()]
        for _ in range(120):
            _g = _grad(*_theta)
            _v = _mu * _v + _g
            _theta = _theta - _eta * _v
            _traj.append(_theta.copy())
        _traj = np.array(_traj)

        _theta_sgd = np.array([-4.0, 3.0])
        _traj_sgd = [_theta_sgd.copy()]
        for _ in range(120):
            _g = _grad(*_theta_sgd)
            _theta_sgd = _theta_sgd - _eta * _g
            _traj_sgd.append(_theta_sgd.copy())
        _traj_sgd = np.array(_traj_sgd)

        _t1 = np.linspace(-5, 5, 200)
        _t2 = np.linspace(-4, 4, 200)
        _T1, _T2 = np.meshgrid(_t1, _t2)
        _L = _loss(_T1, _T2)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.contour(_T1, _T2, _L, levels=30, cmap="Blues", alpha=0.6)
        ax.plot(_traj_sgd[:, 0], _traj_sgd[:, 1], "#e74c3c", lw=1.5, marker=".", markersize=2,
                label="SGD ($\\mu=0$)")
        ax.plot(_traj[:, 0], _traj[:, 1], "#3498db", lw=2, marker=".", markersize=3,
                label=f"SGD+Momentum ($\\mu={_mu:.2f}$)")
        ax.scatter([0], [0], s=100, c="k", marker="*", zorder=5, label="Minimum")
        ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$",
               title=f"$\\mu = {_mu:.2f}$: momentum smooths oscillations",
               xlim=(-5, 5), ylim=(-4, 4))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        ax = axes[1]
        _loss_plain = np.array([_loss(*t) for t in _traj_sgd])
        _loss_mom = np.array([_loss(*t) for t in _traj])
        ax.semilogy(_loss_plain, "#e74c3c", lw=1.5, label="SGD")
        ax.semilogy(_loss_mom, "#3498db", lw=2, label=f"Momentum ($\\mu={_mu:.2f}$)")
        ax.set(xlabel="Step", ylabel="Loss (log scale)",
               title="Momentum converges faster on ill-conditioned loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 2. Adaptive Learning Rates
# ---------------------------------------------------------------------------


@app.cell
def adam_header(mo):
    mo.md(r"""
    ---
    ## 2. Adaptive Learning Rates

    Different parameters may have vastly different gradient magnitudes.
    Adaptive methods maintain per-parameter learning rates.

    **AdaGrad** divides by the root sum of squared past gradients:
    $$\theta_{t+1,j} = \theta_{t,j} - \frac{\eta}{\sqrt{G_{t,j} + \epsilon}}\,g_{t,j},
      \qquad G_{t,j} = \sum_{s=0}^t g_{s,j}^2$$

    Problem: $G_{t,j}$ grows monotonically, so the effective LR decays to zero.

    **RMSProp** fixes this with an exponential moving average:
    $$r_{t+1,j} = \beta\,r_{t,j} + (1-\beta)\,g_{t,j}^2$$

    **Adam** combines momentum with adaptive rates. It maintains:

    - First moment (mean): $\mathbf{m}_{t+1} = \beta_1\,\mathbf{m}_t + (1-\beta_1)\,\mathbf{g}_t$
    - Second moment (variance): $\mathbf{v}_{t+1} = \beta_2\,\mathbf{v}_t + (1-\beta_2)\,\mathbf{g}_t^2$

    **Bias correction** (since $\mathbf{m}_0 = \mathbf{v}_0 = 0$):
    $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \qquad
      \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

    $$\boxed{\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t
      - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}}$$

    The denominator normalizes gradients by their running RMS,
    making the effective step size roughly independent of gradient scale.
    """)
    return


@app.cell
def adam_viz(np, plt):
    def _():
        def _loss(t1, t2):
            return 0.5 * t1**2 + 25 * t2**2

        def _grad(t1, t2):
            return np.array([t1, 50 * t2])

        _start = np.array([-4.0, 3.0])
        _eta = 0.1
        _steps = 150

        def _run_sgd():
            _th = _start.copy()
            _tr = [_th.copy()]
            for _ in range(_steps):
                _th = _th - 0.005 * _grad(*_th)
                _tr.append(_th.copy())
            return np.array(_tr)

        def _run_adam(beta1=0.9, beta2=0.999, eps=1e-8):
            _th = _start.copy()
            _m = np.zeros(2)
            _v = np.zeros(2)
            _tr = [_th.copy()]
            for _t in range(1, _steps + 1):
                _g = _grad(*_th)
                _m = beta1 * _m + (1 - beta1) * _g
                _v = beta2 * _v + (1 - beta2) * _g**2
                _mh = _m / (1 - beta1**_t)
                _vh = _v / (1 - beta2**_t)
                _th = _th - _eta * _mh / (np.sqrt(_vh) + eps)
                _tr.append(_th.copy())
            return np.array(_tr)

        _traj_sgd = _run_sgd()
        _traj_adam = _run_adam()

        _t1 = np.linspace(-5, 5, 200)
        _t2 = np.linspace(-4, 4, 200)
        _T1, _T2 = np.meshgrid(_t1, _t2)
        _L = _loss(_T1, _T2)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.contour(_T1, _T2, _L, levels=30, cmap="Blues", alpha=0.6)
        ax.plot(_traj_sgd[:, 0], _traj_sgd[:, 1], "#e74c3c", lw=1.5, marker=".", markersize=2,
                label="SGD")
        ax.plot(_traj_adam[:, 0], _traj_adam[:, 1], "#2ecc71", lw=2, marker=".", markersize=3,
                label="Adam")
        ax.scatter([0], [0], s=100, c="k", marker="*", zorder=5)
        ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$",
               title="Adam adapts per-parameter: no oscillation",
               xlim=(-5, 5), ylim=(-4, 4))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        ax = axes[1]
        _loss_sgd = np.array([_loss(*t) for t in _traj_sgd])
        _loss_adam = np.array([_loss(*t) for t in _traj_adam])
        ax.semilogy(_loss_sgd, "#e74c3c", lw=1.5, label="SGD")
        ax.semilogy(_loss_adam, "#2ecc71", lw=2, label="Adam")
        ax.set(xlabel="Step", ylabel="Loss (log scale)",
               title="Adam converges faster on ill-conditioned problems")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 3. Weight Initialization
# ---------------------------------------------------------------------------


@app.cell
def init_header(mo):
    mo.md(r"""
    ---
    ## 3. Weight Initialization

    Consider a single layer $\mathbf{z} = W\mathbf{h}$ with $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$.
    If entries of $\mathbf{h}$ have variance $\sigma_h^2$ and $W_{ij}$ are i.i.d. with $\text{Var}(W_{ij}) = \sigma_W^2$:

    $$\text{Var}(z_i) = d_\text{in}\,\sigma_W^2\,\sigma_h^2$$

    After $L$ layers, the variance becomes $(d_\text{in}\,\sigma_W^2)^L\,\sigma_x^2$
    — exponentially diverging or collapsing unless $d_\text{in}\,\sigma_W^2 = 1$.

    **Xavier (Glorot & Bengio, 2010):** Preserves variance in both forward and backward pass:
    $$W_{ij} \sim \mathcal{N}\!\left(0,\;\frac{2}{d_\text{in} + d_\text{out}}\right)$$

    **He (He et al., 2015):** For ReLU, which zeroes half the units (halving the variance):
    $$W_{ij} \sim \mathcal{N}\!\left(0,\;\frac{2}{d_\text{in}}\right)$$

    Use the slider to see how variance propagates through a deep network with different initializations.
    """)
    return


@app.cell
def init_depth_slider(mo):
    init_depth_s = mo.ui.slider(
        start=2, stop=30, value=10, step=1,
        label="Network depth $L$",
    )
    init_depth_s
    return (init_depth_s,)


@app.cell
def init_viz(init_depth_s, np, plt):
    def _():
        _L = init_depth_s.value
        _rng = np.random.default_rng(0)
        _d = 256
        _x = _rng.standard_normal(_d)

        def _propagate(init_scale, activation="relu"):
            _h = _x.copy()
            _vars = [np.var(_h)]
            for _ in range(_L):
                _W = _rng.normal(0, init_scale, (_d, _d))
                _h = _W @ _h
                if activation == "relu":
                    _h = np.maximum(0, _h)
                _vars.append(np.var(_h))
            return np.array(_vars)

        _var_small = _propagate(np.sqrt(1 / (2 * _d)))
        _var_he = _propagate(np.sqrt(2 / _d))
        _var_big = _propagate(np.sqrt(4 / _d))
        _var_xavier = _propagate(np.sqrt(2 / (2 * _d)))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        _layers = np.arange(_L + 1)
        ax.semilogy(_layers, _var_small, "#e74c3c", lw=2, marker="o", markersize=4,
                    label=r"$\sigma_W^2 = 1/(2d)$ — too small")
        ax.semilogy(_layers, _var_xavier, "#f39c12", lw=2, marker="s", markersize=4,
                    label=r"Xavier: $\sigma_W^2 = 2/(d_\text{in}+d_\text{out})$")
        ax.semilogy(_layers, _var_he, "#2ecc71", lw=2.5, marker="^", markersize=5,
                    label=r"He: $\sigma_W^2 = 2/d_\text{in}$ ✓")
        ax.semilogy(_layers, _var_big, "#3498db", lw=2, marker="d", markersize=4,
                    label=r"$\sigma_W^2 = 4/d$ — too large")
        ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="Target variance = 1")
        ax.set(xlabel="Layer", ylabel="Activation variance (log scale)",
               title=f"Variance propagation through {_L} ReLU layers")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _h_he = _rng.standard_normal(_d)
        _h_small = _rng.standard_normal(_d)
        for _ in range(_L):
            _W = _rng.normal(0, np.sqrt(2 / _d), (_d, _d))
            _h_he = np.maximum(0, _W @ _h_he)
            _W2 = _rng.normal(0, np.sqrt(0.5 / _d), (_d, _d))
            _h_small = np.maximum(0, _W2 @ _h_small)

        ax.hist(_h_he[_h_he > 0], bins=40, alpha=0.6, color="#2ecc71", label="He init (healthy)",
                density=True)
        ax.hist(_h_small[_h_small > 0], bins=40, alpha=0.6, color="#e74c3c", label="Too small (collapsed)",
                density=True)
        ax.set(xlabel="Activation value", ylabel="Density",
               title=f"Activation distribution at layer {_L}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 4. Normalization
# ---------------------------------------------------------------------------


@app.cell
def norm_header(mo):
    mo.md(r"""
    ---
    ## 4. Normalization

    Even with good initialization, activations can drift during training.
    Normalization re-centers and re-scales activations at every layer.

    **Batch Normalization** (Ioffe & Szegedy, 2015) normalizes over the
    **batch dimension** — different examples at the same neuron:
    $$\hat{z}_i = \frac{z_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}},
      \qquad \tilde{z}_i = \gamma\,\hat{z}_i + \beta$$

    Problem for sequences: positions near the end may have few examples, making batch statistics unreliable.

    **Layer Normalization** (Ba et al., 2016) normalizes over the **feature dimension** of a single example:
    $$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma + \epsilon} + \beta,
      \qquad \mu = \frac{1}{d}\sum_{j=1}^d x_j$$

    LayerNorm depends only on a single token's vector, making it the standard choice for transformers.

    **RMSNorm** (Zhang & Sennrich, 2019) drops the mean centering:
    $$\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})},
      \qquad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_j x_j^2}$$

    Below: a deep ReLU network with and without LayerNorm.
    """)
    return


@app.cell
def norm_viz(np, plt):
    def _():
        _rng = np.random.default_rng(7)
        _d, _L = 128, 15
        _n_samples = 200

        def _propagate(use_norm=False):
            _means = []
            _stds = []
            for _ in range(_n_samples):
                _h = _rng.standard_normal(_d)
                for _l in range(_L):
                    _W = _rng.normal(0, np.sqrt(2 / _d), (_d, _d))
                    _h = _W @ _h
                    if use_norm:
                        _mu = np.mean(_h)
                        _sig = np.std(_h) + 1e-8
                        _h = (_h - _mu) / _sig
                    _h = np.maximum(0, _h)
                _means.append(np.mean(_h))
                _stds.append(np.std(_h))
            return np.array(_means), np.array(_stds)

        _means_raw, _stds_raw = _propagate(use_norm=False)
        _means_ln, _stds_ln = _propagate(use_norm=True)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.hist(_stds_raw, bins=30, alpha=0.6, color="#e74c3c", label="No normalization", density=True)
        ax.hist(_stds_ln, bins=30, alpha=0.6, color="#2ecc71", label="With LayerNorm", density=True)
        ax.set(xlabel="Std of output activations", ylabel="Density",
               title=f"Output activation spread after {_L} layers\n(over {_n_samples} random inputs)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _h_raw = _rng.standard_normal(_d)
        _h_ln = _rng.standard_normal(_d)
        _var_raw, _var_ln = [np.var(_h_raw)], [np.var(_h_ln)]
        for _l in range(_L):
            _W = _rng.normal(0, np.sqrt(2 / _d), (_d, _d))
            _h_raw = np.maximum(0, _W @ _h_raw)
            _var_raw.append(np.var(_h_raw))

            _W2 = _rng.normal(0, np.sqrt(2 / _d), (_d, _d))
            _h_ln = _W2 @ _h_ln
            _mu = np.mean(_h_ln)
            _sig = np.std(_h_ln) + 1e-8
            _h_ln = (_h_ln - _mu) / _sig
            _h_ln = np.maximum(0, _h_ln)
            _var_ln.append(np.var(_h_ln))

        ax.semilogy(range(_L + 1), _var_raw, "#e74c3c", lw=2, marker="o", markersize=4,
                    label="No normalization")
        ax.semilogy(range(_L + 1), _var_ln, "#2ecc71", lw=2.5, marker="s", markersize=5,
                    label="With LayerNorm")
        ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set(xlabel="Layer", ylabel="Activation variance (log scale)",
               title="LayerNorm stabilizes variance across layers")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 5. Regularization
# ---------------------------------------------------------------------------


@app.cell
def reg_header(mo):
    mo.md(r"""
    ---
    ## 5. Regularization

    Minimizing training loss alone leads to overfitting.
    Regularization introduces an inductive bias favoring simpler models.

    **$L_2$ regularization (weight decay):** Add a penalty $\frac{\lambda}{2}\|\Theta\|_2^2$.
    The gradient becomes $\nabla\hat{\mathcal{R}}_n + \lambda\Theta$, so each update shrinks parameters:
    $$\Theta_{t+1} = \underbrace{(1 - \eta\lambda)}_{\text{decay}}\Theta_t - \eta\,\nabla\hat{\mathcal{R}}_n(\Theta_t)$$

    For Adam, $L_2$ regularization and weight decay are **not equivalent**.
    AdamW decouples the decay from the adaptive learning rate.

    **Dropout:** During training, each hidden unit is zeroed with probability $p$,
    then the surviving units are scaled by $1/(1-p)$ so the expected output is unchanged.
    This is approximately equivalent to training an exponentially large ensemble of sub-networks.

    Use the slider to see how dropout probability affects a small network's learned function.
    """)
    return


@app.cell
def dropout_slider(mo):
    dropout_s = mo.ui.slider(
        start=0.0, stop=0.8, value=0.3, step=0.05,
        label=r"Dropout probability $p$",
    )
    dropout_s
    return (dropout_s,)


@app.cell
def dropout_viz(dropout_s, np, plt):
    def _():
        _p = dropout_s.value
        _rng = np.random.default_rng(42)
        _n = 50
        _x_train = np.sort(_rng.uniform(-3, 3, _n))
        _y_train = np.sin(_x_train) + _rng.normal(0, 0.3, _n)

        _d_hidden = 64
        _W1 = _rng.normal(0, np.sqrt(2), (_d_hidden, 1))
        _b1 = np.zeros(_d_hidden)
        _W2 = _rng.normal(0, np.sqrt(2 / _d_hidden), (1, _d_hidden))
        _b2 = np.zeros(1)

        _lr = 0.01
        for _epoch in range(1000):
            _u = _x_train.reshape(-1, 1) @ _W1.T + _b1
            _h = np.maximum(0, _u)
            _mask = (_rng.random((_n, _d_hidden)) > _p) / max(1 - _p, 1e-8) if _p > 0 else np.ones((_n, _d_hidden))
            _h_drop = _h * _mask
            _out = (_h_drop @ _W2.T + _b2).ravel()
            _err = _out - _y_train
            _dW2 = (_err.reshape(-1, 1) * _h_drop).mean(axis=0, keepdims=True)
            _db2 = _err.mean(keepdims=True)
            _delta = (_err.reshape(-1, 1) * _W2) * (_u > 0) * _mask
            _dW1 = (_delta.T @ _x_train.reshape(-1, 1)) / _n
            _db1 = _delta.mean(axis=0)
            _W2 -= _lr * _dW2
            _b2 -= _lr * _db2
            _W1 -= _lr * _dW1
            _b1 -= _lr * _db1

        _x_test = np.linspace(-4, 4, 300)
        _u_test = _x_test.reshape(-1, 1) @ _W1.T + _b1
        _h_test = np.maximum(0, _u_test)
        _y_pred = (_h_test @ _W2.T + _b2).ravel()

        _preds = []
        for _ in range(30):
            _m = (_rng.random((300, _d_hidden)) > _p) / max(1 - _p, 1e-8) if _p > 0 else np.ones((300, _d_hidden))
            _preds.append((_h_test * _m @ _W2.T + _b2).ravel())
        _preds = np.array(_preds)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.scatter(_x_train, _y_train, s=25, c="#95a5a6", alpha=0.7, label="Training data", zorder=3)
        ax.plot(_x_test, np.sin(_x_test), "k--", lw=1.5, alpha=0.5, label="True $\\sin(x)$")
        ax.plot(_x_test, _y_pred, "#3498db", lw=2.5, label=f"Network (dropout $p={_p:.2f}$)")
        ax.set(xlabel="$x$", ylabel="$y$",
               title=f"Dropout $p = {_p:.2f}$: {'strong regularization' if _p > 0.4 else 'mild' if _p > 0.1 else 'no'} regularization",
               ylim=(-2.5, 2.5))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for _pr in _preds[:20]:
            ax.plot(_x_test, _pr, "#3498db", alpha=0.15, lw=1)
        ax.plot(_x_test, _preds.mean(axis=0), "#e74c3c", lw=2.5, label="Mean of sub-networks")
        ax.scatter(_x_train, _y_train, s=25, c="#95a5a6", alpha=0.7, zorder=3)
        ax.set(xlabel="$x$", ylabel="$y$",
               title=f"20 stochastic forward passes (dropout ensemble)",
               ylim=(-2.5, 2.5))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 6. Learning Rate Schedules
# ---------------------------------------------------------------------------


@app.cell
def schedule_header(mo):
    mo.md(r"""
    ---
    ## 6. Learning Rate Schedules

    A constant learning rate is suboptimal: too large early on causes divergence,
    too small later slows convergence.

    **Cosine annealing:** Smooth decay from $\eta_\max$ to $\eta_\min$ over $T$ steps:
    $$\eta_t = \eta_\min + \frac{1}{2}(\eta_\max - \eta_\min)\left(1 + \cos\!\left(\frac{\pi t}{T}\right)\right)$$

    **Linear warmup:** Start small and increase linearly over $T_w$ steps:
    $$\eta_t = \eta_\max \cdot \frac{t}{T_w}, \qquad t \leq T_w$$

    Warmup prevents large gradient updates in the first few steps,
    when the model is far from any reasonable region and Adam's statistics have not yet stabilized.

    **Warmup + cosine** is the dominant schedule for modern LLM training.

    Use the slider to adjust the warmup fraction and see the combined schedule.
    """)
    return


@app.cell
def warmup_slider(mo):
    warmup_s = mo.ui.slider(
        start=0.0, stop=0.3, value=0.05, step=0.01,
        label="Warmup fraction of total steps",
    )
    warmup_s
    return (warmup_s,)


@app.cell
def schedule_viz(warmup_s, np, plt):
    def _():
        _T = 1000
        _frac = warmup_s.value
        _T_w = int(_frac * _T)
        _eta_max = 3e-4
        _eta_min = 1e-5

        _t = np.arange(_T)

        def _warmup_cosine(t, T_w, T, eta_max, eta_min):
            if t <= T_w:
                return eta_max * t / max(T_w, 1)
            return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * (t - T_w) / (T - T_w)))

        _lr_wc = np.array([_warmup_cosine(ti, _T_w, _T, _eta_max, _eta_min) for ti in _t])
        _lr_const = np.full(_T, _eta_max)
        _lr_step = _eta_max * (0.3 ** np.floor(_t / 300))
        _lr_cosine = _eta_min + 0.5 * (_eta_max - _eta_min) * (1 + np.cos(np.pi * _t / _T))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.plot(_t, _lr_const, "#95a5a6", lw=1.5, label="Constant")
        ax.plot(_t, _lr_step, "#f39c12", lw=1.5, label="Step decay")
        ax.plot(_t, _lr_cosine, "#3498db", lw=1.5, label="Cosine (no warmup)")
        ax.plot(_t, _lr_wc, "#e74c3c", lw=2.5, label=f"Warmup + cosine ({_frac:.0%} warmup)")
        if _T_w > 0:
            ax.axvline(_T_w, color="gray", ls=":", lw=1.5, alpha=0.7, label=f"End warmup (step {_T_w})")
        ax.set(xlabel="Training step", ylabel="Learning rate $\\eta_t$",
               title="Learning rate schedules comparison")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _rng = np.random.default_rng(5)
        _theta = 5.0
        _losses_wc, _losses_const = [], []
        _th_wc, _th_const = 5.0, 5.0
        for _ti in range(_T):
            _g = _th_wc + _rng.normal(0, 0.5)
            _th_wc -= _lr_wc[_ti] * 1e3 * _g
            _losses_wc.append(0.5 * _th_wc**2)

            _g2 = _th_const + _rng.normal(0, 0.5)
            _th_const -= min(_lr_const[_ti], 5e-5) * 1e3 * _g2
            _losses_const.append(0.5 * _th_const**2)

        ax.semilogy(_losses_wc, "#e74c3c", lw=2, label="Warmup + cosine")
        ax.semilogy(_losses_const, "#95a5a6", lw=1.5, label="Constant (small)")
        ax.set(xlabel="Step", ylabel="Loss (log scale)",
               title="Warmup-cosine reaches lower loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 7. Bias-Variance and Double Descent
# ---------------------------------------------------------------------------


@app.cell
def bv_header(mo):
    mo.md(r"""
    ---
    ## 7. Bias-Variance and Double Descent

    For squared loss, the expected test error decomposes as:
    $$\mathbb{E}\!\left[(y - \hat{f}(\mathbf{x}))^2\right]
      = \underbrace{\text{Var}(y|\mathbf{x})}_{\text{irreducible}}
      + \underbrace{(\mathbb{E}[\hat{f}] - f^*)^2}_{\text{bias}^2}
      + \underbrace{\text{Var}(\hat{f})}_{\text{variance}}$$

    - **High bias:** Model too simple, cannot capture the true function.
    - **High variance:** Model too complex, fits noise, unstable across training sets.

    Classical theory predicts a U-shaped test error as complexity grows.
    Modern deep learning shows **double descent**: test error spikes at the
    interpolation threshold (where the model barely fits the training data),
    then *decreases again* as the model grows further.

    The second descent is attributed to the implicit bias of gradient-based optimization,
    which selects smooth interpolators among the many that fit the data.

    **Early stopping** is an implicit form of regularization:
    it limits how far $\Theta$ moves from initialization.
    """)
    return


@app.cell
def bv_viz(np, plt):
    def _():
        _rng = np.random.default_rng(10)
        _n_train = 20
        _n_test = 200
        _n_trials = 50

        def _true_f(x):
            return np.sin(2 * x) + 0.3 * x

        _x_train_all = _rng.uniform(-3, 3, (_n_trials, _n_train))
        _y_train_all = _true_f(_x_train_all) + _rng.normal(0, 0.5, (_n_trials, _n_train))
        _x_test = np.linspace(-3, 3, _n_test)
        _y_test_true = _true_f(_x_test)

        _degrees = range(1, 18)
        _bias2_list, _var_list, _mse_list = [], [], []

        for _d in _degrees:
            _preds = []
            for _trial in range(_n_trials):
                _xt = _x_train_all[_trial]
                _yt = _y_train_all[_trial]
                _V = np.vander(_xt, _d + 1, increasing=True)
                try:
                    _coef = np.linalg.lstsq(_V, _yt, rcond=None)[0]
                except np.linalg.LinAlgError:
                    _coef = np.zeros(_d + 1)
                _V_test = np.vander(_x_test, _d + 1, increasing=True)
                _preds.append(_V_test @ _coef)
            _preds = np.array(_preds)
            _mean_pred = _preds.mean(axis=0)
            _bias2 = np.mean((_mean_pred - _y_test_true)**2)
            _var = np.mean(_preds.var(axis=0))
            _mse = np.mean(np.mean((_preds - _y_test_true[None, :])**2, axis=1))
            _bias2_list.append(_bias2)
            _var_list.append(_var)
            _mse_list.append(_mse)

        _noise = 0.25
        _degrees_arr = np.array(list(_degrees))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.plot(_degrees_arr, _bias2_list, "#3498db", lw=2, marker="o", markersize=4, label="Bias$^2$")
        ax.plot(_degrees_arr, _var_list, "#e74c3c", lw=2, marker="s", markersize=4, label="Variance")
        ax.plot(_degrees_arr, _mse_list, "k-", lw=2.5, marker="^", markersize=5, label="Total MSE")
        ax.axhline(_noise, color="gray", ls=":", lw=1.5, label=f"Irreducible noise $\\sigma^2={_noise}$")
        ax.axvline(_n_train, color="#f39c12", ls="--", lw=2, alpha=0.7, label=f"Interpolation threshold ($d=n={_n_train}$)")
        ax.set(xlabel="Polynomial degree $d$", ylabel="Error",
               title="Bias-variance decomposition\n(spike near interpolation threshold)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, min(max(_mse_list) * 1.2, 15))

        ax = axes[1]
        _train_losses = []
        _test_losses = []
        _xt = _x_train_all[0]
        _yt = _y_train_all[0]
        for _d in _degrees:
            _V = np.vander(_xt, _d + 1, increasing=True)
            _coef = np.linalg.lstsq(_V, _yt, rcond=None)[0]
            _train_pred = _V @ _coef
            _train_losses.append(np.mean((_train_pred - _yt)**2))
            _V_test = np.vander(_x_test, _d + 1, increasing=True)
            _test_pred = _V_test @ _coef
            _test_losses.append(np.mean((_test_pred - _y_test_true)**2))

        ax.semilogy(_degrees_arr, _train_losses, "#3498db", lw=2, marker="o", markersize=4, label="Train")
        ax.semilogy(_degrees_arr, np.clip(_test_losses, 0, 50), "#e74c3c", lw=2, marker="s", markersize=4, label="Test")
        ax.axvline(_n_train, color="#f39c12", ls="--", lw=2, alpha=0.7)
        ax.set(xlabel="Polynomial degree $d$", ylabel="MSE (log scale)",
               title="Training vs test error\n(double descent at interpolation)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def early_stopping_viz(np, plt):
    def _():
        _rng = np.random.default_rng(8)
        _n = 60
        _x = np.sort(_rng.uniform(-3, 3, _n))
        _y = np.sin(2 * _x) + _rng.normal(0, 0.3, _n)
        _x_val = np.sort(_rng.uniform(-3, 3, 30))
        _y_val = np.sin(2 * _x_val) + _rng.normal(0, 0.3, 30)

        _d_h = 64
        _W1 = _rng.normal(0, np.sqrt(2), (_d_h, 1))
        _b1 = np.zeros(_d_h)
        _W2 = _rng.normal(0, np.sqrt(2 / _d_h), (1, _d_h))
        _b2 = np.zeros(1)

        _train_losses, _val_losses = [], []
        _lr = 0.005

        for _epoch in range(500):
            _u = _x.reshape(-1, 1) @ _W1.T + _b1
            _h = np.maximum(0, _u)
            _out = (_h @ _W2.T + _b2).ravel()
            _train_losses.append(np.mean((_out - _y)**2))

            _u_v = _x_val.reshape(-1, 1) @ _W1.T + _b1
            _h_v = np.maximum(0, _u_v)
            _out_v = (_h_v @ _W2.T + _b2).ravel()
            _val_losses.append(np.mean((_out_v - _y_val)**2))

            _err = _out - _y
            _dW2 = (_err.reshape(-1, 1) * _h).mean(axis=0, keepdims=True)
            _db2 = _err.mean(keepdims=True)
            _delta = (_err.reshape(-1, 1) * _W2) * (_u > 0)
            _dW1 = (_delta.T @ _x.reshape(-1, 1)) / _n
            _db1 = _delta.mean(axis=0)
            _W2 -= _lr * _dW2
            _b2 -= _lr * _db2
            _W1 -= _lr * _dW1
            _b1 -= _lr * _db1

        _best_epoch = np.argmin(_val_losses)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(_train_losses, "#3498db", lw=2, label="Train loss")
        ax.plot(_val_losses, "#e74c3c", lw=2, label="Validation loss")
        ax.axvline(_best_epoch, color="#2ecc71", ls="--", lw=2,
                   label=f"Early stopping: epoch {_best_epoch}")
        ax.scatter([_best_epoch], [_val_losses[_best_epoch]], s=100, c="#2ecc71", zorder=5)
        ax.set(xlabel="Epoch", ylabel="MSE",
               title="Early stopping: stop when validation loss increases")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@app.cell
def summary(mo):
    mo.md(r"""
    ---
    ## Summary

    | Component | Standard Choice | Why |
    |-----------|----------------|-----|
    | **Optimizer** | AdamW | Adaptive LR + decoupled weight decay |
    | **Schedule** | Warmup + cosine decay | Stable early training, smooth decay |
    | **Normalization** | LayerNorm / RMSNorm (pre-norm) | Stable gradients, efficient |
    | **Initialization** | He (ReLU) / Xavier (sigmoid/tanh) | Variance preservation |
    | **Gradient clipping** | Global norm $\leq 1.0$ | Prevents explosion |
    | **Regularization** | Weight decay + dropout | Shrinkage + ensemble effect |
    | **Early stopping** | Monitor val loss | Implicit complexity control |

    **Connection to previous notebooks:**
    - `NeuralNetworksTheory.py` covered *what* neural networks compute (forward pass, backprop, UAT).
    - This notebook covers *how to train them* (optimizers, initialization, regularization, schedules).
    - The transformer notes cover the specific architecture; training considerations carry over.
    """)
    return


if __name__ == "__main__":
    app.run()
