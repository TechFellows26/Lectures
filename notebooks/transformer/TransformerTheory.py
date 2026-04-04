# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
# ]
# ///
"""
The Transformer — companion notebook to transformer.tex

Covers the same material with code and plots instead of proofs.
"""

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def title():
    import marimo as mo

    mo.md(r"""
    # The Transformer

    Companion to `notes/detailed-notes/transformer.tex`.
    Same material, code and plots instead of proofs.

    1. Token Embedding
    2. Positional Encoding
    3. Scaled Dot-Product Attention
    4. The Causal Mask
    5. Multi-Head Attention
    6. Add & Norm + Feed-Forward
    7. Generation — sampling strategies
    8. Scaling Laws
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
# 1. Token Embedding
# ---------------------------------------------------------------------------


@app.cell
def embedding_header(mo):
    mo.md(r"""
    ---
    ## 1. Token Embedding

    The input is a discrete token index $w_t \in \{1, \ldots, V\}$.
    We cannot differentiate with respect to integers, so we map each token to a
    continuous vector via a learned **embedding matrix** $\mathbf{E} \in \mathbb{R}^{V \times d}$:

    $$\mathbf{e}_t = \mathbf{E}[w_t] \in \mathbb{R}^{1 \times d}$$

    Equivalently, represent the token as a one-hot vector $\mathbf{o}_t \in \mathbb{R}^{1 \times V}$
    and compute $\mathbf{e}_t = \mathbf{o}_t\,\mathbf{E}$.
    Since $\mathbf{o}_t$ has exactly one nonzero entry, this matrix multiply reduces to a row lookup.

    The matrix $\mathbf{E}$ is learned: tokens that play similar roles get pushed to similar embeddings.

    Below: a toy embedding with $V = 8$ tokens and $d = 3$ dimensions.
    The embedding space clusters semantically related tokens.
    """)
    return


@app.cell
def embedding_viz(np, plt):
    def _():
        _rng = np.random.default_rng(42)
        _V, _d = 8, 3
        _E = _rng.normal(0, 1, (_V, _d))
        _tokens = ["the", "a", "cat", "dog", "runs", "walks", "big", "small"]
        _colors = ["#95a5a6", "#95a5a6", "#e74c3c", "#e74c3c", "#3498db", "#3498db", "#2ecc71", "#2ecc71"]

        fig = plt.figure(figsize=(13, 5))
        ax1 = fig.add_subplot(121, projection="3d")
        for _i in range(_V):
            ax1.scatter(*_E[_i], s=80, c=_colors[_i], zorder=5)
            ax1.text(_E[_i, 0] + 0.1, _E[_i, 1] + 0.1, _E[_i, 2] + 0.1,
                     f'"{_tokens[_i]}"', fontsize=9)
        ax1.set(xlabel="$d_1$", ylabel="$d_2$", zlabel="$d_3$")
        ax1.set_title("Embedding space ($d=3$)\nSimilar tokens cluster together", fontsize=11)

        ax2 = fig.add_subplot(122)
        _im = ax2.imshow(_E, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax2.set_xticks(range(_d))
        ax2.set_xticklabels([f"$d_{k+1}$" for k in range(_d)])
        ax2.set_yticks(range(_V))
        ax2.set_yticklabels([f'"{t}"' for t in _tokens], fontsize=9)
        ax2.set_title("Embedding matrix $\\mathbf{E} \\in \\mathbb{R}^{V \\times d}$", fontsize=11)
        plt.colorbar(_im, ax=ax2, shrink=0.8)

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 2. Positional Encoding
# ---------------------------------------------------------------------------


@app.cell
def pos_header(mo):
    mo.md(r"""
    ---
    ## 2. Positional Encoding

    The embedding $\mathbf{E}[w_t]$ depends only on the token identity, not its position $t$.
    But "dog bites man" and "man bites dog" contain the same tokens —
    position matters. We inject position by **adding** a positional signal:

    $$\mathbf{x}_t = \mathbf{E}[w_t] + \mathbf{P}[t]$$

    Vaswani et al. (2017) used a deterministic sinusoidal encoding:
    $$\mathbf{P}[t,\, 2k] = \sin\!\left(\frac{t}{10000^{2k/d}}\right), \qquad
      \mathbf{P}[t,\, 2k+1] = \cos\!\left(\frac{t}{10000^{2k/d}}\right)$$

    Each pair of dimensions oscillates at a different frequency, giving each position a unique signature.
    Low-index dimensions oscillate fast (local position), high-index dimensions oscillate slowly (global position).

    Key property: $\mathbf{P}[t]$ and $\mathbf{P}[t + \Delta]$ have a fixed linear relationship
    (rotation in sin/cos space), allowing the model to learn relative distances.
    """)
    return


@app.cell
def pos_viz(np, plt):
    def _():
        _d = 64
        _N = 100

        _P = np.zeros((_N, _d))
        _pos = np.arange(_N)[:, None]
        _div = 10000 ** (2 * np.arange(_d // 2)[None, :] / _d)
        _P[:, 0::2] = np.sin(_pos / _div)
        _P[:, 1::2] = np.cos(_pos / _div)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        _im = ax.imshow(_P, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(xlabel="Dimension $k$", ylabel="Position $t$",
               title="Positional encoding $\\mathbf{P} \\in \\mathbb{R}^{N \\times d}$\n(sinusoidal)")
        plt.colorbar(_im, ax=ax, shrink=0.8)

        ax = axes[1]
        for _k, _c in [(0, "#e74c3c"), (2, "#3498db"), (10, "#2ecc71"), (30, "#f39c12")]:
            ax.plot(np.arange(_N), _P[:, _k], color=_c, lw=1.5,
                    label=f"$k={_k}$ (freq $= 1/10000^{{{_k}/{_d}}}$)")
        ax.set(xlabel="Position $t$", ylabel="Encoding value",
               title="Different dimensions = different frequencies")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        _sim = _P @ _P.T
        _im2 = ax.imshow(_sim, cmap="viridis", aspect="equal")
        ax.set(xlabel="Position $j$", ylabel="Position $i$",
               title="$\\mathbf{P}\\mathbf{P}^\\top$: each position is unique\n(diagonal dominant)")
        plt.colorbar(_im2, ax=ax, shrink=0.8)

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 3. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------


@app.cell
def attention_header(mo):
    mo.md(r"""
    ---
    ## 3. Scaled Dot-Product Attention

    To compute a new representation of token $i$, we look at all preceding tokens $j \leq i$
    and take a **weighted sum**, where the weights reflect relevance:
    $$\mathbf{a}_i = \sum_{j \leq i} \alpha_{ij}\,\mathbf{v}_j$$

    The weights $\alpha_{ij}$ come from **queries** and **keys**:
    $$\alpha_{ij} = \frac{\exp\!\left(\mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k}\right)}
                         {\sum_{m \leq i} \exp\!\left(\mathbf{q}_i \cdot \mathbf{k}_m / \sqrt{d_k}\right)}$$

    where $\mathbf{q}_i = \mathbf{x}_i\,\mathbf{W}^Q$, $\mathbf{k}_j = \mathbf{x}_j\,\mathbf{W}^K$,
    $\mathbf{v}_j = \mathbf{x}_j\,\mathbf{W}^V$ are **learned linear projections**.

    **Why $\sqrt{d_k}$?** The raw dot product $\mathbf{q} \cdot \mathbf{k}$ has variance $d_k$
    (sum of $d_k$ independent unit-variance terms). For large $d_k$, the scores become large
    in magnitude, pushing softmax into saturation where gradients vanish.
    Dividing by $\sqrt{d_k}$ restores unit variance.

    Use the slider to see how the temperature (analogous to $1/\sqrt{d_k}$) affects the attention distribution.
    """)
    return


@app.cell
def temp_slider(mo):
    temp_s = mo.ui.slider(
        start=0.1, stop=5.0, value=1.0, step=0.1,
        label=r"Temperature $\tau$ (lower = sharper attention)",
    )
    temp_s
    return (temp_s,)


@app.cell
def attention_viz(temp_s, np, plt):
    def _():
        _rng = np.random.default_rng(7)
        _tau = temp_s.value
        _N, _dk = 8, 16

        _tokens = ["The", "cat", "sat", "on", "the", "warm", "soft", "mat"]
        _X = _rng.standard_normal((_N, _dk))
        _Wq = _rng.normal(0, 1 / np.sqrt(_dk), (_dk, _dk))
        _Wk = _rng.normal(0, 1 / np.sqrt(_dk), (_dk, _dk))

        _Q = _X @ _Wq
        _K = _X @ _Wk
        _scores = _Q @ _K.T / _tau

        _mask = np.triu(np.full((_N, _N), -1e9), k=1)
        _scores_masked = _scores + _mask
        _exp = np.exp(_scores_masked - _scores_masked.max(axis=-1, keepdims=True))
        _attn = _exp / _exp.sum(axis=-1, keepdims=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        _raw = _Q @ _K.T
        _im = ax.imshow(_raw, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(_N))
        ax.set_xticklabels(_tokens, rotation=45, fontsize=8)
        ax.set_yticks(range(_N))
        ax.set_yticklabels(_tokens, fontsize=8)
        ax.set_title("Raw scores $\\mathbf{Q}\\mathbf{K}^\\top$\n(before scaling & masking)")
        plt.colorbar(_im, ax=ax, shrink=0.8)

        ax = axes[1]
        _im2 = ax.imshow(_attn, cmap="Blues", aspect="equal", vmin=0, vmax=1)
        ax.set_xticks(range(_N))
        ax.set_xticklabels(_tokens, rotation=45, fontsize=8)
        ax.set_yticks(range(_N))
        ax.set_yticklabels(_tokens, fontsize=8)
        ax.set_title(f"Attention weights ($\\tau = {_tau:.1f}$)\n(after causal mask + softmax)")
        plt.colorbar(_im2, ax=ax, shrink=0.8)

        ax = axes[2]
        _query_idx = _N - 1
        _weights = _attn[_query_idx, :_query_idx + 1]
        _bars = ax.bar(range(_query_idx + 1), _weights, color="#3498db", edgecolor="k", linewidth=0.5)
        for _b, _w in zip(_bars, _weights):
            if _w > 0.05:
                ax.text(_b.get_x() + _b.get_width() / 2, _w + 0.01,
                        f"{_w:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(_query_idx + 1))
        ax.set_xticklabels(_tokens[:_query_idx + 1], rotation=45, fontsize=8)
        _sharpness = "sharp (confident)" if _tau < 0.5 else ("uniform (uncertain)" if _tau > 3 else "moderate")
        ax.set(ylabel=r"$\alpha_{ij}$",
               title=f'Query = "{_tokens[_query_idx]}" → {_sharpness}')
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 4. The Causal Mask
# ---------------------------------------------------------------------------


@app.cell
def mask_header(mo):
    mo.md(r"""
    ---
    ## 4. The Causal Mask

    In matrix form, all $N^2$ pairwise scores are computed at once:
    $$\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \in \mathbb{R}^{N \times N}$$

    The entry $S_{ij}$ is the score from position $i$ to position $j$.
    But during language modelling, token $i$ must not attend to future tokens $j > i$
    (they haven't been generated yet). We enforce causality by adding a **mask**:

    $$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

    Since $e^{-\infty} = 0$, the softmax zeroes out all future positions.
    The resulting attention weight matrix is **lower-triangular**.

    $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
      = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

    **Cost:** The product $\mathbf{Q}\mathbf{K}^\top$ takes $O(N^2 d_k)$.
    Attention is therefore **quadratic** in sequence length.
    """)
    return


@app.cell
def mask_viz(np, plt):
    def _():
        _N = 8
        _tokens = ["The", "cat", "sat", "on", "the", "warm", "soft", "mat"]

        _mask = np.zeros((_N, _N))
        for _i in range(_N):
            for _j in range(_N):
                if _j > _i:
                    _mask[_i, _j] = -np.inf

        _mask_display = np.zeros((_N, _N))
        for _i in range(_N):
            for _j in range(_N):
                _mask_display[_i, _j] = 1.0 if _j <= _i else 0.0

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        ax = axes[0]
        _rng = np.random.default_rng(3)
        _scores = _rng.normal(0, 1, (_N, _N))
        _im = ax.imshow(_scores, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(_N))
        ax.set_xticklabels(_tokens, rotation=45, fontsize=7)
        ax.set_yticks(range(_N))
        ax.set_yticklabels(_tokens, fontsize=7)
        ax.set_title("Score matrix $\\mathbf{S}$\n(all pairs)")
        plt.colorbar(_im, ax=ax, shrink=0.8)

        ax = axes[1]
        _cmap = plt.cm.colors.ListedColormap(["#e74c3c", "#2ecc71"])
        _im2 = ax.imshow(_mask_display, cmap=_cmap, aspect="equal", vmin=0, vmax=1)
        for _i in range(_N):
            for _j in range(_N):
                _txt = "0" if _j <= _i else "$-\\infty$"
                ax.text(_j, _i, _txt, ha="center", va="center", fontsize=7,
                        color="white" if _j > _i else "black")
        ax.set_xticks(range(_N))
        ax.set_xticklabels(_tokens, rotation=45, fontsize=7)
        ax.set_yticks(range(_N))
        ax.set_yticklabels(_tokens, fontsize=7)
        ax.set_title("Causal mask $\\mathbf{M}$\n(green = attend, red = block)")

        ax = axes[2]
        _masked = _scores + _mask
        _exp = np.exp(_masked - np.nanmax(_masked, axis=-1, keepdims=True))
        _exp = np.where(np.isneginf(_masked), 0, _exp)
        _attn = _exp / _exp.sum(axis=-1, keepdims=True)
        _im3 = ax.imshow(_attn, cmap="Blues", aspect="equal", vmin=0)
        ax.set_xticks(range(_N))
        ax.set_xticklabels(_tokens, rotation=45, fontsize=7)
        ax.set_yticks(range(_N))
        ax.set_yticklabels(_tokens, fontsize=7)
        ax.set_title("After mask + softmax\n(lower-triangular)")
        plt.colorbar(_im3, ax=ax, shrink=0.8)

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 5. Multi-Head Attention
# ---------------------------------------------------------------------------


@app.cell
def mha_header(mo):
    mo.md(r"""
    ---
    ## 5. Multi-Head Attention

    A single set of projections $(\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V)$ learns one type of
    attention pattern. But tokens relate to each other in many ways: syntax, coreference, semantics.

    **Multi-head attention** uses $A$ independent heads, each with its own projections.
    Setting $d_k = d_v = d/A$, each head operates in a $d/A$-dimensional subspace:

    $$\text{head}^c = \text{Attention}(\mathbf{X}\mathbf{W}^{Qc},\;
                                         \mathbf{X}\mathbf{W}^{Kc},\;
                                         \mathbf{X}\mathbf{W}^{Vc})
      \in \mathbb{R}^{N \times d/A}$$

    The heads are concatenated and projected:
    $$\text{MultiHead}(\mathbf{X}) = [\text{head}^1 \| \cdots \| \text{head}^A]\,\mathbf{W}^O$$

    **Total cost:** Same as a single full-dimension head ($4d^2$ parameters),
    but strictly more expressive because each head can specialize.

    Use the slider to see how different heads learn different attention patterns.
    """)
    return


@app.cell
def nheads_slider(mo):
    nheads_s = mo.ui.slider(
        start=1, stop=8, value=4, step=1,
        label="Number of attention heads $A$",
    )
    nheads_s
    return (nheads_s,)


@app.cell
def mha_viz(nheads_s, np, plt):
    def _():
        _A = nheads_s.value
        _rng = np.random.default_rng(12)
        _N, _d = 8, 32
        _dk = _d // _A
        _tokens = ["The", "cat", "sat", "on", "the", "warm", "soft", "mat"]

        _X = _rng.standard_normal((_N, _d))

        _heads_attn = []
        for _c in range(_A):
            _Wq = _rng.normal(0, 1 / np.sqrt(_dk), (_d, _dk))
            _Wk = _rng.normal(0, 1 / np.sqrt(_dk), (_d, _dk))
            _Q = _X @ _Wq
            _K = _X @ _Wk
            _scores = _Q @ _K.T / np.sqrt(_dk)
            _mask = np.triu(np.full((_N, _N), -1e9), k=1)
            _scores += _mask
            _exp = np.exp(_scores - _scores.max(axis=-1, keepdims=True))
            _attn = _exp / _exp.sum(axis=-1, keepdims=True)
            _heads_attn.append(_attn)

        _ncols = min(_A, 4)
        _nrows = (_A + _ncols - 1) // _ncols
        fig, axes = plt.subplots(_nrows, _ncols, figsize=(3.5 * _ncols, 3.5 * _nrows),
                                 squeeze=False)

        _head_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]
        for _c in range(_A):
            _r, _col = divmod(_c, _ncols)
            ax = axes[_r][_col]
            _im = ax.imshow(_heads_attn[_c], cmap="Blues", aspect="equal", vmin=0, vmax=1)
            ax.set_xticks(range(_N))
            ax.set_xticklabels(_tokens, rotation=45, fontsize=7)
            ax.set_yticks(range(_N))
            ax.set_yticklabels(_tokens, fontsize=7)
            ax.set_title(f"Head {_c + 1} ($d_k = {_dk}$)",
                         color=_head_colors[_c % len(_head_colors)], fontweight="bold")

        for _c in range(_A, _nrows * _ncols):
            _r, _col = divmod(_c, _ncols)
            axes[_r][_col].axis("off")

        plt.suptitle(f"$A = {_A}$ heads: each learns a different attention pattern\n"
                     f"(total dim $d = {_d}$, per-head $d_k = {_dk}$)",
                     fontsize=11, y=1.02)
        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 6. Add & Norm + FFN
# ---------------------------------------------------------------------------


@app.cell
def addnorm_header(mo):
    mo.md(r"""
    ---
    ## 6. Add & Norm + Feed-Forward

    After attention, two more components complete the transformer block.

    **Residual connection:** The sub-layer output is *added* to its input:
    $$\mathbf{x}' = f(\mathbf{x}) + \mathbf{x}$$

    This creates a gradient highway: $\frac{\partial \mathbf{x}'}{\partial \mathbf{x}} = \mathbf{I} + \frac{\partial f}{\partial \mathbf{x}}$,
    preventing vanishing gradients in deep stacks.

    **Layer normalization** re-centers and re-scales each token's vector independently:
    $$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma + \epsilon} + \beta,
      \qquad \mu = \frac{1}{d}\sum_j x_j$$

    **Feed-forward network** (applied per-token, independently):
    $$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x}\,\mathbf{W}_1 + \mathbf{b}_1)\,\mathbf{W}_2 + \mathbf{b}_2$$

    with $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$ and $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$.
    This is the only component that introduces nonlinearity and capacity *within* each position.
    Attention mixes information *between* positions; the FFN processes it *within* each position.

    **Pre-norm** (modern standard): LayerNorm *before* sub-layer, then add.
    One full block in pre-norm convention:
    $$\mathbf{O} = \mathbf{X} + \text{MHA}(\text{LN}(\mathbf{X}))$$
    $$\mathbf{H} = \mathbf{O} + \text{FFN}(\text{LN}(\mathbf{O}))$$
    """)
    return


@app.cell
def block_viz(np, plt):
    def _():
        _rng = np.random.default_rng(5)
        _d = 64

        _x = _rng.standard_normal(_d)

        def _layer_norm(x):
            return (x - x.mean()) / (x.std() + 1e-8)

        _x_normed = _layer_norm(_x)

        _ffn_W1 = _rng.normal(0, np.sqrt(2 / _d), (4 * _d, _d))
        _ffn_b1 = np.zeros(4 * _d)
        _ffn_W2 = _rng.normal(0, np.sqrt(2 / (4 * _d)), (_d, 4 * _d))
        _ffn_b2 = np.zeros(_d)

        _h = np.maximum(0, _ffn_W1 @ _x_normed + _ffn_b1)
        _ffn_out = _ffn_W2 @ _h + _ffn_b2
        _x_out = _x + _ffn_out

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        ax = axes[0]
        ax.bar(range(_d), _x, color="#3498db", alpha=0.5, label="Before LN", width=1.0)
        ax.bar(range(_d), _x_normed, color="#e74c3c", alpha=0.5, label="After LN", width=1.0)
        ax.set(xlabel="Dimension", ylabel="Value",
               title="LayerNorm: zero mean, unit variance")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1]
        ax.bar(range(_d), _ffn_out, color="#2ecc71", alpha=0.7, width=1.0, label="FFN correction $f(\\mathbf{x})$")
        ax.set(xlabel="Dimension", ylabel="Value",
               title="FFN learns a per-position correction")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[2]
        ax.bar(range(_d), _x, color="#3498db", alpha=0.4, width=1.0, label="Input $\\mathbf{x}$")
        ax.bar(range(_d), _x_out, color="#9b59b6", alpha=0.4, width=1.0, label="Output $\\mathbf{x} + f(\\mathbf{x})$")
        ax.set(xlabel="Dimension", ylabel="Value",
               title="Residual connection: input + correction")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 7. Generation — Sampling Strategies
# ---------------------------------------------------------------------------


@app.cell
def gen_header(mo):
    mo.md(r"""
    ---
    ## 7. Generation — Sampling Strategies

    Given a prompt $w_{1:N}$, we generate text by repeated next-token prediction.
    At each step the model outputs a probability distribution $\mathbf{p} \in [0,1]^V$.
    The **sampling strategy** determines how we pick $w_{t+1}$:

    **Greedy:** Always take $w_{t+1} = \arg\max_k p_k$. Deterministic but repetitive.

    **Top-$k$:** Zero out all but the $k$ highest-probability tokens, renormalize, and sample.
    When $k = 1$ this is greedy; when $k = V$ this is unrestricted sampling.

    **Top-$p$ (nucleus):** Retain the smallest set $\mathcal{S}$ such that $\sum_{w \in \mathcal{S}} p(w) \geq p$,
    renormalize, and sample. This adapts the number of candidates to the distribution shape:
    peaked distributions keep few tokens, flat ones keep many.

    Use the slider to see how top-$k$ filters the distribution.
    """)
    return


@app.cell
def topk_slider(mo):
    topk_s = mo.ui.slider(
        start=1, stop=30, value=5, step=1,
        label="Top-$k$ value",
    )
    topk_s
    return (topk_s,)


@app.cell
def gen_viz(topk_s, np, plt):
    def _():
        _k = topk_s.value
        _rng = np.random.default_rng(42)
        _V = 30
        _logits = _rng.normal(0, 2, _V)
        _logits[3] += 4
        _logits[7] += 3
        _logits[15] += 2.5

        _exp = np.exp(_logits - _logits.max())
        _probs = _exp / _exp.sum()
        _sorted_idx = np.argsort(-_probs)
        _probs_sorted = _probs[_sorted_idx]

        _topk_mask = np.zeros(_V, dtype=bool)
        _topk_mask[_sorted_idx[:_k]] = True
        _probs_topk = np.where(_topk_mask, _probs, 0)
        _probs_topk = _probs_topk / _probs_topk.sum()

        _cumsum = np.cumsum(_probs_sorted)
        _nucleus_p = 0.9
        _n_nucleus = np.searchsorted(_cumsum, _nucleus_p) + 1
        _topp_mask = np.zeros(_V, dtype=bool)
        _topp_mask[_sorted_idx[:_n_nucleus]] = True
        _probs_topp = np.where(_topp_mask, _probs, 0)
        _probs_topp = _probs_topp / _probs_topp.sum()

        _tok_labels = [f"t{i}" for i in range(_V)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        _colors_full = ["#3498db" if _topk_mask[i] else "#bdc3c7" for i in range(_V)]
        ax.bar(range(_V), _probs, color=_colors_full, edgecolor="k", linewidth=0.3)
        ax.set(xlabel="Token", ylabel="Probability",
               title=f"Full distribution\n(blue = top-{_k} tokens)")
        ax.set_xticks(range(0, _V, 5))
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1]
        ax.bar(range(_V), _probs_topk, color="#3498db", edgecolor="k", linewidth=0.3)
        ax.set(xlabel="Token", ylabel="Probability (renormalized)",
               title=f"After top-{_k}: only {_k} candidates remain")
        ax.set_xticks(range(0, _V, 5))
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[2]
        ax.plot(range(1, _V + 1), _cumsum, "#e74c3c", lw=2.5, marker="o", markersize=3)
        ax.axhline(_nucleus_p, color="gray", ls="--", lw=1.5,
                   label=f"$p = {_nucleus_p}$ threshold")
        ax.axvline(_n_nucleus, color="#2ecc71", ls="--", lw=2,
                   label=f"Nucleus keeps {_n_nucleus} tokens")
        ax.axvline(_k, color="#3498db", ls=":", lw=2,
                   label=f"Top-$k$ keeps {_k} tokens")
        ax.scatter([_n_nucleus], [_cumsum[_n_nucleus - 1]], s=100, c="#2ecc71", zorder=5)
        ax.set(xlabel="Tokens (sorted by probability)", ylabel="Cumulative probability",
               title=f"Top-$p$ ({_nucleus_p}) vs Top-$k$ ({_k})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


# ---------------------------------------------------------------------------
# 8. Scaling Laws
# ---------------------------------------------------------------------------


@app.cell
def scaling_header(mo):
    mo.md(r"""
    ---
    ## 8. Scaling Laws

    Each transformer block has $12d^2$ non-embedding parameters (with $d_{\text{ff}} = 4d$).
    For $L$ blocks: $N_{\text{params}} \approx 12Ld^2$.

    Kaplan et al. (2020) showed that loss follows a **power law** in model size $N$,
    dataset size $D$, and compute $C$:
    $$\mathcal{L}(N) \propto N^{-\alpha_N}, \qquad
      \mathcal{L}(D) \propto D^{-\alpha_D}, \qquad
      \mathcal{L}(C) \propto C^{-\alpha_C}$$

    These **scaling laws** allow us to predict the performance of larger models
    from small-scale experiments. Doubling parameters gives a predictable loss reduction,
    regardless of the specific architecture details.

    **Chinchilla scaling (Hoffmann et al., 2022):** For a fixed compute budget,
    parameters and data should be scaled equally: $N \propto D$.
    This contradicts earlier practice of making models very large with relatively less data.
    """)
    return


@app.cell
def scaling_viz(np, plt):
    def _():
        _N = np.logspace(6, 11, 100)
        _alpha = 0.076
        _A = 8.0

        _loss = _A * _N ** (-_alpha)

        _models = {
            "GPT-2 Small": (117e6, 3.2),
            "GPT-2 Medium": (345e6, 2.9),
            "GPT-2 Large": (774e6, 2.7),
            "GPT-3 (175B)": (175e9, 2.0),
        }

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.loglog(_N, _loss, "#3498db", lw=2.5, label=r"$\mathcal{L}(N) \propto N^{-0.076}$")
        for _name, (_size, _l) in _models.items():
            ax.scatter([_size], [_l], s=80, zorder=5, label=_name)
        ax.set(xlabel="Parameters $N$", ylabel="Test Loss $\\mathcal{L}$",
               title="Scaling law: loss decreases as power law in $N$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

        ax = axes[1]
        _d_vals = np.array([512, 768, 1024, 2048, 4096, 8192, 12288])
        _L_vals = np.array([6, 12, 24, 24, 32, 48, 96])
        _A_vals = np.array([8, 12, 16, 16, 32, 64, 96])
        _params = 12 * _L_vals * _d_vals**2

        _labels = ["125M", "350M", "760M", "1.3B", "6.7B", "30B", "175B"]

        ax.bar(range(len(_d_vals)), _params / 1e9, color="#2ecc71", edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(len(_d_vals)))
        ax.set_xticklabels(_labels, fontsize=8)
        ax.set(xlabel="Model", ylabel="Non-embedding params (billions)",
               title="$12Ld^2$ parameter formula\n(GPT-family sizes)")
        ax.grid(True, alpha=0.3, axis="y")

        for _i, (_di, _Li) in enumerate(zip(_d_vals, _L_vals)):
            ax.text(_i, _params[_i] / 1e9 + 2,
                    f"$d$={_di}\n$L$={_Li}", ha="center", fontsize=7)

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

    | Component | Operation | Shape |
    |-----------|----------|-------|
    | **Embedding** | $\mathbf{e}_t = \mathbf{E}[w_t]$ | $[1 \times d]$ |
    | **Pos. encoding** | $\mathbf{x}_t = \mathbf{e}_t + \mathbf{P}[t]$ | $[1 \times d]$ |
    | **QKV projections** | $Q = XW^Q$, etc. | $[N \times d_k]$ |
    | **Attention** | $\text{softmax}(QK^\top/\sqrt{d_k} + M)\,V$ | $[N \times d_v]$ |
    | **Multi-head** | Concat $A$ heads, project by $W^O$ | $[N \times d]$ |
    | **Add & Norm** | $x + f(\text{LN}(x))$ | $[N \times d]$ |
    | **FFN** | $\text{ReLU}(xW_1 + b_1)W_2 + b_2$ | $[N \times d]$ |
    | **Output head** | $\text{softmax}(h_N \cdot E^\top)$ | $[1 \times V]$ |
    | **Params/block** | $12d^2$ | scalar |

    **The complete forward pass:**

    1. Embed + add positional encoding → $\mathbf{X}^{(0)}$
    2. For $\ell = 1, \ldots, L$: MHA + Add&Norm + FFN + Add&Norm → $\mathbf{X}^{(\ell)}$
    3. Final LayerNorm → linear ($\times E^\top$) → softmax → $\mathbf{p} \in [0,1]^V$

    The transformer replaces recurrence with **parallelizable attention**
    at the cost of $O(N^2)$ per layer. Scaling laws make performance predictable.
    """)
    return


if __name__ == "__main__":
    app.run()
