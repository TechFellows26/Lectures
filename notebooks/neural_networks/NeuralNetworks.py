# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "seaborn",
#     "scikit-learn",
#     "Pillow",
# ]
# ///
"""
Neural Networks on Coriander vs Parsley Images

Dataset: https://github.com/alilakrakbi/Coriander-vs-Parsley
Task: Binary image classification with sklearn MLPClassifier
"""

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def title():
    import marimo as mo

    mo.md(
        r"""
        # Neural Networks — Coriander vs Parsley

        **Dataset:** [Coriander-vs-Parsley](https://github.com/alilakrakbi/Coriander-vs-Parsley)
        **Task:** Binary image classification — can a neural network learn to tell herbs apart?
        **Why this dataset?** Images are visual, compact, and the task is clearly non-linear —
        you cannot separate coriander from parsley with a single hyperplane over raw pixels.
        This makes it a perfect testbed for going from a linear perceptron to a multi-layer network.

        Pipeline:

        > Load Images → EDA → Feature Extraction → Train/Test Split →
        > Perceptron Baseline → 1-Layer MLP → Deeper Networks →
        > Regularization → Evaluation → Weight Visualization
        """
    )
    return (mo,)


@app.cell
def imports():
    import warnings
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from PIL import Image
    from sklearn.linear_model import Perceptron
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        RocCurveDisplay,
        accuracy_score,
        classification_report,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_score,
        train_test_split,
        learning_curve,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 10,
        }
    )
    sns.set_style("whitegrid")
    return (
        ConfusionMatrixDisplay,
        Image,
        MLPClassifier,
        Path,
        Perceptron,
        RocCurveDisplay,
        StratifiedKFold,
        accuracy_score,
        classification_report,
        cross_val_score,
        learning_curve,
        np,
        pd,
        plt,
        roc_auc_score,
        sns,
        train_test_split,
        StandardScaler,
    )


@app.cell
def config(Path):
    DATA_ROOT = Path("data/neural_networks")
    return (DATA_ROOT,)


@app.cell
def load_header(mo):
    mo.md(r"""
    ---
    ## Part I — Load Data

    Images in `data/neural_networks/{coriander,parsley}/`. See `data/neural_networks/README.md` for source.
    We load every image path and record its label (0 = coriander, 1 = parsley).
    """)
    return


@app.cell
def load_data(DATA_ROOT, Image, Path, np):
    _class_dirs = {"coriander": 0, "parsley": 1}

    _paths = []
    _labels = []

    for _cls, _lbl in _class_dirs.items():
        _dir = DATA_ROOT / _cls
        if not _dir.exists():
            print(f"WARNING: {_dir} not found — run the git clone command above")
        else:
            for _p in sorted(_dir.glob("*.jpg")) + sorted(_dir.glob("*.jpeg")) + sorted(_dir.glob("*.png")):
                _paths.append(_p)
                _labels.append(_lbl)

    paths = _paths
    labels = np.array(_labels)

    print(f"Total images: {len(paths)}")
    print(f"  Coriander (0): {(labels == 0).sum()}")
    print(f"  Parsley   (1): {(labels == 1).sum()}")
    return labels, paths


@app.cell
def eda_header(mo):
    mo.md(r"""
    ---
    ## Part II — Exploratory Data Analysis

    Before any modeling, understand what you're working with:

    1. **Class balance** — equal? Imbalance affects which metric matters most
    2. **Sample images** — sanity check; see what the network will learn from
    3. **Mean image per class** — do classes look visually distinct on average?
    4. **Pixel intensity distribution** — uniform? Bimodal? This informs normalization

    The key question: *could a human tell the difference?* If yes, the task is learnable.
    If even a human struggles, the model will likely struggle too.
    """)
    return


@app.cell
def eda_samples(Image, labels, np, paths, plt):
    def _():
        _class_names = ["Coriander", "Parsley"]
        _colors = ["#3498db", "#2ecc71"]
        _rng = np.random.default_rng(0)

        fig, axes = plt.subplots(2, 8, figsize=(16, 5))
        for _cls in range(2):
            _cls_paths = [p for p, l in zip(paths, labels) if l == _cls]
            _sample_idx = _rng.choice(len(_cls_paths), size=min(8, len(_cls_paths)), replace=False)
            for _j, _idx in enumerate(_sample_idx):
                ax = axes[_cls, _j]
                _img = Image.open(_cls_paths[_idx]).convert("RGB")
                ax.imshow(_img)
                ax.axis("off")
                if _j == 0:
                    ax.set_title(_class_names[_cls], color=_colors[_cls], fontsize=11, fontweight="bold", pad=4)

        plt.suptitle("Sample images per class", fontsize=13, y=1.01)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def eda_mean_image(Image, labels, np, paths, plt):
    def _():
        _IMG_SIZE = 64
        _class_names = ["Coriander", "Parsley"]
        _colors = ["#3498db", "#2ecc71"]

        _mean_imgs = []
        _all_pixels = []
        for _cls in range(2):
            _cls_paths = [p for p, l in zip(paths, labels) if l == _cls]
            _stack = []
            for _p in _cls_paths[:200]:
                _img = np.array(Image.open(_p).convert("RGB").resize((_IMG_SIZE, _IMG_SIZE)), dtype=float)
                _stack.append(_img)
                _all_pixels.append(_img.ravel())
            _mean_imgs.append(np.stack(_stack).mean(axis=0).astype(np.uint8))

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))

        for _cls in range(2):
            axes[_cls].imshow(_mean_imgs[_cls])
            axes[_cls].set_title(f"Mean {_class_names[_cls]} image\n(average over samples)")
            axes[_cls].axis("off")

        ax = axes[2]
        for _cls, _color, _name in [(0, "#3498db", "Coriander"), (1, "#2ecc71", "Parsley")]:
            _cls_paths = [p for p, l in zip(paths, labels) if l == _cls]
            _px = []
            for _p in _cls_paths[:100]:
                _arr = np.array(Image.open(_p).convert("L").resize((_IMG_SIZE, _IMG_SIZE)), dtype=float)
                _px.extend(_arr.ravel().tolist())
            ax.hist(_px, bins=50, color=_color, alpha=0.5, label=_name, density=True)
        ax.set_xlabel("Pixel intensity (grayscale)")
        ax.set_ylabel("Density")
        ax.set_title("Pixel Intensity Distribution\n(grayscale, 100 images/class)")
        ax.legend()

        ax = axes[3]
        _counts = [(labels == 0).sum(), (labels == 1).sum()]
        ax.bar(_class_names, _counts, color=["#3498db", "#2ecc71"], edgecolor="k", linewidth=0.5)
        ax.set_title("Class Balance")
        ax.set_ylabel("Number of Images")
        for _i, _v in enumerate(_counts):
            ax.text(_i, _v + 0.5, str(_v), ha="center", fontsize=11)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def features_header(mo):
    mo.md(r"""
    ---
    ## Part III — Feature Extraction

    **Raw pixels as features.** We resize every image to 64 × 64 grayscale,
    then flatten to a vector of $64 \times 64 = 4096$ values.

    **Why RGB?** Both herbs are green leaves — shape and texture are similar, but
    color distribution differs (coriander tends darker, parsley brighter). Dropping
    color to grayscale loses one of the strongest discriminating signals.

    **Why 64×64?** Large enough to preserve texture, small enough to train quickly.
    With 3 channels that gives $64 \times 64 \times 3 = 12\,288$ features per image.

    **Normalization:** We subtract the mean and divide by the standard deviation per feature
    (sklearn `StandardScaler`). This is critical for MLPs — gradient descent converges much
    faster when features are on the same scale, and it prevents any single pixel from dominating.
    """)
    return


@app.cell
def extract_features(Image, StandardScaler, labels, np, paths, train_test_split):
    _IMG_SIZE = 64

    def _load_image(p):
        return np.array(
            Image.open(p).convert("RGB").resize((_IMG_SIZE, _IMG_SIZE)),
            dtype=float,
        ).ravel()

    X_raw = np.stack([_load_image(_p) for _p in paths])
    y = labels.copy()

    print(f"Feature matrix shape: {X_raw.shape}  ({X_raw.shape[0]} images × {X_raw.shape[1]} pixels)")
    print(f"Pixel value range: [{X_raw.min():.0f}, {X_raw.max():.0f}]")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"\nAfter train/test split (80/20, stratified):")
    print(f"  Train: {X_train.shape[0]} images")
    print(f"  Test:  {X_test.shape[0]} images")
    print(f"\nAfter StandardScaler:")
    print(f"  Train mean ≈ {X_train.mean():.4f}, std ≈ {X_train.std():.4f}")
    return X_test, X_test_raw, X_train, X_train_raw, scaler, y_test, y_train


@app.cell
def perceptron_header(mo):
    mo.md(r"""
    ---
    ## Part IV — Perceptron Baseline

    Before going to full neural networks, we start with the **perceptron** —
    the simplest possible linear classifier, and the direct ancestor of the MLP.

    The perceptron finds a weight vector $\mathbf{w}$ such that:
    $$\hat{y} = \text{sign}(\langle \mathbf{w}, \mathbf{x} \rangle + b)$$

    It updates only on mistakes: $\mathbf{w} \leftarrow \mathbf{w} + y\mathbf{x}$.

    **Key question:** Can a linear classifier tell coriander from parsley?
    If the classes are not linearly separable in pixel space, the perceptron
    will plateau — motivating the need for hidden layers.

    From the notes, the perceptron is guaranteed to converge in $T \leq 1/\gamma^2$
    mistakes *only if* the data is linearly separable. If not, it will oscillate.
    """)
    return


@app.cell
def perceptron_model(
    Perceptron,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    y_test,
    y_train,
):
    perceptron = Perceptron(max_iter=200, random_state=42, tol=1e-4)
    perceptron.fit(X_train, y_train)

    pred_perc = perceptron.predict(X_test)
    acc_perc = accuracy_score(y_test, pred_perc)

    print(f"Perceptron — Test Accuracy: {acc_perc:.4f}\n")
    print(classification_report(y_test, pred_perc, target_names=["coriander", "parsley"]))
    return (acc_perc, perceptron, pred_perc)


@app.cell
def mlp_1_header(mo):
    mo.md(r"""
    ---
    ## Part V — MLP with One Hidden Layer

    A **Multi-Layer Perceptron (MLP)** adds one or more hidden layers between input and output.
    Each hidden unit computes:
    $$h_j = \sigma\bigl(\mathbf{w}_j^\top \mathbf{x} + b_j\bigr)$$

    where $\sigma$ is a nonlinear activation function. The output is a linear combination
    of hidden units passed through a final sigmoid:
    $$\hat{p} = \sigma\bigl(\mathbf{v}^\top \mathbf{h} + c\bigr)$$

    **Training:** Minimize binary cross-entropy via backpropagation + Adam optimizer.
    Adam is an adaptive variant of SGD that adjusts the learning rate per parameter —
    it works well out of the box without tuning the step size carefully.

    **Architecture choice:** 256 hidden units. Enough expressiveness for 12k RGB features,
    small enough to train quickly on a CPU. The Universal Approximation Theorem tells
    us this single hidden layer *can* represent any decision boundary —
    whether it actually *learns* it depends on the optimizer and data.
    """)
    return


@app.cell
def mlp_1_model(
    MLPClassifier,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    y_test,
    y_train,
):
    mlp1 = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    mlp1.fit(X_train, y_train)

    pred_mlp1 = mlp1.predict(X_test)
    acc_mlp1 = accuracy_score(y_test, pred_mlp1)

    print(f"MLP (256,) — Test Accuracy: {acc_mlp1:.4f}")
    print(f"Epochs trained: {mlp1.n_iter_}  (early stopping after {mlp1.n_iter_no_change} no-improve epochs)")
    print(f"\n{classification_report(y_test, pred_mlp1, target_names=['coriander', 'parsley'])}")
    return (acc_mlp1, mlp1, pred_mlp1)


@app.cell
def mlp_1_learning_curve(mlp1, plt):
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        ax.plot(mlp1.loss_curve_, "#3498db", lw=2, label="Train loss")
        if mlp1.validation_scores_ is not None:
            _val_loss = [1 - v for v in mlp1.validation_scores_]
            ax.plot(_val_loss, "#e74c3c", lw=2, ls="--", label="Val (1 - accuracy)")
        ax.set(xlabel="Epoch", ylabel="Loss", title="MLP (128,) Training Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if mlp1.validation_scores_ is not None:
            ax.plot(mlp1.validation_scores_, "#2ecc71", lw=2)
            ax.set(xlabel="Epoch", ylabel="Validation Accuracy",
                   title="Validation Accuracy over Training\n(early stopping prevents overfitting)")
            ax.axhline(max(mlp1.validation_scores_), color="gray", ls=":", lw=1, alpha=0.7,
                       label=f"Best = {max(mlp1.validation_scores_):.3f}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def depth_header(mo):
    mo.md(r"""
    ---
    ## Part VI — Comparing Architectures (Depth & Width)

    Does adding more layers help? Depth increases the **expressiveness** of the network —
    each layer can build more abstract representations from the previous one.

    - **Shallow networks** may not capture complex texture patterns in images
    - **Deep networks** can hierarchically combine low-level edges into textures into shapes
    - **Overly deep networks** on small datasets may overfit or suffer from vanishing gradients

    We compare four architectures trained with the same hyperparameters.
    The learning curve (train vs validation accuracy) reveals overfitting.

    **Vanishing gradients in practice:** Watch whether deeper networks train *worse* —
    this is the phenomenon the notes cover theoretically: each extra sigmoid layer
    multiplies gradients by at most 0.25, eventually starving early layers of signal.
    ReLU activations (what we use) mitigate this but don't eliminate it entirely.
    """)
    return


@app.cell
def depth_comparison(
    MLPClassifier,
    StratifiedKFold,
    X_test,
    X_train,
    accuracy_score,
    cross_val_score,
    plt,
    y_test,
    y_train,
):
    def _():
        _architectures = {
            "Perceptron (linear)": (),
            "(128,)": (128,),
            "(256,)": (256,),
            "(256, 128)": (256, 128),
            "(256, 128, 64)": (256, 128, 64),
        }
        _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        _results = {}
        for _name, _layers in _architectures.items():
            if _layers == ():
                from sklearn.linear_model import Perceptron as _P
                _model = _P(max_iter=200, random_state=42)
            else:
                _model = MLPClassifier(
                    hidden_layer_sizes=_layers,
                    activation="relu",
                    solver="adam",
                    max_iter=300,
                    random_state=42,
                )
            _scores = cross_val_score(_model, X_train, y_train, cv=_cv, scoring="accuracy")
            _model.fit(X_train, y_train)
            _test_acc = accuracy_score(y_test, _model.predict(X_test))
            _results[_name] = {"cv_mean": _scores.mean(), "cv_std": _scores.std(), "test_acc": _test_acc}
            print(f"  {_name:>22}  CV: {_scores.mean():.3f}±{_scores.std():.3f}  Test: {_test_acc:.3f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        _names = list(_results.keys())
        _cv_means = [_results[n]["cv_mean"] for n in _names]
        _cv_stds = [_results[n]["cv_std"] for n in _names]
        _test_accs = [_results[n]["test_acc"] for n in _names]
        _x = range(len(_names))

        ax.bar([i - 0.2 for i in _x], _cv_means, width=0.35, color="#3498db", alpha=0.7,
               label="5-fold CV (train set)", yerr=_cv_stds, capsize=4, ecolor="k", linewidth=0.5)
        ax.bar([i + 0.2 for i in _x], _test_accs, width=0.35, color="#e74c3c", alpha=0.7,
               label="Test accuracy", linewidth=0.5)
        ax.set_xticks(list(_x))
        ax.set_xticklabels(_names, rotation=10, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title("Architecture Comparison — CV vs Test Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def reg_header(mo):
    mo.md(r"""
    ---
    ## Part VII — Regularization (Weight Decay)

    MLPs easily overfit — especially with 4096 inputs and a relatively small image dataset.
    The solution is **L2 regularization** (also called weight decay), controlled by `alpha` in sklearn:

    $$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}_{\text{CE}}(\theta) + \alpha \|\theta\|_2^2$$

    This is the same Ridge penalty from linear regression, applied to all network weights.
    It penalizes large weights, which forces the network to spread its "attention" more evenly
    across pixels rather than memorizing specific training images.

    **Practical interpretation:**
    - `alpha = 0`: no regularization — the network can memorize training data
    - `alpha = 0.001`: typical default — light regularization
    - `alpha = 0.1+`: heavy shrinkage — network may underfit

    We select `alpha` by 5-fold cross-validation on the training set.
    """)
    return


@app.cell
def reg_comparison(
    MLPClassifier,
    StratifiedKFold,
    X_train,
    cross_val_score,
    np,
    plt,
    y_train,
):
    def _():
        _alphas = np.logspace(-5, 1, 25)
        _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        _cv_scores = []

        for _a in _alphas:
            _model = MLPClassifier(
                hidden_layer_sizes=(256,),
                activation="relu",
                solver="adam",
                max_iter=300,
                alpha=_a,
                random_state=42,
            )
            _scores = cross_val_score(_model, X_train, y_train, cv=_cv, scoring="accuracy")
            _cv_scores.append((_scores.mean(), _scores.std()))

        _cv_means = [s[0] for s in _cv_scores]
        _cv_stds = [s[1] for s in _cv_scores]
        _best_idx = int(np.argmax(_cv_means))
        _best_alpha = _alphas[_best_idx]
        print(f"Best alpha: {_best_alpha:.5f}  (CV accuracy = {_cv_means[_best_idx]:.4f})")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.semilogx(_alphas, _cv_means, "#3498db", lw=2, marker="o", markersize=4)
        ax.fill_between(_alphas,
                        [m - s for m, s in zip(_cv_means, _cv_stds)],
                        [m + s for m, s in zip(_cv_means, _cv_stds)],
                        alpha=0.2, color="#3498db")
        ax.axvline(_best_alpha, color="#e74c3c", ls="--", lw=2,
                   label=f"Best α = {_best_alpha:.4f}")
        ax.scatter([_best_alpha], [_cv_means[_best_idx]], s=100, c="#e74c3c", zorder=5)
        ax.set(xlabel=r"$\alpha$ (L2 regularization strength)",
               ylabel="5-fold CV Accuracy",
               title="Regularization Path — MLP (256,)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def final_model_header(mo):
    mo.md(r"""
    ---
    ## Part VIII — Final Model & Evaluation

    We train the final model with the best architecture and regularization found above.
    The model is evaluated **once** on the held-out test set.

    **Why only once?** If we iterate on the test set — checking metrics, tweaking, checking again —
    we effectively "train" on it. The test accuracy would no longer be a true estimate of
    out-of-sample performance. This is a common source of overly optimistic published results.

    Metrics:

    | Metric | What it measures |
    |--------|-----------------|
    | **Accuracy** | Overall fraction correct |
    | **Precision** | Of predicted positives, how many are correct |
    | **Recall** | Of actual positives, how many did we find |
    | **F1** | Harmonic mean of precision and recall |
    | **AUC-ROC** | Area under the ROC curve — threshold-independent |
    """)
    return


@app.cell
def final_model(
    MLPClassifier,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    y_test,
    y_train,
):
    best_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        max_iter=600,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        learning_rate_init=0.001,
    )
    best_mlp.fit(X_train, y_train)

    pred_best = best_mlp.predict(X_test)
    acc_best = accuracy_score(y_test, pred_best)

    print(f"Final MLP (256, 128), α=0.0001 — Test Accuracy: {acc_best:.4f}")
    print(f"Epochs trained: {best_mlp.n_iter_}")
    print(f"\n{classification_report(y_test, pred_best, target_names=['coriander', 'parsley'])}")
    return (acc_best, best_mlp, pred_best)


@app.cell
def eval_viz(
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    X_test,
    X_train,
    acc_best,
    acc_mlp1,
    acc_perc,
    best_mlp,
    mlp1,
    perceptron,
    plt,
    roc_auc_score,
    y_test,
    y_train,
):
    def _():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        _models_cm = [
            ("Perceptron", perceptron, acc_perc, "#95a5a6"),
            ("MLP (256,)", mlp1, acc_mlp1, "#3498db"),
            ("MLP (256,128)", best_mlp, acc_best, "#2ecc71"),
        ]
        _names_bar = [m[0] for m in _models_cm]
        _accs_bar = [m[2] for m in _models_cm]
        _cols_bar = [m[3] for m in _models_cm]
        ax.bar(_names_bar, _accs_bar, color=_cols_bar, edgecolor="k", linewidth=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Model Comparison — Test Accuracy")
        for _i, _v in enumerate(_accs_bar):
            ax.text(_i, _v + 0.01, f"{_v:.3f}", ha="center", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1]
        ConfusionMatrixDisplay.from_estimator(
            best_mlp, X_test, y_test,
            display_labels=["coriander", "parsley"],
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title("Confusion Matrix — Final MLP (256, 128)")

        ax = axes[2]
        for _name, _m, _acc, _color in _models_cm:
            if hasattr(_m, "predict_proba"):
                _scores = _m.predict_proba(X_test)[:, 1]
            else:
                _scores = _m.decision_function(X_test)
            _auc = roc_auc_score(y_test, _scores)
            RocCurveDisplay.from_predictions(
                y_test, _scores, ax=ax, color=_color,
                name=f"{_name} (AUC={_auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_title("ROC Curves — Test Set")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.suptitle("Final Evaluation", fontsize=13, y=1.01)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def weights_header(mo):
    mo.md(r"""
    ---
    ## Part IX — Visualizing What the Network Learned

    The first hidden layer weights $W^{(1)} \in \mathbb{R}^{H \times d}$ can be reshaped
    back to $64 \times 64$ images — they show **what each hidden unit is "looking for"** in the input.

    This is only possible because the input is raw pixels (no other transformation).
    Each weight image is a **learned feature detector**: edges, blobs, color gradients
    that are useful for discriminating coriander from parsley.

    In deep networks, the first layer often learns Gabor-like edge detectors (similar to
    what neuroscientists found in V1, the primary visual cortex). With our small MLP on
    64×64 images, we'll see something cruder — but the principle is the same.

    **Comparison to CNNs:** Convolutional networks share weights spatially, so their first-layer
    filters are small (e.g. 3×3 or 7×7). Our fully-connected network has a separate weight
    for every pixel-channel-hidden-unit triple, making $64 \times 64 \times 3 \times 256 = 3\,145\,728$ weights
    in the first layer alone.
    """)
    return


@app.cell
def weight_viz(best_mlp, np, plt):
    def _():
        _W1 = best_mlp.coefs_[0]  # shape (12288, 256)
        _n_show = 32

        fig, axes = plt.subplots(4, 8, figsize=(14, 7))
        for _j, ax in enumerate(axes.flat):
            _w = _W1[:, _j].reshape(64, 64, 3)
            _w_norm = (_w - _w.min()) / (_w.max() - _w.min() + 1e-8)
            ax.imshow(_w_norm)
            ax.axis("off")
            ax.set_title(f"$h_{{{_j+1}}}$", fontsize=6, pad=1)

        plt.suptitle(
            "First-layer weight images (32 of 256 hidden units)\n"
            "Each shows what the hidden unit \"looks for\" in RGB",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def learning_curve_viz(MLPClassifier, X_train, learning_curve, np, plt, y_train):
    def _():
        _model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            max_iter=400,
            random_state=42,
        )
        _train_sizes, _train_scores, _val_scores = learning_curve(
            _model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=5, scoring="accuracy", n_jobs=-1,
        )

        _train_mean = _train_scores.mean(axis=1)
        _train_std = _train_scores.std(axis=1)
        _val_mean = _val_scores.mean(axis=1)
        _val_std = _val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(_train_sizes, _train_mean, "#3498db", lw=2, marker="o", label="Train accuracy")
        ax.fill_between(_train_sizes, _train_mean - _train_std, _train_mean + _train_std,
                        alpha=0.2, color="#3498db")
        ax.plot(_train_sizes, _val_mean, "#e74c3c", lw=2, marker="s", label="CV accuracy")
        ax.fill_between(_train_sizes, _val_mean - _val_std, _val_mean + _val_std,
                        alpha=0.2, color="#e74c3c")
        ax.set(xlabel="Training set size", ylabel="Accuracy",
               title="Learning Curve — MLP (256, 128)\n"
               "Gap between train/val = overfitting; converging curves = more data won't help much")
        ax.legend(fontsize=10)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def hard_examples(X_test, X_test_raw, best_mlp, np, plt, y_test):
    def _():
        _class_names = ["Coriander", "Parsley"]

        _proba_conf = best_mlp.predict_proba(X_test)
        _conf = np.abs(_proba_conf[:, 1] - 0.5)
        _hard_idx = np.argsort(_conf)[:8]

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for _j, _idx in enumerate(_hard_idx):
            ax = axes[_j // 4, _j % 4]
            _img = X_test_raw[_idx].reshape(64, 64, 3).astype(np.uint8)
            ax.imshow(_img)
            _true_label = _class_names[y_test[_idx]]
            _pred_prob = _proba_conf[_idx, 1]
            _pred_label = _class_names[int(_pred_prob >= 0.5)]
            _correct = _true_label == _pred_label
            _color = "#2ecc71" if _correct else "#e74c3c"
            ax.set_title(
                f"True: {_true_label}\nPred: {_pred_label} ({_pred_prob:.2f})",
                fontsize=8, color=_color,
            )
            ax.axis("off")

        plt.suptitle("Hardest Examples — Images Closest to the Decision Boundary\n"
                     "(green = correct, red = wrong)", fontsize=11, y=1.01)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def summary(mo):
    mo.md(r"""
    ---
    ## Summary

    | Step | What & Why |
    |------|------------|
    | **Data** | JPEG images of coriander and parsley, balanced classes |
    | **EDA** | Mean images show class-level visual differences; pixel distributions similar |
    | **Features** | 64×64 RGB → 12 288 raw pixel values (flattened) |
    | **Normalization** | StandardScaler: zero-mean, unit-variance — essential for gradient descent |
    | **Split** | 80/20 stratified train/test — only look at test set once |
    | **Perceptron** | Linear baseline — fails if classes not linearly separable in pixel space |
    | **MLP (256,)** | One hidden layer; ReLU activation; Adam optimizer; early stopping |
    | **Depth** | More layers → more expressiveness, but risk of overfitting on small datasets |
    | **Regularization** | L2 weight decay (α) chosen by 5-fold CV on training set |
    | **Evaluation** | Accuracy, F1, AUC-ROC, confusion matrix on held-out 20% |
    | **Weight viz** | First-layer weights as 64×64 images — learned feature detectors |
    | **Learning curve** | Reveals overfitting vs underfitting regime |

    **Key pedagogical takeaways:**
    - Raw pixels work, but require normalization and a nonlinear model
    - The perceptron plateau motivates the need for hidden layers (nonlinearity)
    - Early stopping and weight decay are practical tools against overfitting
    - Visualizing weights connects the math (backprop update equations) to what's actually learned
    - Real-world image tasks benefit from **convolutional** architectures that exploit spatial structure —
      our fully-connected MLP treats pixel $(i,j)$ and pixel $(i', j')$ as completely independent features
    """)
    return


if __name__ == "__main__":
    app.run()
