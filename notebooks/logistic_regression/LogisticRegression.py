# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "seaborn",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def title():
    import marimo as mo

    mo.md(
        r"""
        # Logistic Regression on Lebanese Newspaper Articles

        **Dataset:** ~1 000 articles scraped from Al Joumhouria and An-Nahar
        **Target:** `source` ∈ {joumhouria, nahar} — binary newspaper classification

        Pipeline:

        > Load & Clean $\to$ EDA $\to$ Text Preprocessing $\to$ TF-IDF $\to$
        > Train/Test Split $\to$ Logistic Regression $\to$ Regularization $\to$
        > Cross-Validation $\to$ Evaluation $\to$ Top Features
        """
    )
    return (mo,)


@app.cell
def imports():
    import warnings

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        RocCurveDisplay,
        accuracy_score,
        classification_report,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline

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
        LogisticRegression,
        LogisticRegressionCV,
        RocCurveDisplay,
        StratifiedKFold,
        TfidfVectorizer,
        accuracy_score,
        classification_report,
        cross_val_score,
        np,
        pd,
        plt,
        roc_auc_score,
        sns,
        train_test_split,
    )


@app.cell
def load_header(mo):
    mo.md(r"""
    ---
    ## Part I — Load and Clean

    Steps:
    1. Read the scraped CSV
    2. Drop rows with missing body text
    3. Drop duplicate articles
    4. Filter to minimum body length (removes photo/video stubs)
    5. Check class balance
    """)
    return


@app.cell
def load_data(pd):
    raw = pd.read_csv("data/logistic_regression/lebanese_newspapers.csv")
    print(f"Raw shape: {raw.shape}")
    print(f"Columns: {list(raw.columns)}")
    raw.head()
    return (raw,)


@app.cell
def clean_data(raw):
    df = raw.copy()

    df = df.dropna(subset=["body", "source"])
    df = df.drop_duplicates(subset=["body"])

    # Remove stubs — anything under 200 characters is unlikely to be a real article
    df = df[df["body"].str.len() >= 200].copy()

    df["label"] = (df["source"] == "joumhouria").astype(int)

    print(f"Clean shape: {df.shape}")
    print(f"\nClass balance:")
    print(df["source"].value_counts().to_string())
    print(f"\nLabel: joumhouria=1, nahar=0")
    df.head()
    return (df,)


@app.cell
def eda_header(mo):
    mo.md(r"""
    ---
    ## Part II — Exploratory Data Analysis

    1. **Class balance** — are the two newspapers equally represented?
    2. **Article length distribution** — body character count per source
    3. **Category distribution** — which sections are covered
    """)
    return


@app.cell
def eda_balance(df, plt, sns):
    def _():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        ax = axes[0]
        counts = df["source"].value_counts()
        ax.bar(counts.index, counts.values, color=["#3498db", "#e74c3c"], edgecolor="black", linewidth=0.5)
        ax.set_title("Class Balance")
        ax.set_ylabel("Article Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 1, str(v), ha="center", fontsize=10)

        ax = axes[1]
        df["body_len"] = df["body"].str.len()
        for src, color in [("joumhouria", "#3498db"), ("nahar", "#e74c3c")]:
            sns.kdeplot(
                df[df["source"] == src]["body_len"],
                ax=ax, label=src, color=color, fill=True, alpha=0.3,
            )
        ax.set_xlabel("Body length (characters)")
        ax.set_title("Article Length Distribution")
        ax.legend()
        ax.set_xlim(0, 8000)

        ax = axes[2]
        top_cats = df["category"].value_counts().head(10)
        ax.barh(top_cats.index[::-1], top_cats.values[::-1], color="#2ecc71", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Count")
        ax.set_title("Top 10 Categories")

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def preprocess_header(mo):
    mo.md(r"""
    ---
    ## Part III — Text Preprocessing

    Arabic text requires light normalization before TF-IDF:

    1. **Strip diacritics (tashkeel)** — vowel marks are not lexically meaningful
    2. **Normalize alef forms** — أ، إ، آ → ا
    3. **Normalize ta marbuta** — ة → ه
    4. **Remove non-Arabic characters** — punctuation, Latin, digits
    5. **Collapse whitespace**

    We do **not** stem or lemmatize — TF-IDF with sublinear TF already handles
    high-frequency terms, and newspaper style differences are often in surface form.
    """)
    return


@app.cell
def preprocess(df):
    import re as _re

    def _normalize_arabic(text: str) -> str:
        text = _re.sub(r"[\u0617-\u061A\u064B-\u065F]", "", text)
        text = _re.sub(r"[أإآا]", "ا", text)
        text = _re.sub(r"ة", "ه", text)
        text = _re.sub(r"[^\u0600-\u06FF\s]", " ", text)
        text = _re.sub(r"\s+", " ", text).strip()
        return text

    df_clean = df.copy()
    df_clean["body_clean"] = df_clean["body"].apply(_normalize_arabic)

    print("Sample cleaned article:")
    print(df_clean["body_clean"].iloc[0][:300])
    return (df_clean,)


@app.cell
def tfidf_header(mo):
    mo.md(r"""
    ---
    ## Part IV — TF-IDF Vectorization

    **Term Frequency–Inverse Document Frequency** converts raw text to a numeric
    feature matrix suitable for logistic regression.

    $$\text{tf-idf}(t, d) = \underbrace{\log(1 + \text{tf}(t, d))}_{\text{sublinear TF}} \times \underbrace{\log\frac{N}{\text{df}(t)}}_{\text{IDF}}$$

    - `max_features=15 000` — vocabulary cap
    - `sublinear_tf=True` — dampens very frequent terms within a document
    - `min_df=3` — ignore words appearing in fewer than 3 articles (likely noise)
    - `ngram_range=(1, 2)` — unigrams and bigrams

    The matrix $\mathbf{X}$ will be sparse: shape $(n_{\text{articles}},\, 15000)$.
    """)
    return


@app.cell
def split_and_vectorize(TfidfVectorizer, df_clean, np, train_test_split):
    X_text = df_clean["body_clean"].values
    y = df_clean["label"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=15_000,
        sublinear_tf=True,
        min_df=3,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"Train: {X_train.shape[0]:,} articles | Test: {X_test.shape[0]:,} articles")
    print(f"Vocabulary size: {X_train.shape[1]:,} features")
    print(f"Matrix density: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4%}")
    print(f"\nClass balance in train set:")
    _unique, _counts = np.unique(y_train, return_counts=True)
    for _u, _c in zip(_unique, _counts):
        print(f"  {'joumhouria' if _u == 1 else 'nahar':>12}: {_c}")
    return X_test, X_test_text, X_train, vectorizer, y_test, y_train


@app.cell
def lr_header(mo):
    mo.md(r"""
    ---
    ## Part V — Logistic Regression (Unregularized)

    From the notes, the logistic regression estimator solves:

    $$\hat{\boldsymbol{\beta}}_{\text{logit}} = \underset{\boldsymbol{\beta} \in \mathbb{R}^d}{\arg\min}\; \frac{1}{n} \sum_{i=1}^{n} \ell(\mathbf{X}_i^\top \boldsymbol{\beta},\, Y_i)$$

    where the **logistic loss** is:

    $$\ell(\hat{y}, y) = y \log(1 + e^{-\hat{y}}) + (1-y) \log(1 + e^{\hat{y}})$$

    This is equivalent to **maximizing the log-likelihood** under a Bernoulli model
    (see notes §4 — Maximum Likelihood Derivation). No closed-form solution exists;
    sklearn uses L-BFGS.

    There is **no regularization** here (achieved by setting `C` very large — sklearn's
    `LogisticRegression` uses `C = 1/λ`, so `C=1e6` ≈ unregularized).
    """)
    return


@app.cell
def lr_unregularized(
    LogisticRegression,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    y_test,
    y_train,
):
    lr_unregularized = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs", random_state=42)
    lr_unregularized.fit(X_train, y_train)

    pred_unrg = lr_unregularized.predict(X_test)
    acc_unrg = accuracy_score(y_test, pred_unrg)

    print(f"Unregularized Logistic Regression — Test Accuracy: {acc_unrg:.4f}\n")
    print(classification_report(y_test, pred_unrg, target_names=["nahar", "joumhouria"]))
    return


@app.cell
def sigmoid_viz(mo):
    mo.md(r"""
    ---
    ## Part V-B — The Sigmoid Function

    The model predicts:

    $$P(Y = 1 \mid \mathbf{x}) = \sigma(\boldsymbol{\theta}^\top \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\theta}^\top \mathbf{x}}}$$

    The **decision boundary** is the hyperplane $\boldsymbol{\theta}^\top \mathbf{x} = 0$,
    where $P = 0.5$. The signed distance from a point to this boundary is:

    $$\text{distance} = \frac{\beta_0 + \mathbf{x}^\top \boldsymbol{\beta}}{|\boldsymbol{\beta}|}$$

    Larger $|\boldsymbol{\beta}|$ → probabilities concentrate more sharply near 0 or 1.
    """)
    return


@app.cell
def sigmoid_plot(np, plt):
    def _():
        z = np.linspace(-8, 8, 300)
        sigma = 1 / (1 + np.exp(-z))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax.plot(z, sigma, color="#3498db", lw=2.5, label=r"$\sigma(z) = \frac{1}{1+e^{-z}}$")
        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
        ax.fill_between(z, sigma, 0.5, where=(z >= 0), alpha=0.15, color="#3498db", label="Predict joumhouria")
        ax.fill_between(z, sigma, 0.5, where=(z <= 0), alpha=0.15, color="#e74c3c", label="Predict nahar")
        ax.set_xlabel(r"$\boldsymbol{\theta}^\top \mathbf{x}$")
        ax.set_ylabel(r"$P(Y=1 \mid \mathbf{x})$")
        ax.set_title("Sigmoid — Decision Boundary at 0")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for scale, color, label in [(0.5, "#9b59b6", r"$|\boldsymbol{\beta}|$ small"), (1.0, "#3498db", "medium"), (3.0, "#e74c3c", r"large")]:
            ax.plot(z, 1 / (1 + np.exp(-scale * z)), color=color, lw=2, label=label)
        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel(r"$\boldsymbol{\theta}^\top \mathbf{x}$")
        ax.set_ylabel(r"$P(Y=1 \mid \mathbf{x})$")
        ax.set_title(r"Effect of $|\boldsymbol{\beta}|$ on confidence")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def reg_header(mo):
    mo.md(r"""
    ---
    ## Part VI — Regularization

    With 15 000 TF-IDF features and ~800 training articles, we have $d \gg n$ in spirit —
    many rare words will overfit. From the notes (§6 Regularization):

    **L2 (Ridge logistic regression):**
    $$\hat{\boldsymbol{\beta}}_{\text{ridge}} = \underset{\boldsymbol{\beta}}{\arg\min}\; \frac{1}{n}\sum_i \ell(\mathbf{X}_i^\top\boldsymbol{\beta}, Y_i) + \lambda\|\boldsymbol{\beta}\|_2^2$$

    Shrinks all coefficients — no feature elimination.

    **L1 (Lasso logistic regression):**
    $$\hat{\boldsymbol{\beta}}_{\text{lasso}} = \underset{\boldsymbol{\beta}}{\arg\min}\; \frac{1}{n}\sum_i \ell(\mathbf{X}_i^\top\boldsymbol{\beta}, Y_i) + \lambda\|\boldsymbol{\beta}\|_1$$

    Produces exact zeros — automatic feature selection. Especially useful here since
    most of the 15 000 vocabulary items are irrelevant to newspaper identity.

    `C = 1/λ` in sklearn. We select `C` via stratified 5-fold CV.
    """)
    return


@app.cell
def lr_l2(
    LogisticRegressionCV,
    StratifiedKFold,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    np,
    y_test,
    y_train,
):
    _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _Cs = np.logspace(-3, 3, 20)

    lr_l2 = LogisticRegressionCV(
        Cs=_Cs, cv=_cv, penalty="l2", solver="lbfgs",
        max_iter=1000, scoring="accuracy", random_state=42,
    )
    lr_l2.fit(X_train, y_train)

    pred_l2 = lr_l2.predict(X_test)
    acc_l2 = accuracy_score(y_test, pred_l2)

    print(f"L2 Logistic Regression — Best C: {lr_l2.C_[0]:.4f}  (λ = {1/lr_l2.C_[0]:.4f})")
    print(f"Test Accuracy: {acc_l2:.4f}\n")
    print(classification_report(y_test, pred_l2, target_names=["nahar", "joumhouria"]))
    return (lr_l2,)


@app.cell
def lr_l1(
    LogisticRegressionCV,
    StratifiedKFold,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    np,
    y_test,
    y_train,
):
    _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _Cs = np.logspace(-3, 3, 20)

    lr_l1 = LogisticRegressionCV(
        Cs=_Cs, cv=_cv, penalty="l1", solver="liblinear",
        max_iter=1000, scoring="accuracy", random_state=42,
    )
    lr_l1.fit(X_train, y_train)

    pred_l1 = lr_l1.predict(X_test)
    acc_l1 = accuracy_score(y_test, pred_l1)

    n_nonzero = (lr_l1.coef_[0] != 0).sum()
    print(f"L1 Logistic Regression — Best C: {lr_l1.C_[0]:.4f}  (λ = {1/lr_l1.C_[0]:.4f})")
    print(f"Non-zero coefficients: {n_nonzero} / {lr_l1.coef_.shape[1]}")
    print(f"Test Accuracy: {acc_l1:.4f}\n")
    print(classification_report(y_test, pred_l1, target_names=["nahar", "joumhouria"]))
    return (lr_l1,)


@app.cell
def reg_path_viz(StratifiedKFold, X_train, cross_val_score, np, plt, y_train):
    def _():
        from sklearn.linear_model import LogisticRegression as _LR

        _Cs = np.logspace(-3, 2, 40)
        _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        _cv_scores_l2, _cv_scores_l1 = [], []

        for _C in _Cs:
            _s_l2 = cross_val_score(_LR(C=_C, max_iter=500, solver="lbfgs"), X_train, y_train, cv=_cv)
            _s_l1 = cross_val_score(_LR(C=_C, max_iter=500, solver="liblinear", penalty="l1"), X_train, y_train, cv=_cv)
            _cv_scores_l2.append(_s_l2.mean())
            _cv_scores_l1.append(_s_l1.mean())

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(1 / _Cs, _cv_scores_l2, "b-o", markersize=4, lw=1.5, label="L2")
        ax.plot(1 / _Cs, _cv_scores_l1, "r-s", markersize=4, lw=1.5, label="L1")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda = 1/C$ (regularization strength)")
        ax.set_ylabel("5-fold CV Accuracy")
        ax.set_title("Regularization Path — CV Accuracy vs λ")
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
    ## Part VII — Cross-Validation Comparison

    Stratified 5-fold CV on the training set, comparing all three models.
    Stratified = each fold preserves the class ratio, important for classification.
    """)
    return


@app.cell
def cv_comparison(
    LogisticRegression,
    StratifiedKFold,
    X_train,
    cross_val_score,
    lr_l1,
    lr_l2,
    pd,
    plt,
    y_train,
):
    def _():
        _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        _models = {
            "Unregularized": LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs", random_state=42),
            "L2 (Ridge)": LogisticRegression(C=lr_l2.C_[0], max_iter=1000, solver="lbfgs", random_state=42),
            "L1 (Lasso)": LogisticRegression(C=lr_l1.C_[0], max_iter=1000, solver="liblinear", penalty="l1", random_state=42),
        }

        _results = {}
        for _name, _model in _models.items():
            _scores = cross_val_score(_model, X_train, y_train, cv=_cv, scoring="accuracy")
            _results[_name] = _scores

        _df = pd.DataFrame(_results)

        fig, ax = plt.subplots(figsize=(8, 5))
        _bp = ax.boxplot(
            [_df[c] for c in _df.columns],
            labels=_df.columns,
            patch_artist=True,
            medianprops=dict(color="red", linewidth=2),
        )
        _palette = ["#95a5a6", "#3498db", "#2ecc71"]
        for _patch, _color in zip(_bp["boxes"], _palette):
            _patch.set_facecolor(_color)
            _patch.set_alpha(0.6)

        ax.set_ylabel("Accuracy")
        ax.set_title("5-Fold Stratified CV Accuracy")
        ax.set_ylim(0, 1.05)

        print("5-Fold CV Accuracy (mean ± std):")
        for _col in _df.columns:
            print(f"  {_col:>18}: {_df[_col].mean():.4f} ± {_df[_col].std():.4f}")

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def eval_header(mo):
    mo.md(r"""
    ---
    ## Part VIII — Final Evaluation on Test Set

    | Metric | Definition |
    |--------|-----------|
    | **Accuracy** | $\frac{TP + TN}{n}$ |
    | **Precision** | $\frac{TP}{TP + FP}$ — of predicted positives, how many are right |
    | **Recall** | $\frac{TP}{TP + FN}$ — of actual positives, how many did we catch |
    | **F1** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |
    | **AUC-ROC** | Area under the receiver operating characteristic curve |
    """)
    return


@app.cell
def confusion_matrices(
    ConfusionMatrixDisplay,
    LogisticRegression,
    X_test,
    X_train,
    lr_l1,
    lr_l2,
    plt,
    y_test,
    y_train,
):
    def _():
        _models = {
            "Unregularized": LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs", random_state=42),
            "L2 (Ridge)": LogisticRegression(C=lr_l2.C_[0], max_iter=1000, solver="lbfgs", random_state=42),
            "L1 (Lasso)": LogisticRegression(C=lr_l1.C_[0], max_iter=1000, solver="liblinear", penalty="l1", random_state=42),
        }
        _colors = ["#95a5a6", "#3498db", "#2ecc71"]
        _labels = ["nahar", "joumhouria"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (_name, _model), _color in zip(axes, _models.items(), _colors):
            _model.fit(X_train, y_train)
            ConfusionMatrixDisplay.from_estimator(
                _model, X_test, y_test, display_labels=_labels,
                cmap="Blues", ax=ax, colorbar=False,
            )
            ax.set_title(_name)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def roc_curves(
    LogisticRegression,
    RocCurveDisplay,
    X_test,
    X_train,
    lr_l1,
    lr_l2,
    plt,
    roc_auc_score,
    y_test,
    y_train,
):
    def _():
        _models = {
            "Unregularized": LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs", random_state=42),
            "L2 (Ridge)": LogisticRegression(C=lr_l2.C_[0], max_iter=1000, solver="lbfgs", random_state=42),
            "L1 (Lasso)": LogisticRegression(C=lr_l1.C_[0], max_iter=1000, solver="liblinear", penalty="l1", random_state=42),
        }
        _colors = ["#95a5a6", "#3498db", "#2ecc71"]

        fig, ax = plt.subplots(figsize=(7, 6))
        for (_name, _model), _color in zip(_models.items(), _colors):
            _model.fit(X_train, y_train)
            _proba = _model.predict_proba(X_test)[:, 1]
            _auc = roc_auc_score(y_test, _proba)
            RocCurveDisplay.from_predictions(
                y_test, _proba, ax=ax, color=_color,
                name=f"{_name} (AUC={_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_title("ROC Curves — Test Set")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def features_header(mo):
    mo.md(r"""
    ---
    ## Part IX — Top Predictive Features

    This is the pedagogical payoff: which words most strongly predict each newspaper?

    The logistic regression coefficient $\beta_j$ for feature $j$ represents:
    $$\log\frac{P(\text{joumhouria} \mid \text{word}_j)}{P(\text{nahar} \mid \text{word}_j)} \propto \beta_j$$

    - **Large positive $\beta_j$** → word predicts Al Joumhouria
    - **Large negative $\beta_j$** → word predicts An-Nahar

    We use the **L1 model** (sparser — only genuinely useful words survive),
    which makes for a cleaner and more interpretable result.
    """)
    return


@app.cell
def top_features(
    LogisticRegression,
    X_train,
    lr_l1,
    np,
    plt,
    vectorizer,
    y_train,
):
    def _():
        _model = LogisticRegression(
            C=lr_l1.C_[0], max_iter=1000, solver="liblinear", penalty="l1", random_state=42
        )
        _model.fit(X_train, y_train)

        _feature_names = np.array(vectorizer.get_feature_names_out())
        _coefs = _model.coef_[0]

        _n = 25
        _top_joumhouria = np.argsort(_coefs)[-_n:][::-1]
        _top_nahar = np.argsort(_coefs)[:_n]

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        for ax, _idx, _title, _color in [
            (axes[0], _top_joumhouria, "Top words → Al Joumhouria", "#3498db"),
            (axes[1], _top_nahar[::-1], "Top words → An-Nahar", "#e74c3c"),
        ]:
            _words = _feature_names[_idx]
            _vals = np.abs(_coefs[_idx])
            ax.barh(range(len(_words)), _vals, color=_color, edgecolor="black", linewidth=0.3, alpha=0.8)
            ax.set_yticks(range(len(_words)))
            ax.set_yticklabels(_words, fontsize=9)
            ax.set_xlabel("|Coefficient|")
            ax.set_title(_title)
            ax.grid(True, alpha=0.3, axis="x")

        plt.suptitle("Most Predictive Words per Newspaper (L1 Logistic Regression)", fontsize=13)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def misclassified(
    LogisticRegression,
    X_test,
    X_test_text,
    X_train,
    lr_l1,
    np,
    y_test,
    y_train,
):
    _model = LogisticRegression(
        C=lr_l1.C_[0], max_iter=1000, solver="liblinear", penalty="l1", random_state=42
    )
    _model.fit(X_train, y_train)
    _pred = _model.predict(X_test)
    _proba = _model.predict_proba(X_test)

    _wrong = np.where(_pred != y_test)[0]
    _wrong_sorted = _wrong[np.argsort(_proba[_wrong, _pred[_wrong]])[::-1]][:5]

    print("Most confidently wrong predictions (L1 model):\n")
    _label_map = {0: "nahar", 1: "joumhouria"}
    for _i in _wrong_sorted:
        _true = _label_map[y_test[_i]]
        _predicted = _label_map[_pred[_i]]
        _conf = _proba[_i, _pred[_i]]
        print(f"  True: {_true:>12} | Predicted: {_predicted:>12} | Confidence: {_conf:.3f}")
        print(f"  Body: {X_test_text[_i][:120]}...\n")
    return


@app.cell
def summary(mo):
    mo.md(r"""
    ---
    ## Summary

    | Step | What |
    |------|------|
    | **Data** | ~1 000 Arabic news articles — Al Joumhouria (1) vs An-Nahar (0) |
    | **Cleaning** | Drop nulls, duplicates, stubs < 200 chars |
    | **Preprocessing** | Strip diacritics, normalize alef/ta marbuta, remove non-Arabic |
    | **Features** | TF-IDF, 15 000 vocab, bigrams, sublinear TF, min_df=3 |
    | **Split** | 80/20 stratified train/test |
    | **Unregularized** | Logistic loss via L-BFGS, `C=1e6` ≈ no penalty |
    | **L2** | Ridge logistic — shrinks all coefficients, `C` chosen via 5-fold CV |
    | **L1** | Lasso logistic — sparsifies coefficients, automatic feature selection |
    | **Evaluation** | Accuracy, F1, AUC-ROC, confusion matrix on held-out 20% |
    | **Interpretation** | Top L1 coefficients → most discriminative Arabic words per newspaper |

    **Why L1 for text?** Most vocabulary items are shared between newspapers.
    L1 forces irrelevant words to exactly zero, leaving only the terms that are
    genuinely characteristic of each source's editorial style and vocabulary.
    """)
    return


if __name__ == "__main__":
    app.run()
