# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "seaborn",
#     "scikit-learn",
#     "statsmodels",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def title():
    import marimo as mo

    mo.md(
        r"""
        # Linear Regression on Lebanese Real Estate Listings

        **Dataset:** ~6 000 property listings scraped from realestate.com.lb (buy only)
        **Target:** `log_price` = log(1 + price_usd) — log-transformed listing price

        Pipeline:

        > Load & Clean $\to$ EDA $\to$ Train/Test Split $\to$ OLS $\to$ Huber (Robust) $\to$
        > Ridge $\to$ Lasso $\to$ Cross-Validation $\to$ Evaluation
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
    import statsmodels.api as sm
    from scipy import stats
    from sklearn.linear_model import (
        HuberRegressor,
        Lasso,
        LassoCV,
        LinearRegression,
        Ridge,
        RidgeCV,
    )
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from statsmodels.robust.robust_linear_model import RLM
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson

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
        HuberRegressor,
        KFold,
        Lasso,
        LassoCV,
        LinearRegression,
        RLM,
        Ridge,
        RidgeCV,
        StandardScaler,
        cross_val_score,
        durbin_watson,
        het_breuschpagan,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
        plt,
        r2_score,
        sm,
        sns,
        stats,
        train_test_split,
        variance_inflation_factor,
    )


@app.cell
def load_header(mo):
    mo.md(r"""
    ---
    ## Part I — Load and Clean

    Steps:
    1. Read the scraped CSV
    2. Drop columns useless for regression
    3. Drop rows with missing target or key features
    4. Cast ID columns (`governorate_id`, `district_id`, `community_id`) to categorical strings
    5. One-hot encode categorical features
    """)
    return


@app.cell
def load_data(pd):
    raw = pd.read_csv("data/linear_regression/lebanese_zillow_like_listings.csv")
    print(f"Raw shape: {raw.shape}")
    print(f"Columns: {list(raw.columns)}")
    raw.head()
    return (raw,)


@app.cell
def show_info(raw):
    print("Null counts:")
    print(raw.isnull().sum().to_string())
    print(f"\nDtypes:\n{raw.dtypes.to_string()}")
    return


@app.cell
def describe_raw(raw):
    raw.describe()
    return


@app.cell
def clean_data(np, raw):
    df = raw.copy()

    # Drop columns useless for regression
    _drop_cols = [
        "listing_id",
        "title",
        "listing_url",
        "scraped_at_utc",
        "agency",
        "agent_name",
        "price_period",
    ]
    df = df.drop(columns=_drop_cols)

    # ── Filter 1: keep only BUY listings ──────────────────────────────────
    df = df[df["status"] == "buy"].copy()
    df = df.drop(columns=["status"])
    _n_after_buy = len(df)

    # ── Filter 2: remove non-residential / sparse property types ──────────
    # Land has no bedrooms/bathrooms concept — completely different pricing.
    # Types with < 10 rows have too few observations for reliable estimation.
    _type_counts = df["property_type"].value_counts()
    _sparse_types = _type_counts[_type_counts < 10].index.tolist()
    _exclude_types = ["Land"] + _sparse_types
    df = df[~df["property_type"].isin(_exclude_types)]
    _n_after_type = len(df)
    print(f"Removed property types: {_exclude_types}")
    print(f"  {_n_after_buy} → {_n_after_type} rows")

    # ── Filter 3: drop rows with missing target or key features ───────────
    df = df.dropna(subset=["price_usd", "bedrooms", "bathrooms", "area_sqm"])

    # ── Cast ID columns to string — they are categorical, NOT numeric ─────
    for _col in ["community_id", "district_id", "governorate_id"]:
        if _col in df.columns:
            df[_col] = df[_col].astype(str)

    # Log-transform the target (price is right-skewed)
    df["log_price"] = np.log1p(df["price_usd"])

    # ── Filter 4: MAD-based outlier removal ───────────────────────────────
    # MAD is fully robust (no normality assumption). Modified z-score > 3.5
    # flags extreme tails only (Iglewicz & Hoaglin, 1993).
    def _mad_filter(series, threshold=3.5):
        _med = series.median()
        _mad = np.median(np.abs(series - _med))
        if _mad == 0:
            return series == series  # keep all if MAD is zero
        _modified_z = 0.6745 * (series - _med) / _mad
        return np.abs(_modified_z) <= threshold

    _n_before = len(df)
    _mask_price = _mad_filter(df["log_price"])
    _mask_area = _mad_filter(df["area_sqm"])
    df = df[_mask_price & _mask_area].copy()
    print(
        f"\nMAD outlier removal (|modified z| > 3.5): "
        f"{_n_before} → {len(df)} rows (removed {_n_before - len(df)})"
    )

    print(f"\nClean shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(
        f"\nPrice range: ${df['price_usd'].min():,.0f} — ${df['price_usd'].max():,.0f}"
    )
    print(f"Median price: ${df['price_usd'].median():,.0f}")
    df.head()
    return (df,)


@app.cell
def eda_header(mo):
    mo.md(r"""
    ---
    ## Part II — Exploratory Data Analysis

    1. **Target distribution** — raw price vs log_price (our actual target)
    2. **Scatter plots** — log_price vs key numeric features
    3. **Correlation heatmap** — identify feature relationships
    """)
    return


@app.cell
def eda_price_dist(df, plt, sns):
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        sns.histplot(
            df["price_usd"], bins=60, color="#3498db", edgecolor="white", ax=ax
        )
        ax.set_xlabel("Price (USD)")
        ax.set_title("Price Distribution (Raw — right-skewed)")
        _med = df["price_usd"].median()
        ax.axvline(_med, color="red", linestyle="--", label=f"Median: ${_med:,.0f}")
        ax.legend()

        ax = axes[1]
        sns.histplot(
            df["log_price"], bins=60, color="#2ecc71", edgecolor="white", ax=ax
        )
        ax.set_xlabel("log(1 + Price)")
        ax.set_title("log_price (target — more symmetric)")
        _med_log = df["log_price"].median()
        ax.axvline(
            _med_log, color="red", linestyle="--", label=f"Median: {_med_log:.2f}"
        )
        ax.legend()

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def eda_scatter(df, plt):
    def _():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, col, color in zip(
            axes,
            ["area_sqm", "bedrooms", "bathrooms"],
            ["#e74c3c", "#3498db", "#2ecc71"],
        ):
            ax.scatter(df[col], df["log_price"], alpha=0.15, s=8, color=color)
            ax.set_xlabel(col)
            ax.set_ylabel("log(1 + Price)")
            ax.set_title(f"log_price vs {col}")
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def eda_boxplots(df, plt, sns):
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        _top_types = df["property_type"].value_counts().head(6).index
        _sub = df[df["property_type"].isin(_top_types)]
        _order = _sub.groupby("property_type")["log_price"].median().sort_values().index
        sns.boxplot(
            data=_sub,
            x="property_type",
            y="log_price",
            order=_order,
            ax=ax,
            palette="viridis",
        )
        ax.set_title("log_price by Property Type (Top 6)")
        ax.set_ylabel("log(1 + Price)")
        ax.tick_params(axis="x", rotation=30)

        ax = axes[1]
        _furn_order = df.groupby("furnished")["log_price"].median().sort_values().index
        sns.boxplot(
            data=df,
            x="furnished",
            y="log_price",
            order=_furn_order,
            ax=ax,
            palette="viridis",
        )
        ax.set_title("log_price by Furnished Status")
        ax.set_ylabel("log(1 + Price)")

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def eda_correlation(df, np, plt, sns):
    def _():
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("price_usd")
        corr = df[numeric_cols].corr()

        price_corr = (
            corr["log_price"].drop("log_price").sort_values(key=abs, ascending=False)
        )
        print("Correlation with log_price:")
        print(price_corr.round(3).to_string())

        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Correlation Heatmap (Numeric Features)")
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def feat_header(mo):
    mo.md(r"""
    ---
    ## Part III — Feature Engineering & Train/Test Split

    1. One-hot encode `property_type`, `furnished`, and `governorate_id`
       (location ID treated as **categorical**, like `zipcode` in the Zillow notebook)
    2. Drop `community_id` and `district_id` (too many categories, sparsity)
    3. 80/20 train/test split
    """)
    return


@app.cell
def prepare_features(df, pd, train_test_split):
    _TARGET = "log_price"

    # Categorical features (all are strings/objects now — status already filtered to buy only)
    _cat_cols = ["property_type", "furnished", "governorate_id"]
    _num_cols = ["bedrooms", "bathrooms", "area_sqm"]

    # Build feature matrix — drop community_id and district_id (too sparse)
    _df_model = df[_num_cols + _cat_cols + [_TARGET]].copy()

    # Keep raw price separately (for back-conversion only, never in X)
    _price = df["price_usd"]

    # One-hot encode all categoricals
    _df_encoded = pd.get_dummies(
        _df_model,
        columns=_cat_cols,
        drop_first=True,
        dtype=int,
    )

    # Separate X and y
    _y = _df_encoded[_TARGET]
    _X = _df_encoded.drop(columns=[_TARGET])

    # Train/test split
    X_train, X_test, y_train, y_test, price_train, price_test = train_test_split(
        _X, _y, _price, test_size=0.20, random_state=42
    )

    print(f"Features: {_X.shape[1]}")
    print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    print(f"\nFeature names:\n{list(_X.columns)}")
    return X_test, X_train, price_test, y_test, y_train


@app.cell
def standardize_header(mo):
    mo.md(r"""
    ---
    ## Part IV — Standardization

    Ridge and Lasso penalize $\|\boldsymbol{\theta}\|$. If features are on different scales,
    the penalty affects them unevenly. Standardize to mean 0, variance 1.
    Scaler is **fit on training set only**, then applied to both train and test.
    """)
    return


@app.cell
def standardize(StandardScaler, X_test, X_train, pd):
    _scaler = StandardScaler()
    _cols = X_train.columns

    X_train_sc = pd.DataFrame(
        _scaler.fit_transform(X_train), columns=_cols, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        _scaler.transform(X_test), columns=_cols, index=X_test.index
    )
    print("Standardized (train):")
    print(X_train_sc.describe().loc[["mean", "std"]].round(4).iloc[:, :5])
    return X_test_sc, X_train_sc


@app.cell
def ols_header(mo):
    mo.md(r"""
    ---
    ## Part V — OLS (Ordinary Least Squares)

    $$\hat{\boldsymbol{\theta}}_{\text{OLS}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

    Baseline model — no regularization.
    """)
    return


@app.cell
def ols_fit(X_train_sc, sm, y_train):
    X_ols = sm.add_constant(X_train_sc.astype(float))
    ols_model = sm.OLS(y_train.astype(float), X_ols).fit()
    print(ols_model.summary())
    return X_ols, ols_model


@app.cell
def ols_coeff_plot(ols_model, pd, plt):
    def _():
        _coefs = ols_model.params.drop("const")
        _top = _coefs.abs().nlargest(15)
        _df = pd.DataFrame(
            {
                "feature": _top.index,
                "coefficient": _coefs[_top.index].values,
            }
        ).sort_values("coefficient")

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in _df["coefficient"]]
        ax.barh(
            _df["feature"],
            _df["coefficient"],
            color=colors,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("OLS Coefficient")
        ax.set_title("Top 15 OLS Coefficients by Magnitude")
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def huber_header(mo):
    mo.md(r"""
    ---
    ## Part V-B — Robust Regression (Huber M-Estimator)

    OLS minimizes $\sum e_i^2$, so a single huge outlier can dominate the fit.
    Huber's M-estimator uses a **hybrid loss**:

    $$\rho_\delta(e) = \begin{cases} \tfrac{1}{2}e^2 & |e| \le \delta \\ \delta\,|e| - \tfrac{1}{2}\delta^2 & |e| > \delta \end{cases}$$

    - Small residuals ($|e| \le \delta$): quadratic, same influence as OLS
    - Large residuals ($|e| > \delta$): linear, so **fat-tailed outliers are downweighted**
      instead of removed

    No data is thrown away — outliers just lose their outsized pull on $\hat{\boldsymbol{\theta}}$.
    """)
    return


@app.cell
def huber_fit(RLM, X_ols, plt, sm, y_train):
    _X = X_ols.astype(float)
    _y = y_train.astype(float)

    huber_model = RLM(_y, _X, M=sm.robust.norms.HuberT()).fit()
    print(huber_model.summary())

    # Show how many observations were downweighted
    _weights = huber_model.weights
    _n_down = (_weights < 1.0).sum()
    _n_total = len(_weights)
    print(
        f"\nDownweighted observations: {_n_down} / {_n_total} "
        f"({_n_down / _n_total * 100:.1f}%)"
    )

    def _():
        fig, ax = plt.subplots(figsize=(10, 3))
        _sorted_w = sorted(_weights)
        ax.bar(range(len(_sorted_w)), _sorted_w, width=1.0, color="#9b59b6", alpha=0.7)
        ax.axhline(
            1.0,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Full weight (no downweighting)",
        )
        ax.set_xlabel("Observation (sorted by weight)")
        ax.set_ylabel("Huber Weight")
        ax.set_title("Huber Weights — Observations Below 1.0 Are Outliers Being Tamed")
        ax.legend(fontsize=9)
        plt.tight_layout()
        return plt.gca()

    _()
    return (huber_model,)


@app.cell
def residual_header(mo):
    mo.md(r"""
    ---
    ## Part VI — Residual Analysis

    Check OLS assumptions:
    - **Homoskedasticity**: constant variance of residuals
    - **Normality**: residuals ~ $\mathcal{N}(0, \sigma^2)$
    """)
    return


@app.cell
def residual_plots(np, ols_model, plt, sm, stats):
    def _():
        _resid = ols_model.resid
        _fitted = ols_model.fittedvalues

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax = axes[0, 0]
        ax.scatter(_fitted, _resid, alpha=0.15, s=5, color="#3498db")
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel("Fitted values $\\hat{y}$")
        ax.set_ylabel("Residuals $\\hat{\\varepsilon}$")
        ax.set_title("Residuals vs Fitted")

        ax = axes[0, 1]
        sm.qqplot(_resid, line="45", ax=ax, alpha=0.15, markersize=2, color="#3498db")
        ax.set_title("Normal Q-Q Plot")

        ax = axes[1, 0]
        ax.hist(
            _resid, bins=60, density=True, color="#3498db", edgecolor="white", alpha=0.8
        )
        _x = np.linspace(_resid.min(), _resid.max(), 200)
        ax.plot(
            _x,
            stats.norm.pdf(_x, _resid.mean(), _resid.std()),
            color="red",
            linewidth=2,
            label="$\\mathcal{N}(0, \\hat{\\sigma}^2)$",
        )
        ax.legend()
        ax.set_xlabel("Residual")
        ax.set_title("Residual Distribution")

        ax = axes[1, 1]
        ax.scatter(
            _fitted,
            np.sqrt(np.abs(_resid / _resid.std())),
            alpha=0.15,
            s=5,
            color="#e67e22",
        )
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("$\\sqrt{|\\text{Standardized Residual}|}$")
        ax.set_title("Scale-Location")

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def regression_conditions_header(mo):
    mo.md(r"""
    ---
    ## Part VI-B — Regression Conditions Check

    | # | Assumption | Test |
    |---|-----------|------|
    | 1 | **Linearity** | Residuals vs Fitted plot (no systematic pattern) |
    | 2 | **Independence** | Durbin-Watson ($d \approx 2$) |
    | 3 | **Homoskedasticity** | Breusch-Pagan ($H_0$: constant variance) |
    | 4 | **Normality** | Shapiro-Wilk on residuals |
    | 5 | **No multicollinearity** | VIF $< 10$ |
    """)
    return


@app.cell
def regression_conditions(
    X_ols,
    X_train_sc,
    durbin_watson,
    het_breuschpagan,
    np,
    ols_model,
    pd,
    stats,
    variance_inflation_factor,
):
    _resid = ols_model.resid

    print("=" * 60)
    print("REGRESSION CONDITIONS DIAGNOSTIC")
    print("=" * 60)

    # 1. Breusch-Pagan
    _bp_stat, _bp_pval, _, _ = het_breuschpagan(_resid, X_ols)
    print("\n1. Breusch-Pagan (Homoskedasticity):")
    print(f"   Statistic = {_bp_stat:.2f}, p-value = {_bp_pval:.2e}")
    if _bp_pval < 0.05:
        print("   REJECTED — heteroskedasticity present")
    else:
        print("   PASS — no evidence of heteroskedasticity")

    # 2. Durbin-Watson
    _dw = durbin_watson(_resid)
    print("\n2. Durbin-Watson (Independence):")
    print(f"   DW statistic = {_dw:.4f}")
    if 1.5 < _dw < 2.5:
        print("   PASS — no significant autocorrelation")
    else:
        print("   WARNING — possible autocorrelation")

    # 3. Shapiro-Wilk (sample)
    _sample = np.random.RandomState(42).choice(
        _resid, size=min(5000, len(_resid)), replace=False
    )
    _sw_stat, _sw_pval = stats.shapiro(_sample)
    print("\n3. Shapiro-Wilk (Normality):")
    print(f"   Statistic = {_sw_stat:.4f}, p-value = {_sw_pval:.2e}")
    if _sw_pval < 0.05:
        print("   REJECTED — residuals not normally distributed")
    else:
        print("   PASS — residuals appear normal")

    # 4. VIF
    _X_vif = X_train_sc.astype(float).values
    _vif_names = X_train_sc.columns.tolist()
    _vif_df = pd.DataFrame(
        {
            "Feature": _vif_names,
            "VIF": [
                variance_inflation_factor(_X_vif, _i) for _i in range(len(_vif_names))
            ],
        }
    ).sort_values("VIF", ascending=False)

    print("\n4. Variance Inflation Factor (Multicollinearity):")
    print(f"   VIF > 10: {(_vif_df['VIF'] > 10).sum()} features")
    print(f"   VIF > 5:  {(_vif_df['VIF'] > 5).sum()} features")
    print("\n   Top 10 VIF:")
    print(_vif_df.head(10).to_string(index=False))
    print("\n" + "=" * 60)
    return


@app.cell
def ridge_header(mo):
    mo.md(r"""
    ---
    ## Part VII — Ridge Regression ($L_2$ Regularization)

    $$\hat{\boldsymbol{\theta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$$

    Shrinks all coefficients toward zero — does not eliminate any.
    $\lambda$ chosen via 5-fold CV.
    """)
    return


@app.cell
def ridge_fit(RidgeCV, X_train_sc, np, plt, y_train):
    def _():
        _alphas = np.logspace(-3, 4, 200)

        _ridge_cv = RidgeCV(alphas=_alphas, cv=5, scoring="neg_mean_squared_error")
        _ridge_cv.fit(X_train_sc, y_train)

        print(f"Best lambda: {_ridge_cv.alpha_:.4f}")
        print(f"Train R2: {_ridge_cv.score(X_train_sc, y_train):.4f}")

        _coefs = []
        for _a in _alphas:
            from sklearn.linear_model import Ridge as _Ridge

            _m = _Ridge(alpha=_a).fit(X_train_sc, y_train)
            _coefs.append(_m.coef_)
        _coefs = np.array(_coefs)

        fig, ax = plt.subplots(figsize=(10, 5))
        for _i in range(_coefs.shape[1]):
            ax.plot(_alphas, _coefs[:, _i], linewidth=0.7, alpha=0.7)
        ax.axvline(
            _ridge_cv.alpha_,
            color="red",
            linestyle="--",
            label=f"CV lambda* = {_ridge_cv.alpha_:.2f}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel("Coefficient value")
        ax.set_title("Ridge Coefficient Path")
        ax.legend()
        plt.tight_layout()
        plt.gca()
        return _ridge_cv

    ridge_cv = _()
    return (ridge_cv,)


@app.cell
def lasso_header(mo):
    mo.md(r"""
    ---
    ## Part VIII — Lasso Regression ($L_1$ Regularization)

    $$\hat{\boldsymbol{\theta}}_{\text{lasso}} = \arg\min_{\boldsymbol{\theta}} \left\{ \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \lambda\|\boldsymbol{\theta}\|_1 \right\}$$

    The $L_1$ penalty produces **exact zero coefficients** — automatic feature selection.
    """)
    return


@app.cell
def lasso_fit(LassoCV, X_train_sc, np, plt, y_train):
    def _():
        _alphas = np.logspace(-4, 1, 200)

        _lasso_cv = LassoCV(alphas=_alphas, cv=5, max_iter=10000, random_state=42)
        _lasso_cv.fit(X_train_sc, y_train)

        print(f"Best lambda: {_lasso_cv.alpha_:.6f}")
        print(f"Train R2: {_lasso_cv.score(X_train_sc, y_train):.4f}")

        _n_nonzero = np.sum(_lasso_cv.coef_ != 0)
        _n_zero = np.sum(_lasso_cv.coef_ == 0)
        print(f"\nNon-zero coefficients: {_n_nonzero}")
        print(f"Eliminated (zero):     {_n_zero}")

        _coefs = []
        for _a in _alphas:
            from sklearn.linear_model import Lasso as _Lasso

            _m = _Lasso(alpha=_a, max_iter=10000).fit(X_train_sc, y_train)
            _coefs.append(_m.coef_)
        _coefs = np.array(_coefs)

        fig, ax = plt.subplots(figsize=(10, 5))
        for _i in range(_coefs.shape[1]):
            ax.plot(_alphas, _coefs[:, _i], linewidth=0.7, alpha=0.7)
        ax.axvline(
            _lasso_cv.alpha_,
            color="red",
            linestyle="--",
            label=f"CV lambda* = {_lasso_cv.alpha_:.4f}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel("Coefficient value")
        ax.set_title("Lasso Coefficient Path")
        ax.legend()
        plt.tight_layout()
        plt.gca()
        return _lasso_cv

    lasso_cv = _()
    return (lasso_cv,)


@app.cell
def lasso_survivors(X_train_sc, lasso_cv, pd):
    _coefs = pd.Series(lasso_cv.coef_, index=X_train_sc.columns)
    _nonzero = _coefs[_coefs != 0].sort_values(key=abs, ascending=False)
    print(f"Lasso non-zero coefficients ({len(_nonzero)} features):\n")
    print(_nonzero.round(4).to_string())
    return


@app.cell
def cv_header(mo):
    mo.md(r"""
    ---
    ## Part IX — Cross-Validation Comparison

    10-fold CV on the training set to compare OLS, Ridge, Lasso, and Huber.
    """)
    return


@app.cell
def cv_comparison(
    HuberRegressor,
    KFold,
    Lasso,
    LinearRegression,
    Ridge,
    X_train_sc,
    cross_val_score,
    lasso_cv,
    pd,
    plt,
    ridge_cv,
    y_train,
):
    def _():
        _kf = KFold(n_splits=10, shuffle=True, random_state=42)
        _models = {
            "OLS": LinearRegression(),
            "Ridge": Ridge(alpha=ridge_cv.alpha_),
            "Lasso": Lasso(alpha=lasso_cv.alpha_, max_iter=10000),
            "Huber": HuberRegressor(epsilon=1.35, max_iter=200),
        }
        _cv_results = {}
        for _name, _model in _models.items():
            _scores = cross_val_score(
                _model, X_train_sc, y_train, cv=_kf, scoring="neg_mean_squared_error"
            )
            _cv_results[_name] = -_scores

        _cv_df = pd.DataFrame(_cv_results)

        fig, ax = plt.subplots(figsize=(9, 5))
        _bp = ax.boxplot(
            [_cv_df[c] for c in _cv_df.columns],
            labels=_cv_df.columns,
            patch_artist=True,
            medianprops=dict(color="red", linewidth=2),
        )
        _palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
        for _patch, _color in zip(_bp["boxes"], _palette):
            _patch.set_facecolor(_color)
            _patch.set_alpha(0.6)

        ax.set_ylabel("MSE (lower is better)")
        ax.set_title("10-Fold Cross-Validation MSE")

        for _i, _col in enumerate(_cv_df.columns):
            _mean = _cv_df[_col].mean()
            ax.text(
                _i + 1,
                _mean,
                f"{_mean:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()

        print("10-Fold CV MSE (mean +/- std):")
        for _col in _cv_df.columns:
            print(
                f"  {_col:10s}: {_cv_df[_col].mean():.4f} +/- {_cv_df[_col].std():.4f}"
            )
        return plt.gca()

    _()
    return


@app.cell
def eval_header(mo):
    mo.md(r"""
    ---
    ## Part X — Final Evaluation on Test Set

    | Metric | Definition |
    |--------|-----------|
    | **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
    | **RMSE** | $\sqrt{\text{MSE}}$ |
    | **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ |
    | **$R^2$** | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ |
    """)
    return


@app.cell
def final_evaluation(
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
    X_test_sc,
    X_train_sc,
    lasso_cv,
    mean_absolute_error,
    mean_squared_error,
    np,
    pd,
    plt,
    r2_score,
    ridge_cv,
    y_test,
    y_train,
):
    _models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=ridge_cv.alpha_),
        "Lasso": Lasso(alpha=lasso_cv.alpha_, max_iter=10000),
        "Huber": HuberRegressor(epsilon=1.35, max_iter=200),
    }

    _results = []
    predictions = {}
    for _name, _model in _models.items():
        _model.fit(X_train_sc, y_train)
        _pred = _model.predict(X_test_sc)
        predictions[_name] = _pred

        _n = len(y_test)
        _d = X_test_sc.shape[1]
        _r2 = r2_score(y_test, _pred)

        _results.append(
            {
                "Model": _name,
                "MSE": mean_squared_error(y_test, _pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, _pred)),
                "MAE": mean_absolute_error(y_test, _pred),
                "R2": _r2,
                "Adj R2": 1 - (_n - 1) / (_n - _d - 1) * (1 - _r2),
            }
        )

    eval_df = pd.DataFrame(_results).set_index("Model")
    print("Test Set Evaluation (on log_price):\n")
    print(eval_df.round(4).to_string())

    def _():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        _metrics = ["RMSE", "R2", "MAE"]
        _colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

        for ax, metric in zip(axes, _metrics):
            _vals = eval_df[metric]
            _bars = ax.bar(
                _vals.index,
                _vals.values,
                color=_colors,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_title(metric)
            ax.set_ylabel(metric)
            for _bar, _v in zip(_bars, _vals.values):
                ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _v,
                    f"{_v:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        fig.suptitle("Test Set Performance", fontsize=13, y=1.02)
        plt.tight_layout()
        return plt.gca()

    _()
    return (predictions,)


@app.cell
def actual_vs_predicted(np, plt, predictions, price_test, r2_score):
    def _():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        _colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

        for ax, (_name, _pred), _color in zip(axes.flat, predictions.items(), _colors):
            _pred_usd = np.expm1(_pred)
            _actual_usd = price_test.values

            ax.scatter(_actual_usd, _pred_usd, alpha=0.2, s=8, color=_color)
            _max_val = max(_actual_usd.max(), _pred_usd.max()) * 1.05
            ax.plot(
                [0, _max_val],
                [0, _max_val],
                "k--",
                linewidth=1,
                label="Perfect prediction",
            )
            ax.set_xlabel("Actual Price (USD)")
            ax.set_ylabel("Predicted Price (USD)")
            ax.set_title(f"{_name}: Actual vs Predicted")
            ax.legend(fontsize=8)
            _r2 = r2_score(_actual_usd, _pred_usd)
            ax.text(
                0.05,
                0.92,
                f"R2 = {_r2:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def oos_sample_header(mo):
    mo.md(r"""
    ---
    ## Part XI — Out-of-Sample Predictions

    A random sample of 20 test-set listings showing actual price vs each model's
    prediction (converted back to USD from log-space), plus the percentage error.
    """)
    return


@app.cell
def oos_sample_table(np, pd, predictions, price_test, y_test):
    _sample_idx = price_test.sample(n=20, random_state=42).index

    _sample_df = pd.DataFrame(
        {
            "Actual (USD)": price_test.loc[_sample_idx].values,
        }
    )

    for _name, _pred_log in predictions.items():
        _pred_series = pd.Series(np.expm1(_pred_log), index=y_test.index)
        _sample_df[f"{_name} Pred (USD)"] = _pred_series.loc[_sample_idx].values
        _sample_df[f"{_name} Error %"] = (
            (_sample_df[f"{_name} Pred (USD)"] - _sample_df["Actual (USD)"])
            / _sample_df["Actual (USD)"]
            * 100
        )

    _sample_df.index = pd.RangeIndex(1, len(_sample_df) + 1, name="#")

    for _col in _sample_df.columns:
        if "USD" in _col:
            _sample_df[_col] = _sample_df[_col].apply(lambda x: f"${x:,.0f}")
        elif "Error" in _col:
            _sample_df[_col] = _sample_df[_col].apply(lambda x: f"{x:+.1f}%")

    print("Out-of-Sample Predictions (20 random test listings):\n")
    print(_sample_df.to_string())
    return


@app.cell
def oos_line_plot(np, pd, plt, predictions, price_test, y_test):
    def _():
        _sample_idx = price_test.sample(n=50, random_state=7).index
        _actual_usd = price_test.loc[_sample_idx].values
        _sort_order = np.argsort(_actual_usd)
        _actual_sorted = _actual_usd[_sort_order]
        _x = np.arange(len(_actual_sorted))

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(
            _x,
            _actual_sorted,
            "ko-",
            markersize=4,
            linewidth=1.5,
            label="Actual",
            zorder=5,
        )

        _colors = {
            "OLS": "#e74c3c",
            "Ridge": "#3498db",
            "Lasso": "#2ecc71",
            "Huber": "#9b59b6",
        }
        for _name, _pred_log in predictions.items():
            _pred_series = pd.Series(np.expm1(_pred_log), index=y_test.index)
            _pred_usd = _pred_series.loc[_sample_idx].values[_sort_order]
            ax.plot(
                _x,
                _pred_usd,
                "s--",
                markersize=3,
                linewidth=1,
                alpha=0.8,
                color=_colors[_name],
                label=_name,
            )

        ax.set_xlabel("Sample (sorted by actual price)")
        ax.set_ylabel("Price (USD)")
        ax.set_title("OOS Predictions vs Actual — 50 Random Test Listings")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def comparison_header(mo):
    mo.md(r"""
    ---
    ## Part XII — Coefficient Comparison

    - **OLS:** no shrinkage
    - **Huber:** same features, but **outliers downweighted** — robust to fat tails
    - **Ridge ($L_2$):** all coefficients shrink toward zero, none are exactly zero
    - **Lasso ($L_1$):** coefficients shrink and some are **set to zero** — feature selection
    """)
    return


@app.cell
def coeff_comparison(
    X_train_sc,
    huber_model,
    lasso_cv,
    ols_model,
    pd,
    plt,
    ridge_cv,
):
    def _():
        _cols = X_train_sc.columns
        _ols_coefs = ols_model.params.drop("const")
        _ols_aligned = _ols_coefs.reindex(_cols).fillna(0)

        _huber_coefs = huber_model.params.drop("const")
        _huber_aligned = _huber_coefs.reindex(_cols).fillna(0)

        _comp_df = pd.DataFrame(
            {
                "OLS": _ols_aligned.values,
                "Huber": _huber_aligned.values,
                "Ridge": ridge_cv.coef_,
                "Lasso": lasso_cv.coef_,
            },
            index=_cols,
        )

        _top = _comp_df["OLS"].abs().nlargest(12).index
        _plot_df = _comp_df.loc[_top].sort_values("OLS")

        fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

        for ax, _col, _color in zip(
            axes,
            ["OLS", "Huber", "Ridge", "Lasso"],
            ["#e74c3c", "#9b59b6", "#3498db", "#2ecc71"],
        ):
            ax.barh(
                _plot_df.index,
                _plot_df[_col],
                color=_color,
                edgecolor="black",
                linewidth=0.4,
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Coefficient")
            ax.set_title(_col)

        fig.suptitle(
            "Coefficient Comparison (Top 12 by OLS Magnitude)", fontsize=13, y=1.01
        )
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def summary(mo):
    mo.md(r"""
    ---
    ## Summary

    | Step | What |
    |------|------|
    | **Data** | Lebanese **buy** listings — lands & sparse types removed (realestate.com.lb) |
    | **Target** | `log(1 + price_usd)` (log-transform to reduce skewness) |
    | **Features** | `area_sqm`, `bedrooms`, `bathrooms` + one-hot encoded `property_type`, `furnished`, `governorate_id` |
    | **OLS** | Baseline — no regularization |
    | **Huber** | Robust M-estimator — **downweights fat-tailed outliers** instead of removing them |
    | **Ridge** | $L_2$ penalty — shrinks all coefficients |
    | **Lasso** | $L_1$ penalty — eliminates irrelevant features |
    | **Conditions** | Breusch-Pagan, Durbin-Watson, Shapiro-Wilk, VIF |
    | **Evaluation** | R2, RMSE, MAE on held-out 20% test set |
    """)
    return


if __name__ == "__main__":
    app.run()
