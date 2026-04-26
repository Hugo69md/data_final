"""
Multilinear regression on Southeast Asia cleaned_data.csv
--------------------------------------------------------
Model 1 : target = GDP per Capita
          predictors = Urban Population, CO2 Emissions per Capita, Life Expectancy

Model 2 : target = Life Expectancy
          predictors = Basic Drinking Water Access, Access to Electricity, GDP per Capita

Splits   : 75 % train / 25 % test  (random seed drawn every run)
Outputs  : regression_report.txt
           model1_pred_vs_actual.png , model1_residuals.png
           model2_pred_vs_actual.png , model2_residuals.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import random

# ---------------------------------------------------------------------------
# 1. Load data & build GDP per capita column
# ---------------------------------------------------------------------------
df = pd.read_csv("cleaned_data.csv")

GDP_COL   = "GDP (current USD, billions)"
POP_COL   = "Population (millions)"
URBAN     = "Urban Population (%)"
CO2       = "CO2 Emissions per Capita (metric tons)"
LIFE      = "Life Expectancy at Birth (years)"
WATER     = "Basic Drinking Water Access (%)"
ELEC      = "Access to Electricity (% of population)"
GDP_CAP   = "GDP per Capita (USD)"

# GDP (billions USD) / Population (millions) * 1000  -> USD per person
df[GDP_CAP] = df[GDP_COL] / df[POP_COL] * 1000


# ---------------------------------------------------------------------------
# 2. Plotting helpers -- produces the TWO required graphs per regression
# ---------------------------------------------------------------------------
def plot_pred_vs_actual(y_train, yhat_train, y_test, yhat_test,
                        target, r2_train, r2_test, filename, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_train, yhat_train, alpha=0.7, label=f"Train (R²={r2_train:.3f})",
                color="steelblue", edgecolor="k")
    plt.scatter(y_test,  yhat_test,  alpha=0.9, label=f"Test  (R²={r2_test:.3f})",
                color="darkorange", marker="^", edgecolor="k")

    # perfect prediction line y = x
    all_y = np.concatenate([y_train, y_test, yhat_train, yhat_test])
    lo, hi = all_y.min(), all_y.max()
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="Ideal  (y = x)")

    plt.xlabel(f"Actual  {target}")
    plt.ylabel(f"Predicted  {target}")
    plt.title(f"Predicted vs Actual — {title}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_residuals(yhat_train, resid_train, yhat_test, resid_test,
                   target, filename, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(yhat_train, resid_train, alpha=0.7, label="Train",
                color="steelblue", edgecolor="k")
    plt.scatter(yhat_test,  resid_test,  alpha=0.9, label="Test",
                color="darkorange", marker="^", edgecolor="k")
    plt.axhline(0, color="k", linestyle="--", lw=1)
    plt.xlabel(f"Predicted  {target}")
    plt.ylabel("Residual  (actual − predicted)")
    plt.title(f"Residuals vs Predicted — {title}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Generic regression runner (fit + report + 2 graphs)
# ---------------------------------------------------------------------------
def run_regression(data, target, predictors, title, plot_tag):
    subset = data[[target] + predictors].dropna()
    X = subset[predictors].values
    y = subset[target].values

    randnumb = random.randint(0, 10000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=randnumb
    )

    # --- sklearn for prediction metrics ---
    model = LinearRegression().fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    r2_train  = r2_score(y_train, y_pred_train)
    r2_test   = r2_score(y_test,  y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))

    # --- statsmodels for p-values / confidence intervals ---
    X_sm = sm.add_constant(X_train)
    sm_model = sm.OLS(y_train, X_sm).fit()

    # --- TWO plots per regression ---------------------------------------
    plot_pred_vs_actual(
        y_train, y_pred_train, y_test, y_pred_test,
        target=target,
        r2_train=r2_train, r2_test=r2_test,
        filename=f"{plot_tag}_pred_vs_actual.png",
        title=title,
    )
    plot_residuals(
        y_pred_train, y_train - y_pred_train,
        y_pred_test,  y_test  - y_pred_test,
        target=target,
        filename=f"{plot_tag}_residuals.png",
        title=title,
    )

    # --- build a textual report -----------------------------------------
    lines = []
    lines.append("=" * 78)
    lines.append(title)
    lines.append("=" * 78)
    lines.append(f"Target          : {target}")
    lines.append(f"Predictors      : {', '.join(predictors)}")
    lines.append(f"Rows used       : {len(subset)}  (train={len(X_train)}, test={len(X_test)})")
    lines.append(f"random_state    : {randnumb}")
    lines.append("")
    lines.append("--- sklearn LinearRegression (75/25 split) -----------------------------------")
    lines.append(f"R^2  (train)    : {r2_train:.4f}")
    lines.append(f"R^2  (test)     : {r2_test:.4f}")
    lines.append(f"RMSE (train)    : {rmse_train:.4f}")
    lines.append(f"RMSE (test)     : {rmse_test:.4f}")
    lines.append(f"Intercept       : {model.intercept_:.4f}")
    for name, coef in zip(predictors, model.coef_):
        lines.append(f"  beta[{name:<45}] = {coef:+.6f}")
    lines.append("")
    lines.append("--- statsmodels OLS (same training set, gives p-values) ---------------------")
    lines.append(f"R^2             : {sm_model.rsquared:.4f}")
    lines.append(f"Adjusted R^2    : {sm_model.rsquared_adj:.4f}")
    lines.append(f"F-statistic     : {sm_model.fvalue:.4f}")
    lines.append(f"Prob (F-stat)   : {sm_model.f_pvalue:.4g}")
    lines.append("")
    names  = ["const"] + predictors
    coefs  = sm_model.params
    pvals  = sm_model.pvalues
    tvals  = sm_model.tvalues
    stderr = sm_model.bse
    ci     = sm_model.conf_int(alpha=0.05)

    header = f"{'Variable':<48} {'coef':>12} {'std err':>10} {'t':>8} {'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, n in enumerate(names):
        lines.append(
            f"{n:<48} {coefs[i]:>12.4f} {stderr[i]:>10.4f} {tvals[i]:>8.3f} "
            f"{pvals[i]:>10.4g} {ci[i][0]:>10.4f} {ci[i][1]:>10.4f}"
        )
    lines.append("")
    lines.append("Significance (alpha = 0.05):")
    for n, p in zip(names, pvals):
        flag = "SIGNIFICANT" if p < 0.05 else "not significant"
        lines.append(f"  {n:<48} p = {p:.4g}  -> {flag}")
    lines.append("")
    lines.append("Full statsmodels summary:")
    lines.append(str(sm_model.summary()))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Run the two regressions
# ---------------------------------------------------------------------------
report1 = run_regression(
    df,
    target=GDP_CAP,
    predictors=[URBAN, LIFE, CO2],
    title="MODEL 1 - GDP per Capita ~ Urban Population + CO2 Emissions + Life Expectancy",
    plot_tag="model1",
)

report2 = run_regression(
    df,
    target=LIFE,
    predictors=[WATER, ELEC, GDP_CAP],
    title="MODEL 2 - Life Expectancy ~ Basic Water Access + Access to Electricity + GDP per Capita",
    plot_tag="model2",
)

# ---------------------------------------------------------------------------
# 5. Save everything
# ---------------------------------------------------------------------------
full_report = (
    "Multilinear Regression Report - Southeast Asia dataset\n"
    "======================================================\n\n"
    + report1
    + "\n\n"
    + report2
)

with open("regression_report.txt", "w", encoding="utf-8") as f:
    f.write(full_report)

print(full_report)
print("\nReport written to regression_report.txt")
print("Plots saved: model1_pred_vs_actual.png, model1_residuals.png,"
      " model2_pred_vs_actual.png, model2_residuals.png")