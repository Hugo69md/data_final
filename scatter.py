"""
Visualise the parallel between GDP per Capita and Life Expectancy
-----------------------------------------------------------------
Three identical scatter plots (x = GDP per Capita, y = Life Expectancy),
each coloured (hue) by a different third variable:

    1. Basic Drinking Water Access (%)
    2. Access to Electricity (% of population)
    3. Urban Population (%)

Input  : cleaned_data.csv
Output : scatter_gdp_life_water.png
         scatter_gdp_life_electricity.png
         scatter_gdp_life_urban.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. Load data & compute GDP per capita
# ---------------------------------------------------------------------------
df = pd.read_csv("cleaned_data.csv")

GDP_COL  = "GDP (current USD, billions)"
POP_COL  = "Population (millions)"
LIFE     = "Life Expectancy at Birth (years)"
WATER    = "Basic Drinking Water Access (%)"
ELEC     = "Access to Electricity (% of population)"
URBAN    = "Urban Population (%)"
GDP_CAP  = "GDP per Capita (USD)"

# GDP (billions USD) / Population (millions) * 1000  -> USD per person
df[GDP_CAP] = df[GDP_COL] / df[POP_COL] * 1000


# ---------------------------------------------------------------------------
# 2. Generic plotting helper (keeps all 3 plots visually identical)
# ---------------------------------------------------------------------------
def scatter_hue(data, hue_col, filename):
    subset = data[[GDP_CAP, LIFE, hue_col]].dropna()

    plt.figure(figsize=(9, 6))
    ax = sns.scatterplot(
        data=subset,
        x=GDP_CAP,
        y=LIFE,
        hue=hue_col,
        palette="viridis",
        s=90,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("GDP per Capita (USD)")
    ax.set_ylabel("Life Expectancy at Birth (years)")
    ax.set_title(f"GDP per Capita vs Life Expectancy\n(hue = {hue_col})")
    ax.grid(alpha=0.3)
    plt.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Produce the three graphs
# ---------------------------------------------------------------------------
scatter_hue(df, WATER, "scatter_gdp_life_water.png")
scatter_hue(df, ELEC,  "scatter_gdp_life_electricity.png")
scatter_hue(df, URBAN, "scatter_gdp_life_urban.png")

print("Saved:")
print("  - scatter_gdp_life_water.png")
print("  - scatter_gdp_life_electricity.png")
print("  - scatter_gdp_life_urban.png")