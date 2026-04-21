"""
Correlation analysis on Southeast Asia cleaned_data.csv
Generates:
  1. Heatmap of correlations between GDP per capita and every other factor
  2. Heatmap of correlations between Life Expectancy and every other factor
  3. BONUS: Combined heatmap showing correlations of both GDP per capita
     and Life Expectancy against every other factor side-by-side
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. Load & prepare the data
# ---------------------------------------------------------------------------
df = pd.read_csv("cleaned_data.csv")

# Make sure Year is numeric (and drop fully empty rows if any)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(how="all")

# The GDP column is in billions USD, Population in millions.
# GDP per capita (USD) = GDP * 1e9 / (Population * 1e6) = GDP / Population * 1000
gdp_col  = "GDP (current USD, billions)"
pop_col  = "Population (millions)"
df["GDP per Capita (USD)"] = df[gdp_col] / df[pop_col] * 1000

# Life-expectancy column is the last numeric one in the cleaned file
life_col = df.columns[-2]   # the column we just appended is last; life exp is the one before
# (we just added GDP per Capita at the end, so last-but-one is life expectancy)
# rename for clarity
df = df.rename(columns={life_col: "Life Expectancy (years)"})
life_col = "Life Expectancy (years)"

# Keep only numeric columns for correlation (drop Year – it's an index-like field,
# keep it commented out if you'd rather include it)
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["Year"])

# ---------------------------------------------------------------------------
# 2. Compute the full correlation matrix once
# ---------------------------------------------------------------------------
corr = numeric_df.corr(method="pearson")

# ---------------------------------------------------------------------------
# 3. Heatmap 1 – GDP per capita vs every other factor
# ---------------------------------------------------------------------------
gdp_corr = corr[["GDP per Capita (USD)"]].drop(index="GDP per Capita (USD)")
gdp_corr = gdp_corr.sort_values("GDP per Capita (USD)", ascending=False)

plt.figure(figsize=(6, 8))
sns.heatmap(
    gdp_corr,
    annot=True, fmt=".2f",
    cmap="coolwarm", vmin=-1, vmax=1,
    cbar_kws={"label": "Pearson correlation"},
    linewidths=0.5,
)
plt.title("Correlation of GDP per Capita with other factors\n(Southeast Asia, 2010-2022)")
plt.tight_layout()
plt.savefig("heatmap_gdp_per_capita.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 4. Heatmap 2 – Life expectancy vs every other factor
# ---------------------------------------------------------------------------
life_corr = corr[[life_col]].drop(index=life_col)
life_corr = life_corr.sort_values(life_col, ascending=False)

plt.figure(figsize=(6, 8))
sns.heatmap(
    life_corr,
    annot=True, fmt=".2f",
    cmap="coolwarm", vmin=-1, vmax=1,
    cbar_kws={"label": "Pearson correlation"},
    linewidths=0.5,
)
plt.title("Correlation of Life Expectancy with other factors\n(Southeast Asia, 2010-2022)")
plt.tight_layout()
plt.savefig("heatmap_life_expectancy.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 5. BONUS Heatmap 3 – GDP per capita AND Life expectancy vs every other factor
# ---------------------------------------------------------------------------
combined = corr[["GDP per Capita (USD)", life_col]].drop(
    index=["GDP per Capita (USD)", life_col]
)
# sort rows by the average absolute correlation, so the strongest drivers are on top
combined["__order__"] = combined.abs().mean(axis=1)
combined = combined.sort_values("__order__", ascending=False).drop(columns="__order__")

plt.figure(figsize=(8, 8))
sns.heatmap(
    combined,
    annot=True, fmt=".2f",
    cmap="coolwarm", vmin=-1, vmax=1,
    cbar_kws={"label": "Pearson correlation"},
    linewidths=0.5,
)
plt.title("Correlation of GDP per Capita & Life Expectancy\nwith every other factor (Southeast Asia, 2010-2022)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("heatmap_gdp_and_life_expectancy.png", dpi=150)
plt.show()

print("Done. Three PNG files saved:")
print("  - heatmap_gdp_per_capita.png")
print("  - heatmap_life_expectancy.png")
print("  - heatmap_gdp_and_life_expectancy.png")