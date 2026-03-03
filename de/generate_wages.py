"""
generate_wages.py
─────────────────
Generates the synthetic wage dataset used in the omitted variable bias tutorial
and saves it as wages.csv in the same directory.

Run this script once before opening the tutorial notebook:
    python generate_wages.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)

n = 4000

# ── Demographic variables ──────────────────────────────────────────────────
sex       = np.random.binomial(1, 0.5, n)            # 0 = Female, 1 = Male
age       = np.random.normal(38, 10, n).clip(22, 62).round().astype(int)
education = np.random.normal(14, 2.5, n).clip(8, 20).round().astype(int)

# ── Work history ───────────────────────────────────────────────────────────
# Men accumulate slightly more experience on average (reflecting structural
# labour-market inequalities built into the simulation).
experience     = np.maximum(0, age - education - 6) + np.random.normal(2 * sex, 1.5, n).clip(-3, 3)
hours_per_week = np.random.normal(40, 8, n).clip(20, 50)
hours_per_week += np.where(sex == 1,
                           np.random.normal( 2, 3, n),   # men work slightly more hours
                           np.random.normal(-2, 3, n))   # women slightly fewer
tenure = np.random.exponential(5, n).clip(0, 30)

# ── Wage formula ───────────────────────────────────────────────────────────
# The +5.0 * sex term introduces a ~$5/hr (~18%) sex wage gap.
wage = (
    5.0
    + 0.50 * education
    + 0.30 * experience
    + 0.15 * age
    + 0.10 * hours_per_week
    + 0.40 * tenure
    + 5.00 * sex                         # sex wage gap
    + np.random.normal(0, 3, n)          # random noise
).clip(10, None)                         # no wage below $10/hr

# ── Save ───────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    'age':            age,
    'education':      education,
    'experience':     experience,
    'hours_per_week': hours_per_week,
    'tenure':         tenure,
    'sex':            sex,
    'wage':           wage,
})

df.to_csv("wages.csv", index=False)
print(f"Saved wages.csv  ({len(df)} rows, columns: {', '.join(df.columns)})")
