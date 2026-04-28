import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "/Users/emilbundesen/Desktop/Bachelor/Data/49_Industry_monthly.csv",
    sep=",",
    header=6,
    nrows=2404  # læser kun de relevante rækker fra starten
)

df = df.rename(columns={df.columns[0]: "Date"})
df = df[df["Date"].astype(str).str.match(r"^\d{6,8}$", na=False)]

df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
df = df.set_index("Date")

df = df.apply(pd.to_numeric, errors="coerce")
df.replace([-99.99, -999], np.nan, inplace=True)

df = df.sort_index()
df_clean = df[~df.index.duplicated(keep="first")].copy()


print("Dataframe shape:", df_clean.shape)
print("Periode:", df_clean.index.min(), "til", df_clean.index.max())

print(df_clean.head())

"""
w_labels = ["0%", "10%", "25%", "50%", "75%", "90%", "99%", "100%"]
x        = np.arange(len(w_labels))

sr_paper = [0.56, 0.68, 0.75, 0.79, 0.80, 0.79, 0.73, 0.71]
sr_own   = [0.54, 0.64, 0.69, 0.72, 0.74, 0.74, 0.70, 0.69]

fig, ax1 = plt.subplots(figsize=(10, 5))

# SR kurver
ax1.plot(x, sr_paper, marker="o", linewidth=2, color="steelblue",
         label="EPO (Pedersen 2021)")
ax1.plot(x, sr_own,   marker="s", linewidth=2, color="darkorange",
         linestyle="--", label="Egen implementering")

ax1.set_xlabel("Shrinkage parameter $w$", fontsize=12)
ax1.set_ylabel("Sharpe Ratio", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(w_labels)
ax1.set_ylim(0.40, 0.95)
ax1.grid(True, alpha=0.3)

# Legende
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1 , fontsize=10, loc="lower left")

plt.title("Replikering af Equity 1",
          fontsize=13, pad=15)
plt.tight_layout()
plt.show()
"""