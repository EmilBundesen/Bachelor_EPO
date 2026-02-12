import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Dette afsnit renser datasættet for det daglige afkast for 49 industrier
"""
df = pd.read_csv(
    "/Users/emilbundesen/Desktop/Bachelor/Data/49_Industry_Portfolios_Daily.csv",
    sep=",",
    header=5,
)

df = df[df[df.columns[0]].astype(str).str.match(r"^\d{8}$", na=False)]  # Beholder rækker med faktiske dage
df = df.rename(columns={df.columns[0]: "Date"})
df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
df = df.set_index("Date")
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.replace([-99.99, -999], np.nan, inplace=True)  # missing data

start_date = "1985-01-01"
end_date = ("2025-12-31")
df = df.sort_index()
df_clean = df.loc[start_date:end_date]
df_clean = df_clean[~df_clean.index.duplicated(keep='first')]


"""
Læs og konverter månedlig risikofri rente til daglig rente
"""
rf_df = pd.read_csv(
    "/Users/emilbundesen/Desktop/Bachelor/Data/Månedlig_rf.csv",
    sep=",",
    skiprows=3
)

# Behold kun rækker med gyldige datoer
rf_df = rf_df[rf_df.iloc[:,0].astype(str).str.match(r"^\d{6,8}$", na=False)]
rf_df = rf_df.rename(columns={rf_df.columns[0]: "Date"})
rf_df["Date"] = pd.to_datetime(rf_df["Date"].astype(str), format="%Y%m")
rf_df = rf_df.set_index("Date")

# Konverter RF-kolonnen til decimal
rf_df["RF"] = pd.to_numeric(rf_df["RF"], errors="coerce") / 100

# Begræns tidsperiode
rf_df = rf_df.loc[start_date:end_date]

# Forward-fill RF til alle handelsdage i df_clean
rf_daily = rf_df.reindex(df_clean.index, method='ffill')

# Konverter månedlig RF til daglig RF (ca. 21 handelsdage per måned)
rf_daily["RF"] = (1 + rf_daily["RF"])**(1/21) - 1

"""
Sanity check
"""
print("Industridata shape:", df_clean.shape)
print("Daglig RF shape:", rf_daily.shape)
print("Periode:", df_clean.index.min(), "til", df_clean.index.max())

"""
Beregn kumulativt merafkast for hver industri
"""
daglig_afkast = df_clean.fillna(0) / 100  # konverter til decimal

daglig_merafkast = daglig_afkast.subtract(rf_daily["RF"], axis=0)

kum_merafkast = (1 + daglig_merafkast).cumprod() - 1

# Bedste og dårligste industrier
sidste_dag_afkast = kum_merafkast.iloc[-1]
top3 = sidste_dag_afkast.sort_values(ascending=False).head(3).index
bottom3 = sidste_dag_afkast.sort_values(ascending=True).head(3).index

print(kum_merafkast)

print(f"Top 3 industrier: {top3}")
print(f"Dårligste 3 industrier: {bottom3}")

"""
Plot kumulativt merafkast
"""
plt.figure(figsize=(14,7))

for industry in top3:
    plt.plot(kum_merafkast.index, kum_merafkast[industry], label=f"Top: {industry}", linewidth=2)

for industry in bottom3:
    plt.plot(kum_merafkast.index, kum_merafkast[industry], label=f"Bottom: {industry}", linestyle='--', linewidth=2)

plt.title("Kumulativt dagligt merafkast (1985–2025) – Top 3 vs. Bottom 3 industrier")
plt.xlabel("Date")
plt.ylabel("Kumulativt merafkast")
plt.grid(True)
plt.legend()
plt.show()