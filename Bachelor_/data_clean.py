import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Dette afsnit renser datasættet for det daglige afkast for 49 industrier
"""

df = pd.read_csv(
    "/Users/emilbundesen/Desktop/Bachelor/Data/49_Industry_Portfolios_Daily.csv",
    sep=",",
    header=5) #fjerner headers

midpoint = len(df) // 2
df = df.iloc[:midpoint] #tager kun value-weigthed daglig afkast

df = df[df[df.columns[0]].astype(str).str.match(r"^\d{8}$", na=False)]  # Behold kun rækker med faktiske datoer

df = df.rename(columns={df.columns[0]: "Date"})
df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d") #konvertere til date-format
df = df.set_index("Date")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce") #konvertere daglig afkast til numerisk

df.replace([-99.99, -999], np.nan, inplace=True) #fjerner missing values

df = df.sort_index()
df_clean = df.loc["1985-01-01":"2025-12-31"]
df_clean = df_clean[~df_clean.index.duplicated(keep='first')]

"""
Dette afsnit læser og korrigerer for rf, så vi kun analysere på merafkast. Vi bruger daglig rf
"""

rf_df = pd.read_csv(
    "/Users/emilbundesen/Desktop/Bachelor/Data/F-F_Research_Data_Factors_daily.csv",
    skiprows=3
)

rf_df = rf_df[rf_df.iloc[:,0].astype(str).str.match(r"^\d{8}$", na=False)] #Beholder run rækker med faktiske dage dvs. på formatet yyyymmdd
rf_df = rf_df.rename(columns={rf_df.columns[0]: "Date"}) #navngiver 1 søjle
rf_df["Date"] = pd.to_datetime(rf_df["Date"], format="%Y%m%d") #konverterer til datatype date
rf_df = rf_df.set_index("Date") #indeksere på Date

rf_df["RF"] = pd.to_numeric(rf_df["RF"], errors="coerce") / 100 #omskriver den risikofrie rente til numerisk værdi i pct.
rf_df = rf_df.loc["1985-01-01":"2025-12-31"] #fjerner rækker udenfor tidsrækken
rf_df = rf_df[~rf_df.index.duplicated(keep='first')] #Fjerner duplikerede rækker: 10080 rækker = 40 års data

"""
Dette afsnit sikrer, at rf og df_clean matcher
"""

df_clean, rf = df_clean.align(rf_df["RF"], join="inner", axis=0)

# Sanity check
print("Dataframe shape:", df_clean.shape)
print("RF shape:", rf.shape)
print("Periode:", df_clean.index.min(), "til", df_clean.index.max())

"""
Denne sektion finder og viser bedste og værste industrier

Der skal korrigeres med den risikofrierente, så vi kun kigger på merafkast
"""

daglig_afkast = df_clean.fillna(0) / 100 #man kunne ændre måden vi håndtere manglende data

kum = (1 + daglig_afkast).cumprod()
kum_rf = (1 + rf).cumprod()

kum_merafkast = kum.div(kum_rf, axis=0) - 1 #Ratio-metode


#Bedste og dårligste industrier
sidste_dag_afkast = kum_merafkast.iloc[-1]
top3 = sidste_dag_afkast.sort_values(ascending=False).head(3).index
bottom3 = sidste_dag_afkast.sort_values(ascending=True).head(3).index

print(f"Top 3 industrier: {top3}")
print(f"Dårligste 3 industrier: {bottom3}")

"""
Denne sektion plotter kumulativt afkast
"""

plt.figure(figsize=(14,7))

for industry in top3:
    plt.plot(kum_merafkast.index, kum_merafkast[industry], label=f"Top: {industry}", linewidth=2)

for industry in bottom3:
    plt.plot(kum_merafkast.index, kum_merafkast[industry], label=f"Bottom: {industry}", linestyle='--', linewidth=2)

plt.title("Kumulativt dagligt merafkast (1985–2024) – Top 3 vs. Bottom 3 industrier")
plt.xlabel("Date")
plt.ylabel("Kumulativt merafkast")
plt.grid(True)
plt.legend()
plt.show()

print(kum_merafkast)