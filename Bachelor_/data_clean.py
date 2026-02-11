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

df = df[df[df.columns[0]].astype(str).str.match(r"^\d{8}$", na=False)] #Beholder run rækker med faktiske dage dvs. på formatet yyyymmdd

df = df.rename(columns={df.columns[0]: "Date"})

df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d") #Konverter til datatype date
df = df.set_index("Date")

for col in df.columns: #Konverter resten af kolonner til numerisk
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.replace([-99.99, -999], np.nan, inplace=True) #missing data

start_date = "1985-01-01" #start for det data vi kigger på (afgrænsning)
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")  #i dag
df = df.sort_index()
df_clean = df.loc[start_date:end_date] #fjerner rækker udenfor tidsrækken
df_clean = df_clean[~df_clean.index.duplicated(keep='first')]#Fjerner duplikerede rækker: 10300 rækker = 41 års data

"""
Denne sektion finder og viser bedste og værste industrier

Der skal korrigeres med den risikofrierente, så vi kun kigger på merafkast
"""



daglig_afkast = df_clean / 100  #Da værdierne er i %

kum_afkast = (1 + daglig_afkast).cumprod() - 1

sidste_dag_afkast = kum_afkast.iloc[-1]  # kumulativt afkast på sidste dag
top3 = sidste_dag_afkast.sort_values(ascending=False).head(3).index
bottom3 = sidste_dag_afkast.sort_values(ascending=True).head(3).index

print(f"Top 3 industrier: {top3}")
print(f"Dårligste 3 industrier: {bottom3}")

plt.figure(figsize=(14,7))

# Top 3
for industry in top3:
    plt.plot(kum_afkast.index, kum_afkast[industry], label=f"Top: {industry}", linewidth=2)

# Bottom 3
for industry in bottom3:
    plt.plot(kum_afkast.index, kum_afkast[industry], label=f"Bottom: {industry}", linestyle='--', linewidth=2)

plt.title("Kumulativt dagligt afkast (1985–2025) – Top 3 vs. Bottom 3 industrier")
plt.xlabel("Date")
plt.ylabel("Kumulativt afkast")
plt.grid(True)
plt.legend()
plt.show()