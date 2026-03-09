import pandas as pd
import numpy as np

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