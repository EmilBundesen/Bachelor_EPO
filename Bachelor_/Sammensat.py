import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konstanter
start_date = "1985-01-01"
end_date = "2025-12-31"
Missing_values = [-99.99, -999]
PERCENT_TO_DECIMAL = 100

# ── Dataindsamling ─────────────────────────────────────────────────────────────

def get_monthly_return() -> pd.DataFrame:
    df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/49_Industry_monthly.csv",
        sep=",", header=6
    )
    df = df.iloc[:1194]
    df = df[df[df.columns[0]].astype(str).str.match(r"^\d{6}$", na=False)]
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace(Missing_values, np.nan, inplace=True)
    df = df.sort_index().loc[start_date:end_date]
    df = df[~df.index.duplicated(keep="first")]
    return df

def get_daily_return() -> pd.DataFrame:
    df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/49_Industry_Portfolios_Daily.csv",
        sep=",", header=5, low_memory=False
    )
    midpoint = len(df) // 2
    df = df.iloc[:midpoint]
    df = df[df[df.columns[0]].astype(str).str.match(r"^\d{8}$", na=False)]
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    df = df.set_index("Date")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace(Missing_values, np.nan, inplace=True)
    df = df.sort_index().loc[start_date:end_date]
    df = df[~df.index.duplicated(keep="first")]
    return df

def get_monthly_risk() -> pd.DataFrame:
    rf_df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/Månedlig_rf.csv",
        sep=",", skiprows=3
    )
    rf_df = rf_df[rf_df.iloc[:, 0].astype(str).str.match(r"^\d{6,8}$", na=False)]
    rf_df = rf_df.rename(columns={rf_df.columns[0]: "Date"})
    rf_df["Date"] = pd.to_datetime(rf_df["Date"].astype(str), format="%Y%m")
    rf_df = rf_df.set_index("Date")
    rf_df["RF"] = pd.to_numeric(rf_df["RF"], errors="coerce") / PERCENT_TO_DECIMAL
    rf_df = rf_df.loc[start_date:end_date]
    return rf_df

def convert_monthly_rf_to_daily(rf_monthly: pd.DataFrame,
                                 trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    rf_daily = rf_monthly.reindex(trading_dates, method="ffill")
    rf_daily["year_month"] = rf_daily.index.to_period("M")
    trading_days_per_month = rf_daily.groupby("year_month").size()
    rf_daily["trading_days"] = rf_daily["year_month"].map(trading_days_per_month)
    rf_daily["RF"] = (1 + rf_daily["RF"]) ** (1 / rf_daily["trading_days"]) - 1
    rf_daily = rf_daily.drop(columns=["year_month", "trading_days"])
    return rf_daily

# ── Merafkast ──────────────────────────────────────────────────────────────────

def calculate_excess_returns(returns_df: pd.DataFrame,
                              rf: pd.DataFrame) -> pd.DataFrame:
    return (returns_df / PERCENT_TO_DECIMAL).subtract(rf["RF"], axis=0)

# ── 1) Korrelationstabel ───────────────────────────────────────────────────────

def print_correlation_table(corr_matrix: pd.DataFrame, title: str) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.2f}".format)
    print("\n" + "=" * 60)
    print(f"KORRELATIONSTABEL — {title}")
    print("=" * 60)
    print(corr_matrix.to_string())

# ── 2) Heatmap (nedre trekant) ─────────────────────────────────────────────────

def plot_lower_triangle_heatmap(corr_matrix: pd.DataFrame, title: str) -> None:
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # skjul øvre trekant + diagonal

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"shrink": 0.7, "label": "Korrelation"}
    )
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Industrier", fontsize=11)
    ax.set_ylabel("Industrier", fontsize=11)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()
    plt.show()

# ── 3) Top/bund korrelationstabel ──────────────────────────────────────────────

def print_top_bottom_correlation_table(corr_matrix: pd.DataFrame,
                                        title: str, n: int = 3) -> None:
    avg_corr = {
        col: corr_matrix[col].drop(col).mean()
        for col in corr_matrix.columns
    }
    avg_series = pd.Series(avg_corr).sort_values(ascending=False)

    top    = avg_series.head(n).reset_index()
    bottom = avg_series.tail(n).reset_index()
    top.columns    = ["Industri", "Gns. korrelation"]
    bottom.columns = ["Industri", "Gns. korrelation"]

    print("\n" + "=" * 45)
    print(f"TOP {n} HØJESTE — {title}")
    print("=" * 45)
    print(top.to_string(index=False, float_format="{:.3f}".format))

    print("\n" + "=" * 45)
    print(f"BUND {n} LAVESTE — {title}")
    print("=" * 45)
    print(bottom.to_string(index=False, float_format="{:.3f}".format))

# ── Samlet analyse per frekvens ────────────────────────────────────────────────

def run_correlation_analysis(excess_returns: pd.DataFrame,
                              label: str, n: int = 3) -> None:
    corr_matrix = excess_returns.corr()
    print_correlation_table(corr_matrix, label)
    plot_lower_triangle_heatmap(corr_matrix, f"Korrelationsmatrix — {label} (1985–2025)")
    print_top_bottom_correlation_table(corr_matrix, label, n=n)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Indlæs data
    df_monthly = get_monthly_return()
    df_daily   = get_daily_return()
    rf_monthly = get_monthly_risk()
    rf_daily   = convert_monthly_rf_to_daily(rf_monthly, df_daily.index)

    # Beregn merafkast
    monthly_excess = calculate_excess_returns(df_monthly, rf_monthly)
    daily_excess   = calculate_excess_returns(df_daily, rf_daily)

    # Kør analyse for begge frekvenser
    run_correlation_analysis(monthly_excess, label="Månedlig", n=3)
    run_correlation_analysis(daily_excess,   label="Daglig",   n=3)

if __name__ == "__main__":
    main()