import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Konstanter
start_date = "1985-01-01"
end_date = "2025-12-31"
Missing_values = [-99.99, -999]
PERCENT_TO_DECIMAL = 100

#Dataindsamling
def get_monthly_return() -> pd.DataFrame:
    df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/49_Industry_monthly.csv",
        sep=",",
        header=6
    )

    df = df.iloc[:1194]  # begræns periode
    df = df[df[df.columns[0]].astype(str).str.match(r"^\d{6}$", na=False)]
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date")

    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace(Missing_values, np.nan, inplace=True)
    df = df.sort_index()
    df = df.loc[start_date:end_date]
    df = df[~df.index.duplicated(keep="first")]

    return df

def get_daily_return() -> pd.DataFrame:
    df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/49_Industry_Portfolios_Daily.csv",
        sep=",",
        header=5,
        low_memory=False
    )

    midpoint = len(df) // 2
    df = df.iloc[:midpoint]  # tager kun value-weigthed daglig afkast

    df = df[df[df.columns[0]].astype(str).str.match(r"^\d{8}$", na=False)]  # Behold kun rækker med faktiske datoer

    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")  # konvertere til date-format
    df = df.set_index("Date")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # konvertere daglig afkast til numerisk

    df.replace(Missing_values, np.nan, inplace=True)  # fjerner missing values

    df = df.sort_index()
    df_clean = df.loc[start_date:end_date]
    df_clean = df_clean[~df_clean.index.duplicated(keep='first')]

    return df_clean

def get_monthly_risk() -> pd.DataFrame:
    rf_df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/Månedlig_rf.csv",
        sep=",",
        skiprows=3
    )
    rf_df = rf_df[rf_df.iloc[:, 0].astype(str).str.match(r"^\d{6,8}$", na=False)]
    rf_df = rf_df.rename(columns={rf_df.columns[0]: "Date"})
    rf_df["Date"] = pd.to_datetime(rf_df["Date"].astype(str), format="%Y%m")
    rf_df = rf_df.set_index("Date")

    rf_df["RF"] = pd.to_numeric(rf_df["RF"], errors="coerce") / PERCENT_TO_DECIMAL
    rf_df = rf_df.loc[start_date:end_date]

    return rf_df

#Konverter rf til daglig rf
def convert_monthly_rf_to_daily(rf_monthly: pd.DataFrame,
                                trading_dates: pd.DatetimeIndex) -> pd.DataFrame:

    rf_daily = rf_monthly.reindex(trading_dates, method='ffill')
    rf_daily["year_month"] = rf_daily.index.to_period("M")
    trading_days_per_month = rf_daily.groupby("year_month").size()
    rf_daily["trading_days"] = rf_daily["year_month"].map(trading_days_per_month)
    rf_daily["RF"] = (1 + rf_daily["RF"]) ** (1 / rf_daily["trading_days"]) - 1
    rf_daily = rf_daily.drop(columns=["year_month", "trading_days"])
    return rf_daily

#Beregning af merafkast
def calculate_daily_excess_returns(returns_df: pd.DataFrame,
                                  rf_daily: pd.DataFrame) -> pd.DataFrame:
    returns_decimal = returns_df / PERCENT_TO_DECIMAL
    excess = returns_decimal.subtract(rf_daily["RF"], axis=0)
    return excess

def calculate_monthly_excess_returns(returns_df: pd.DataFrame,
                                    rf_monthly: pd.DataFrame) -> pd.DataFrame:
    returns_decimal = returns_df / PERCENT_TO_DECIMAL
    excess = returns_decimal.subtract(rf_monthly["RF"], axis=0)
    return excess

#Kumulativt merafkast
def calculate_cumulative_returns(excess_returns: pd.DataFrame) -> pd.DataFrame:
    return (1 + excess_returns).cumprod() - 1

#Identifikation
def identify_top_bottom_industries(cumulative_returns: pd.DataFrame,
                                  n: int = 3) -> tuple:
    last_values = cumulative_returns.iloc[-1].dropna()
    top = last_values.sort_values(ascending=False).head(n).index.tolist()
    bottom = last_values.sort_values(ascending=True).head(n).index.tolist()
    return top, bottom

def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Beregn korrelationsmatrix for et DataFrame med afkast.
    """
    return returns_df.corr()


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, title: str) -> None:
    """
    Plot heatmap af korrelationsmatrix.
    """
    plt.figure(figsize=(12, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # masker øverste trekant
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=False,
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Korrelation"}
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Industrier", fontsize=12)
    plt.ylabel("Industrier", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


def calculate_correlation_matrix_and_average(returns_df: pd.DataFrame) -> tuple:
    """
    Beregn korrelationsmatrix og gennemsnitlig korrelation per industri.
    Returnerer:
        correlation_matrix: DataFrame med korrelationer
        avg_corr_series: Series med gennemsnitlig korrelation per industri
    """
    correlation_matrix = returns_df.corr()

    # Gennemsnitlig korrelation uden selv-korrelation
    avg_correlations = {}
    for industry in correlation_matrix.columns:
        other_corr = correlation_matrix[industry].drop(industry)  # drop diagonal
        avg_correlations[industry] = other_corr.mean()

    avg_corr_series = pd.Series(avg_correlations)

    return correlation_matrix, avg_corr_series


def display_top_bottom_correlations(avg_corr_series: pd.Series, n: int = 5, title: str = "") -> None:
    """
    Vis top n højeste og laveste gennemsnitskorrelationer.
    """
    print(f"\n{title}")
    print("-" * 60)

    print("\nTop {0} højeste gennemsnitskorrelationer:".format(n))
    for ind, val in avg_corr_series.sort_values(ascending=False).head(n).items():
        print(f"{ind}: {val:.3f}")

    print("\nTop {0} laveste gennemsnitskorrelationer:".format(n))
    for ind, val in avg_corr_series.sort_values(ascending=True).head(n).items():
        print(f"{ind}: {val:.3f}")

# Main
def main():
    # Indlæs data
    df_monthly = get_monthly_return()
    df_daily = get_daily_return()
    rf_monthly = get_monthly_risk()

    # Daglig RF fra månedlig
    rf_daily = convert_monthly_rf_to_daily(rf_monthly, df_daily.index)

    # Beregn daglig merafkast
    daily_excess = calculate_daily_excess_returns(df_daily, rf_daily)
    cum_daily = calculate_cumulative_returns(daily_excess)
    top_daily, bottom_daily = identify_top_bottom_industries(cum_daily)

    # Beregn månedlig merafkast
    monthly_excess = calculate_monthly_excess_returns(df_monthly, rf_monthly)
    cum_monthly = calculate_cumulative_returns(monthly_excess)
    top_monthly, bottom_monthly = identify_top_bottom_industries(cum_monthly)

    # Print resultater
    print("\n" + "="*60)
    print(f"Daglig merafkast ({start_date}-{end_date})")
    print("="*60)
    print("\nTop 3 (Daglig):")
    for i, ind in enumerate(top_daily, 1):
        print(f"{i}. {ind}: {cum_daily[ind].iloc[-1]:.2f}")
    print("\nBund 3 (Daglig):")
    for i, ind in enumerate(bottom_daily, 1):
        print(f"{i}. {ind}:{cum_daily[ind].iloc[-1]:.2f}")

    print("\n" + "="*60)
    print(f"Månedlig merafkast ({start_date}-{end_date})")
    print("="*60)
    print("\nTop 3 (Månedlig):")
    for i, ind in enumerate(top_monthly, 1):
        print(f"{i}. {ind}: {cum_monthly[ind].iloc[-1]:.2f}")
    print("\nBund 3 (Månedlig):")
    for i, ind in enumerate(bottom_monthly, 1):
        print(f"{i}. {ind}: {cum_monthly[ind].iloc[-1]:.2f}")

    # Beregn korrelation
    # Daglige
    daily_corr_matrix, daily_avg_corr = calculate_correlation_matrix_and_average(daily_excess)
    plot_correlation_heatmap(daily_corr_matrix, "Daglig korrelationsmatrix (1985-2025)")
    display_top_bottom_correlations(daily_avg_corr, n=5, title="Daglige afkast: Gennemsnitlige korrelationer")

    # Månedlige
    monthly_corr_matrix, monthly_avg_corr = calculate_correlation_matrix_and_average(monthly_excess)
    plot_correlation_heatmap(monthly_corr_matrix, "Månedlig korrelationsmatrix (1985-2025)")
    display_top_bottom_correlations(monthly_avg_corr, n=5, title="Månedlige afkast: Gennemsnitlige korrelationer")

if __name__ == "__main__":
    main()