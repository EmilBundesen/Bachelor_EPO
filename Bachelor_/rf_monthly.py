import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tkinter import Tk, filedialog

#konstanter
START_DATE = "1985-01-01"
END_DATE = "2025-12-31"
MISSING_VALUES = [-99.99, -999] #fra Kenneth French
PERCENT_TO_DECIMAL = 100
TOP_N_INDUSTRIES = 3
INDUSTRY_DATA_HEADER_ROW = 5 #fra Kenneth French
RF_DATA_SKIP_ROWS = 3 #fra Kenneth French

#Plot-konstanter
PLOT_WIDTH = 14
PLOT_HEIGHT = 7
LINE_WIDTH = 2


#Datarens
def get_file_path(prompt: str) -> Path:
    print(prompt)

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Åbn fil-dialog
    file_path = filedialog.askopenfilename(
        title=prompt,
        filetypes=[
            ("CSV filer", "*.csv"),
            ("Alle filer", "*.*")
        ]
    )

    root.destroy()
    if file_path:
        return Path(file_path)
    else:
        print("Ingen fil valgt.")
        exit()


def load_and_clean_industry_data(filepath: Path, start: str, end: str) -> pd.DataFrame:

    df = pd.read_csv(filepath, sep=",", header=INDUSTRY_DATA_HEADER_ROW)
    df = df[df[df.columns[0]].astype(str).str.match(r"^\d{8}$", na=False)]  #Behold kun rækker med faktiske datoer (8 cifre: YYYYMMDD)

    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d") #Konverter dato-kolonnen
    df = df.set_index("Date")

    for col in df.columns: # Konverter alle kolonner til numeriske værdier
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace(MISSING_VALUES, np.nan, inplace=True) #Erstat missing values med NaN

    df = df.sort_index() # Filtrér efter dato og fjern duplikater
    df_clean = df.loc[start:end] #Begrens tidsperiode
    df_clean = df_clean[~df_clean.index.duplicated(keep='first')]

    return df_clean


def load_and_clean_rf_data(filepath: Path, start: str, end: str) -> pd.DataFrame:

    rf_df = pd.read_csv(filepath, sep=",", skiprows=RF_DATA_SKIP_ROWS)
    rf_df = rf_df[rf_df.iloc[:, 0].astype(str).str.match(r"^\d{6,8}$", na=False)]# Behold kun rækker med gyldige datoer

    rf_df = rf_df.rename(columns={rf_df.columns[0]: "Date"})
    rf_df["Date"] = pd.to_datetime(rf_df["Date"].astype(str), format="%Y%m") #Konverter dato-kolonnen
    rf_df = rf_df.set_index("Date")

    rf_df["RF"] = pd.to_numeric(rf_df["RF"], errors="coerce") / PERCENT_TO_DECIMAL # Konverter RF til decimal (i rå format er det i % dvs. vi skal dele med 100)
    rf_df = rf_df.loc[start:end] # Begræns tidsperiode

    return rf_df


def convert_monthly_rf_to_daily(rf_monthly: pd.DataFrame,
                                trading_dates: pd.DatetimeIndex) -> pd.DataFrame:

    rf_daily = rf_monthly.reindex(trading_dates, method='ffill') #fylder daglig afkast med månedlig kast
    rf_daily["year_month"] = rf_daily.index.to_period("M")

    trading_days_per_month = rf_daily.groupby("year_month").size() #tæller antal handelsdage pr. måned

    rf_daily["trading_days"] = rf_daily["year_month"].map(trading_days_per_month) #mapper med ekstiterende df
    rf_daily["RF"] = (1 + rf_daily["RF"]) ** (1 / rf_daily["trading_days"]) - 1 #daglig rf

    rf_daily = rf_daily.drop(columns=["year_month", "trading_days"]) #renser for midlertidig kolonne

    return rf_daily


# beregning

def calculate_excess_returns(returns_df: pd.DataFrame,
                             rf_daily: pd.DataFrame) -> pd.DataFrame:

    daglig_afkast = returns_df / PERCENT_TO_DECIMAL  # Konverter til decimal
    daglig_merafkast = daglig_afkast.subtract(rf_daily["RF"], axis=0)

    return daglig_merafkast


def calculate_cumulative_returns(excess_returns: pd.DataFrame) -> pd.DataFrame:
    kum_merafkast = (1 + excess_returns).cumprod() - 1
    return kum_merafkast

def calculate_log_cumulative_returns(excess_returns: pd.DataFrame) -> pd.DataFrame:

    clipped_returns = (1 + excess_returns).clip(lower=1e-10) #sikre strengt positive beregninger
    log_kum_merafkast = np.log(clipped_returns).cumsum()
    return log_kum_merafkast


def identify_top_bottom_industries(cumulative_returns: pd.DataFrame,
                                   n: int = TOP_N_INDUSTRIES) -> tuple:
    sidste_dag_afkast = cumulative_returns.iloc[-1]
    top_industries = sidste_dag_afkast.sort_values(ascending=False).head(n).index
    bottom_industries = sidste_dag_afkast.sort_values(ascending=True).head(n).index

    return top_industries, bottom_industries


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> tuple:
    correlation_matrix = returns_df.corr()

    # Beregn gennemsnitlig korrelation for hver industri med alle andre
    avg_correlations = {}
    for industry in correlation_matrix.columns:
        # Tag alle korrelationer for denne industri
        other_correlations = correlation_matrix[industry].drop(industry)
        avg_correlations[industry] = other_correlations.mean()

    avg_correlation_series = pd.Series(avg_correlations)

    return correlation_matrix, avg_correlation_series


#visualisering

def plot_cumulative_returns(cumulative_returns: pd.DataFrame,
                            top_industries: pd.Index,
                            bottom_industries: pd.Index,
                            title: str = "Kumulativt dagligt merafkast (1985–2025)") -> None:

    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))


    for industry in top_industries:
        plt.plot(cumulative_returns.index, cumulative_returns[industry], label=f"Top: {industry}", linewidth=LINE_WIDTH)

    for industry in bottom_industries:
        plt.plot(cumulative_returns.index, cumulative_returns[industry], label=f"Bottom: {industry}", linestyle='--', linewidth=LINE_WIDTH)

    plt.title(f"{title} – Top {TOP_N_INDUSTRIES} vs. Bottom {TOP_N_INDUSTRIES} industrier")
    plt.xlabel("Date")
    plt.ylabel("Kumulativt merafkast")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 10))

    sns.heatmap(
        correlation_matrix,
        annot=False,  # Vis ikke værdier i hver celle (for mange industrier)
        cmap='RdYlGn',  # Rød-Gul-Grøn farveskala
        center=0,  # Center på 0
        vmin=-1,  # Minimum værdi
        vmax=1,  # Maximum værdi
        square=True,  # Firkantede celler
        linewidths=0.5,  # Linje mellem celler
        cbar_kws={"shrink": 0.8, "label": "Korrelation"}
    )

    plt.title("Korrelationsmatrix mellem 49 industrier (1985-2025)", fontsize=16, pad=20)
    plt.xlabel("Industrier", fontsize=12)
    plt.ylabel("Industrier", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


def print_summary_statistics(df_clean: pd.DataFrame,
                             rf_daily: pd.DataFrame,
                             top_industries: pd.Index,
                             bottom_industries: pd.Index,
                             cumulative_returns: pd.DataFrame,
                             log_cumulative_returns: pd.DataFrame,
                             avg_correlations: pd.Series) -> None:
    rows_in_industry = len(df_clean)
    rows_in_rf = len(rf_daily)

    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)
    if rows_in_industry == rows_in_rf:
        print("Ens antal rækker i industrier og rf")
    else:
        print("Fejl. Ikke end rækker")
        print(f"   Forskel: {abs(rows_in_industry - rows_in_rf)} rækker")

    print(f"Periode: {df_clean.index.min()} til {df_clean.index.max()}")

    print("\n" + "=" * 60)
    print("RESULTATER")
    print("=" * 60)
    print(f"\nTop {TOP_N_INDUSTRIES} industrier:")
    for i, industry in enumerate(top_industries, 1):
        final_return = cumulative_returns[industry].iloc[-1]
        print(f"  {i}. {industry}: {final_return:.2f}")

    print(f"\nDårligste {TOP_N_INDUSTRIES} industrier:")
    for i, industry in enumerate(bottom_industries, 1):
        final_return = cumulative_returns[industry].iloc[-1]
        print(f"  {i}. {industry}: {final_return:.2f}")

    print(f"\nLogkumulativt - Top {TOP_N_INDUSTRIES} værdier:")
    top_log = log_cumulative_returns.iloc[-1].sort_values(ascending=False).head(TOP_N_INDUSTRIES)
    for industry, value in top_log.items():
        print(f"  {industry}: {value:.4f}")

    print(f"\nLogkumulativt - Bottom {TOP_N_INDUSTRIES} værdier:")
    bottom_log = log_cumulative_returns.iloc[-1].sort_values(ascending=True).head(TOP_N_INDUSTRIES)
    for industry, value in bottom_log.items():
        print(f"  {industry}: {value:.4f}")

    print(f"\nGennemsnitlig korrelation for hver industri med alle andre:")
    print("-" * 60)
    # Sorter efter gennemsnitlig korrelation (højest først)
    sorted_correlations = avg_correlations.sort_values(ascending=False)
    for industry, avg_corr in sorted_correlations.items():
        print(f"  {industry}: {avg_corr:.4f}")

    print(f"\nOverordnet gennemsnit: {avg_correlations.mean():.4f}")
    print("=" * 60 + "\n")

# main

def main():
    """
        Workflow:
            1. Indlæser daglige data
            2. Indlæser risikofri rente (månedlige data)
            3. Konverterer RF til daglig frekvens
            4. Beregner merafkast, kumulativt afkast og korrelationer
            5. Identificerer top/bund performende industrier
            6. Visualiserer resultater
        """

    industry_file = get_file_path(
        "Indtast sti til industridata CSV (49_Industry_Portfolios_Daily.csv): "
    )
    rf_file = get_file_path(
        "Indtast sti til risikofri rente CSV (Månedlig_rf.csv): "
    )

    # Indlæs og rens data
    df_clean = load_and_clean_industry_data(industry_file, START_DATE, END_DATE)
    rf_monthly = load_and_clean_rf_data(rf_file, START_DATE, END_DATE)
    rf_daily = convert_monthly_rf_to_daily(rf_monthly, df_clean.index)

    # Beregn afkast
    daglig_merafkast = calculate_excess_returns(df_clean, rf_daily)
    kum_merafkast = calculate_cumulative_returns(daglig_merafkast)
    log_kum_merafkast = calculate_log_cumulative_returns(daglig_merafkast)

    # Beregn korrelation
    correlation_matrix, avg_correlations = calculate_correlation_matrix(daglig_merafkast)

    # Identificer top og bund industrier
    top3, bottom3 = identify_top_bottom_industries(kum_merafkast)

    # Print resultater
    print_summary_statistics(df_clean, rf_daily, top3, bottom3,
                             kum_merafkast, log_kum_merafkast, avg_correlations)

    # Visualiser
    plot_cumulative_returns(kum_merafkast, top3, bottom3,
                            "Kumulativt dagligt merafkast (1985–2025)")
    plot_cumulative_returns(log_kum_merafkast, top3, bottom3,
                            "Logkumulativt dagligt merafkast (1985–2025)")

    # Visualiser korrelation
    plot_correlation_heatmap(correlation_matrix)

if __name__ == "__main__":
    main()