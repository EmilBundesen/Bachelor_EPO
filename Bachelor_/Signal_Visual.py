import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Konstanter ────────────────────────────────────────────────────────────────
DATA_START_DATE    = "1984-01-01"
PLOT_START_DATE    = "1985-01-01"
END_DATE           = "2025-12-31"
MISSING_VALUES     = [-99.99, -999]
PERCENT_TO_DECIMAL = 100
LOOKBACK_MONTHS    = 12
TOP_N_VOLATILE     = 10
VOL_WINDOW         = 6  # måneder til rullende volatilitet


# ── Datahentning ──────────────────────────────────────────────────────────────
def get_monthly_return() -> pd.DataFrame:
    df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/49_Industry_monthly.csv",
        sep=",",
        header=6
    )
    df = df.iloc[:1194]
    df = df[df[df.columns[0]].astype(str).str.match(r"^\d{6}$", na=False)]
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date")
    df.columns = df.columns.str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace(MISSING_VALUES, np.nan, inplace=True)
    df = df.sort_index().loc[DATA_START_DATE:END_DATE]
    df = df[~df.index.duplicated(keep="first")]
    return df

def get_monthly_risk() -> pd.DataFrame:
    rf_df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/Månedlig_rf.csv",
        sep=",",
        skiprows=3
    )
    rf_df = rf_df[rf_df.iloc[:, 0].astype(str).str.match(r"^\d{6,8}$", na=False)]
    rf_df = rf_df.rename(columns={rf_df.columns[0]: "Date"})
    rf_df["Date"] = pd.to_datetime(rf_df["Date"].astype(str), format="%Y%m")
    rf_df = rf_df.set_index("Date")
    rf_df["RF"] = pd.to_numeric(rf_df["RF"], errors="coerce") / PERCENT_TO_DECIMAL
    rf_df = rf_df.loc[DATA_START_DATE:END_DATE]
    return rf_df


# ── Merafkast ─────────────────────────────────────────────────────────────────
def calculate_monthly_excess_returns(returns_df: pd.DataFrame,
                                     rf_monthly: pd.DataFrame) -> pd.DataFrame:
    returns_decimal = returns_df / PERCENT_TO_DECIMAL
    excess = returns_decimal.subtract(rf_monthly["RF"], axis=0)
    return excess


# ── Volatilitet (fuld historik, til rangering) ────────────────────────────────
def identify_top_volatile_industries(df_monthly: pd.DataFrame,
                                     n: int = TOP_N_VOLATILE) -> list:
    annual_vol = df_monthly.std() * np.sqrt(12)
    obs_count = df_monthly.notna().sum()
    annual_vol = annual_vol[obs_count >= 12]
    top_volatile = annual_vol.sort_values(ascending=False).head(n).index.tolist()

    print(f"\nTop {n} mest volatile industrier (fuld historik, råafkast %):")
    print("-" * 50)
    for i, ind in enumerate(top_volatile, 1):
        print(f"  {i:2d}. {ind:<20s}  σ = {annual_vol[ind]:.2f}% annualiseret")

    return top_volatile


# ── Rullende volatilitet (annualiseret) ───────────────────────────────────────
def compute_rolling_volatility(df_monthly: pd.DataFrame,
                                window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    Beregner rullende annualiseret volatilitet (std * sqrt(12))
    på månedlige råafkast i procent.
    """
    rolling_vol = (
        df_monthly
        .rolling(window=window, min_periods=window)
        .std()
        * np.sqrt(12)
    )
    return rolling_vol


# ── XS-momentum signal ────────────────────────────────────────────────────────
def compute_xsmom(monthly_excess: pd.DataFrame,
                  lookback: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    roll = (monthly_excess
            .rolling(window=lookback, min_periods=lookback)
            .sum())

    out = pd.DataFrame(np.nan, index=roll.index, columns=roll.columns)

    for date, row in roll.iterrows():
        avail = row.dropna()
        if len(avail) < 2:
            continue
        demeaned = avail - avail.mean()
        pos = demeaned[demeaned > 0].sum()
        neg = demeaned[demeaned < 0].sum()
        if pos == 0 or neg == 0:
            continue
        c_t = 1.0 / max(pos, abs(neg))
        out.loc[date, demeaned.index] = c_t * demeaned

    return out


# ── Hjælpefunktion: fælles aksestil ──────────────────────────────────────────
def _style_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)


# ── Plot 1: Rullende volatilitet ──────────────────────────────────────────────
def plot_rolling_volatility(df_monthly: pd.DataFrame,
                             industries: list,
                             window: int = VOL_WINDOW) -> None:
    rolling_vol = compute_rolling_volatility(df_monthly, window)
    vol_subset  = rolling_vol[industries].loc[PLOT_START_DATE:]

    fig, ax = plt.subplots(figsize=(16, 6))

    for ind in industries:
        ax.plot(vol_subset.index, vol_subset[ind], linewidth=1.2, label=ind)

    ax.set_title(
        f"Rullende annualiseret volatilitet ({window}-måneders vindue) "
        f"— top {TOP_N_VOLATILE} mest volatile industrier\n"
        f"{PLOT_START_DATE} til {END_DATE}",
        fontsize=13, pad=12
    )
    ax.set_ylabel(f"{window}-m rullende σ (%, annualiseret)", fontsize=11)
    ax.set_xlabel("Dato", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.7)
    _style_time_axis(ax)

    plt.tight_layout()
    plt.savefig(
        "/Users/emilbundesen/Desktop/Bachelor/Rolling_Vol.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()


# ── Plot 2: XS-momentum signal ────────────────────────────────────────────────
def plot_signal(xsmom_signal: pd.DataFrame,
                industries: list) -> None:
    signal_subset = xsmom_signal[industries].loc[PLOT_START_DATE:]

    fig, ax = plt.subplots(figsize=(16, 6))

    for ind in industries:
        ax.plot(signal_subset.index, signal_subset[ind], linewidth=1.2, label=ind)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"XS-momentum signal — top {TOP_N_VOLATILE} mest volatile industrier\n"
        f"{PLOT_START_DATE} til {END_DATE}",
        fontsize=13, pad=12
    )
    ax.set_ylabel("XS-momentum signal ($w_t$)", fontsize=11)
    ax.set_xlabel("Dato", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.7)
    _style_time_axis(ax)

    plt.tight_layout()
    plt.savefig(
        "/Users/emilbundesen/Desktop/Bachelor/Signal_EQ_1.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    monthly_ret    = get_monthly_return()
    rf             = get_monthly_risk()
    monthly_excess = calculate_monthly_excess_returns(monthly_ret, rf)

    top_volatile   = identify_top_volatile_industries(monthly_ret, n=TOP_N_VOLATILE)
    xsmom          = compute_xsmom(monthly_excess, LOOKBACK_MONTHS)

    # Plot 1: rullende volatilitet
    plot_rolling_volatility(monthly_ret, top_volatile, window=VOL_WINDOW)

    # Plot 2: XS-momentum signal
    plot_signal(xsmom, top_volatile)

if __name__ == "__main__":
    main()