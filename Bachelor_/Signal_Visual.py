import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Konstanter ────────────────────────────────────────────────────────────────
DATA_START_DATE    = "1984-01-01"   # hent data fra her (12 mdr. lookback-buffer)
PLOT_START_DATE    = "1985-01-01"   # vis signal fra her
END_DATE           = "2025-12-31"
MISSING_VALUES     = [-99.99, -999]
PERCENT_TO_DECIMAL = 100
LOOKBACK_MONTHS    = 12
TOP_N_VOLATILE     = 5


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



# ── Volatilitet (månedlige råafkast, annualiseret med sqrt(12)) ───────────────
def identify_top_volatile_industries(df_monthly: pd.DataFrame,
                                     n: int = TOP_N_VOLATILE) -> list:
    """
    Annualiseret volatilitet = std(månedlige råafkast) * sqrt(12).
    Rangerer over hele perioden og returnerer de n mest volatile industrier.
    """
    annual_vol = (df_monthly / PERCENT_TO_DECIMAL).std() * np.sqrt(12)
    top_volatile = annual_vol.sort_values(ascending=False).head(n).index.tolist()

    print(f"\nTop {n} mest volatile industrier (månedlig, annualiseret, {PLOT_START_DATE}–{END_DATE}):")
    print("-" * 50)
    for i, ind in enumerate(top_volatile, 1):
        print(f"  {i:2d}. {ind:<20s}  σ = {annual_vol[ind]:.4f}")

    return top_volatile


# ── XS-momentum signal ────────────────────────────────────────────────────────
def compute_xsmom(monthly_ret: pd.DataFrame,
                  lookback: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """
    Cross-sectional momentum signal (Eq. 24–25).
    Rullende sum over `lookback` måneder → cross-sectional demean → skalering.
    c_t = 1 / sum_positive  (= 1 / |sum_negative| da cross-sectional sum = 0)
    """
    roll = (monthly_ret
            .rolling(window=lookback, min_periods=lookback)
            .sum())

    out = pd.DataFrame(np.nan, index=roll.index, columns=roll.columns)

    for date, row in roll.iterrows():
        avail = row.dropna()
        if len(avail) < 2:
            continue

        demeaned = avail - avail.mean()

        pos = demeaned[demeaned > 0].sum()
        if pos == 0:
            continue

        c_t = 1.0 / pos   # Eq. 25
        out.loc[date, demeaned.index] = c_t * demeaned

    return out


# ── Visualisering ─────────────────────────────────────────────────────────────
def plot_signal(xsmom_signal: pd.DataFrame,
                industries: list) -> None:
    """
    Enkelt panel: XS-momentum signal over tid for de 6 mest volatile industrier.
    """
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
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("/Users/emilbundesen/Desktop/Bachelor/Signal_EQ_1.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df_monthly = get_monthly_return()
    monthly_raw = df_monthly / PERCENT_TO_DECIMAL

    top_volatile = identify_top_volatile_industries(df_monthly, n=TOP_N_VOLATILE)


    xsmom_signal = compute_xsmom(monthly_raw, lookback=LOOKBACK_MONTHS)

    plot_signal(xsmom_signal, top_volatile)

if __name__ == "__main__":
    main()