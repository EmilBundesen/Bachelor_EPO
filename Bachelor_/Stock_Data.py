import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Equity_1 import (
    compute_risk_model,
    get_monthly_risk,
    build_epo_panel,
    build_dynamic_oos_epo,
    backtest_mvo_no_shrink,
    performance_summary,
    CANDIDATE_WS,
    GAMMA,
    RISK_WINDOW,
    CORR_PRESHRINK,
    LOOKBACK_MONTHS,
    MIN_HISTORY_OOS,
    backtest_strategy,
    epo_weights,
)

# ── Konfiguration ─────────────────────────────────────────────
sectors = {
    "pharma":   ["PFE","MRK","BMY","JNJ","AMGN","ABT","BAX","LLY",
                 "GILD","MDT","CAH","MCK","CI","UNH","BDX",
                 "SYK","ZBH","HUM","DVA","WST"],
    "defense":  ["LMT","NOC","RTX","GD","BA","TXT","LHX","HII",
                 "CW","TDG","HEI","BWXT","LDOS","SAIC","CACI",
                 "HON","GE","MMM","EMR","ETN"],
    "energy":   ["XOM","CVX","COP","EOG","OXY","DVN","APA","SLB",
                 "HAL","BKR","KMI","WMB","OKE","PSX","VLO",
                 "MPC","SUN","FANG"],
    "tech":     ["MSFT","INTC","IBM","ORCL","CSCO","QCOM","TXN","ADI",
                 "AMAT","LRCX","KLAC","HPQ","ADBE","SAP","ACN",
                 "MU","NXPI","AVGO","CRM","DELL"],
    "consumer": ["KO","PEP","PG","CL","KMB","GIS","MDLZ","UL",
                 "PM","MO","BTI","DEO","BUD","STZ","BF-B",
                 "HRL","CPB","SJM","HSY","MKC"],
}

START_DATE        = "1990-01-01"
END_DATE          = "2025-12-31"
BACKTEST_START    = "2000-01-01"
MAX_NAN_THRESHOLD = 0.20
N_LONG_SHORT      = 5


# ── Datahentning ──────────────────────────────────────────────

def get_rf_monthly(start: str, end: str) -> pd.Series:
    rf_df = get_monthly_risk()
    rf_df.index = rf_df.index.to_period("M").to_timestamp("M")
    rf = rf_df["RF"].loc[start:end].dropna()
    rf.name = "RF"
    return rf


def get_stock_returns(sectors, start, end, max_nan=MAX_NAN_THRESHOLD):
    all_returns, all_prices, ticker_to_sector = [], [], {}
    for sector, tickers in sectors.items():
        print(f"Henter {sector}...")
        data = yf.download(tickers, start=start, end=end,
                           auto_adjust=True, threads=False,
                           progress=True)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        rets = data.pct_change(fill_method=None)
        valid = rets.columns[rets.isna().mean() <= max_nan].tolist()
        rets  = rets[valid]
        data  = data[valid]
        for t in valid:
            ticker_to_sector[t] = sector
        all_returns.append(rets)
        all_prices.append(data)          # ← gem råpriser separat

    daily_rets   = pd.concat(all_returns, axis=1)
    daily_prices = pd.concat(all_prices,  axis=1)
    daily_rets.index   = pd.to_datetime(daily_rets.index)
    daily_prices.index = pd.to_datetime(daily_prices.index)
    return daily_rets, daily_prices, ticker_to_sector


def to_monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = (1 + daily).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M").to_timestamp("M")
    return monthly


def compute_monthly_excess(monthly, rf):
    return monthly.sub(rf.reindex(monthly.index).ffill(), axis=0)


def subset(df, start, end):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return df.loc[(df.index >= s) & (df.index <= e)]


# ── Signal ────────────────────────────────────────────────────

def compute_combined_signal(monthly_ret, ticker_to_sector,
                             lookback=LOOKBACK_MONTHS):
    """
    To-lags momentum signal:
      Lag 1 — Sektor-TSMOM: fortegn på sektorens 12m afkast
      Lag 2 — Intra-sektor XSMOM (Eq. 24-25): rang inden for sektor
    Skalering: unit-leverage per sektor, 1/n_sektorer på tværs.
    """
    roll = monthly_ret.rolling(window=lookback, min_periods=lookback).sum()
    sector_groups = {}
    for ticker, sector in ticker_to_sector.items():
        sector_groups.setdefault(sector, []).append(ticker)

    out = pd.DataFrame(np.nan, index=roll.index, columns=roll.columns)
    n_sectors = len(sector_groups)

    for date, row in roll.iterrows():
        sector_signals = {}
        for sector, tickers in sector_groups.items():
            avail = row[tickers].dropna()
            if len(avail) < 2:
                continue
            direction = np.sign(avail.mean())
            if direction == 0:
                continue
            demeaned = avail - avail.mean()
            pos = demeaned[demeaned > 0].sum()
            neg = demeaned[demeaned < 0].sum()
            if pos == 0 or neg == 0:
                continue
            c_t = 1.0 / max(pos, abs(neg))
            sector_signals[sector] = direction * c_t * demeaned
        for sector, sig in sector_signals.items():
            out.loc[date, sig.index] = sig.values / n_sectors

    return out


# ── Benchmarks ────────────────────────────────────────────────

def backtest_vol_scaled(monthly_excess, xsmom, vols_dict):
    """Vol-skaleret anchor = EPO^s(w=1): vægt_i = signal_i / σ_i²"""
    risk_dates = set(vols_dict)
    sig_dates  = set(xsmom.index)

    def weight_fn(date):
        if date not in risk_dates or date not in sig_dates:
            return pd.Series(dtype=float)
        sig = xsmom.loc[date].dropna()
        v   = vols_dict[date].reindex(sig.index).dropna()
        common = sig.reindex(v.index).dropna()
        v = v.reindex(common.index)
        raw_w = common / (v ** 2)
        pos = raw_w[raw_w > 0].sum()
        neg = raw_w[raw_w < 0].sum()
        if pos == 0 or neg == 0:
            return pd.Series(dtype=float)
        return raw_w / max(pos, abs(neg))

    return backtest_strategy(monthly_excess, weight_fn,
                             name="Vol-Scaled (INDMOM)")


def backtest_simple_momentum(monthly_excess, monthly_ret,
                              n=N_LONG_SHORT, lookback=LOOKBACK_MONTHS):
    """
    Simpel momentum: køb de n aktier med højest 12m råafkast,
    sælg de n med lavest. Equal-weighted, unit-leverage.
    """
    roll = monthly_ret.rolling(window=lookback, min_periods=lookback).sum()
    idx  = monthly_excess.index
    rets, dates = [], []

    for t in range(len(idx) - 1):
        date      = idx[t]
        next_date = idx[t + 1]
        momentum  = roll.loc[date].dropna()
        if len(momentum) < n * 2:
            continue
        long_tickers  = momentum.nlargest(n).index
        short_tickers = momentum.nsmallest(n).index
        r_long  = monthly_excess.loc[next_date, long_tickers].dropna()
        r_short = monthly_excess.loc[next_date, short_tickers].dropna()
        if len(r_long) == 0 or len(r_short) == 0:
            continue
        rets.append(r_long.mean() - r_short.mean())
        dates.append(next_date)

    return pd.Series(rets, index=dates,
                     name=f"Simple Momentum (L{n}/S{n})")


# ── Terminal output ───────────────────────────────────────────

def print_top_bottom_stocks(monthly_excess, ticker_to_sector, end_date):
    """Top-5 og bottom-5 aktier på kumuleret afkast over hele perioden."""
    e    = pd.to_datetime(end_date)
    full = monthly_excess.loc[:e]
    cum_per_stock = (1 + full).prod() - 1

    top5 = cum_per_stock.nlargest(5)
    bot5 = cum_per_stock.nsmallest(5)

    print("\n" + "=" * 55)
    print("TOP 5 AKTIER — kumuleret afkast (hele periode)")
    print("=" * 55)
    print(f"  {'Aktie':<8} {'Sektor':<12} {'Kum. afkast':>12}")
    print("-" * 55)
    for ticker, val in top5.items():
        print(f"  {ticker:<8} {ticker_to_sector.get(ticker,''):<12} {val:>11.1%}")

    print("\n" + "=" * 55)
    print("BOTTOM 5 AKTIER — kumuleret afkast (hele periode)")
    print("=" * 55)
    print(f"  {'Aktie':<8} {'Sektor':<12} {'Kum. afkast':>12}")
    print("-" * 55)
    for ticker, val in bot5.items():
        print(f"  {ticker:<8} {ticker_to_sector.get(ticker,''):<12} {val:>11.1%}")


def print_top_bottom_correlations(daily, ticker_to_sector, n=10):
    """
    Beregner GENNEMSNITLIG korrelation per aktie (dvs. gns. af alle
    parvis korrelationer for den pågældende aktie mod alle andre).
    Udskriver de n aktier med højest og lavest gennemsnitlig korrelation.

    Negative gennemsnitlige korrelationer er matematisk næsten umulige
    for aktier — hvis de opstår er det et tegn på meget kort overlappende
    historik. Derfor filtreres par med < 252 fælles observationer fra.
    """
    daily_rets = daily.pct_change(fill_method=None)
    tickers    = daily_rets.columns.tolist()

    # Beregn korrelationsmatrix kun på overlappende observationer
    # (pairwise complete observations)
    corr_matrix = daily_rets.corr(min_periods=252)

    # Gennemsnitlig korrelation per aktie — ekskluder diagonal (=1)
    # og NaN-par (utilstrækkelig historik)
    np.fill_diagonal(corr_matrix.values, np.nan)
    mean_corr = corr_matrix.mean(axis=1).dropna().sort_values(ascending=False)

    print("\n" + "=" * 65)
    print(f"TOP {n} AKTIER — højest gns. korrelation mod alle andre")
    print("=" * 65)
    print(f"  {'Aktie':<8} {'Sektor':<12} {'Gns. korrelation':>18}")
    print("-" * 65)
    for ticker, val in mean_corr.head(n).items():
        print(f"  {ticker:<8} {ticker_to_sector.get(ticker,''):<12} {val:>18.4f}")

    print("\n" + "=" * 65)
    print(f"TOP {n} AKTIER — lavest gns. korrelation mod alle andre")
    print("=" * 65)
    print(f"  {'Aktie':<8} {'Sektor':<12} {'Gns. korrelation':>18}")
    print("-" * 65)
    for ticker, val in mean_corr.tail(n).iloc[::-1].items():
        print(f"  {ticker:<8} {ticker_to_sector.get(ticker,''):<12} {val:>18.4f}")


# ── Visualiseringer ───────────────────────────────────────────

def plot_visualizations(monthly_excess, daily, ticker_to_sector,
                        end_date):
    """
    Plot 1: Kumuleret afkast — top-5 og bottom-5 aktier
    Plot 2: Kumuleret afkast pr. sektor
    """
    sns.set_theme(style="whitegrid", palette="tab10")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    e    = pd.to_datetime(end_date)
    full = monthly_excess.loc[:e]

    def cum(series):
        r = series.dropna()
        return (1 + r).cumprod() - 1

    sector_groups: dict[str, list] = {}
    for ticker, sector in ticker_to_sector.items():
        sector_groups.setdefault(sector, []).append(ticker)

    cum_per_stock = (1 + full).prod() - 1
    top5 = cum_per_stock.nlargest(5)
    bot5 = cum_per_stock.nsmallest(5)

    # ── Plot 1: Top-5 og bottom-5 aktier ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, tickers, title in [
        (axes[0], top5.index, "Top 5 aktier"),
        (axes[1], bot5.index, "Bottom 5 aktier"),
    ]:
        for ticker in tickers:
            series = full[ticker].dropna()
            ax.plot(cum(series),
                    label=f"{ticker} ({ticker_to_sector.get(ticker,'')})",
                    linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Kumuleret afkast")
        ax.set_xlabel("Dato")
        ax.legend(fontsize=8)
    plt.suptitle("Kumuleret afkast — top og bottom aktier (hele periode)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # ── Plot 2: Kumuleret afkast pr. sektor ───────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for sector, tickers in sector_groups.items():
        available = [t for t in tickers if t in full.columns]
        if not available:
            continue
        sector_ret = full[available].mean(axis=1)
        ax.plot(cum(sector_ret), label=sector.capitalize(), linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Kumuleret afkast pr. sektor (hele periode, equal-weighted)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Kumuleret afkast")
    ax.set_xlabel("Dato")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("EPO — Yahoo Finance enkeltaktier")
    print(f"Data: {START_DATE} → {END_DATE}")
    print(f"Backtest OOS fra: {BACKTEST_START}")
    print("=" * 65)

    # 1. Data
    daily, daily_prices, ticker_to_sector = get_stock_returns(
        sectors, START_DATE, END_DATE)
    monthly        = to_monthly_returns(daily)
    rf             = get_rf_monthly(START_DATE, END_DATE)
    monthly_excess = compute_monthly_excess(monthly, rf)
    print(f"\nAktier: {monthly_excess.shape[1]}  |  "
          f"Periode: {monthly_excess.index[0].date()} → "
          f"{monthly_excess.index[-1].date()}")

    # 2. Signal
    print("\nBeregner signal...")
    xsmom = compute_combined_signal(monthly_excess, ticker_to_sector,
                                    LOOKBACK_MONTHS)

    # 3. Risikomodel
    print("\nBygger risikomodel (60m, 5% pre-shrinkage)...")
    corr_shrunk, vols = compute_risk_model(
        monthly_excess, window=RISK_WINDOW,
        theta=CORR_PRESHRINK, verbose=True)

    print("Bygger unshrunk risikomodel til MVO...")
    corr_raw, vols_raw = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=0.0, verbose=False)

    # 4. Benchmarks
    print("\nBacktester benchmarks...")
    indmom_full = backtest_vol_scaled(monthly_excess, xsmom, vols)
    mvo_full    = backtest_mvo_no_shrink(
        monthly_excess, xsmom, corr_raw, vols_raw, GAMMA)
    mom_full    = backtest_simple_momentum(monthly_excess, monthly,
                                           n=N_LONG_SHORT)

    # 5. EPO
    print("\nBygger EPO panel...")
    epo_panel = build_epo_panel(
        monthly_excess, xsmom, corr_shrunk, vols, GAMMA, CANDIDATE_WS)
    print("\nBygger dynamisk OOS EPO...")
    epo_dyn = build_dynamic_oos_epo(
        epo_panel, oos_start=BACKTEST_START, min_history=MIN_HISTORY_OOS)

    # 6. Performance tabel
    s, e = BACKTEST_START, END_DATE
    rows = [
        performance_summary(subset(indmom_full, s, e),
                            "Vol-Scaled / INDMOM (anchor)"),
        performance_summary(subset(mvo_full,    s, e),
                            "MVO (no shrinkage, θ=0)"),
        performance_summary(subset(mom_full,    s, e),
                            f"Simple Momentum (L{N_LONG_SHORT}/S{N_LONG_SHORT})"),
        performance_summary(subset(epo_dyn,     s, e),
                            "EPO: out-of-sample"),
    ]
    w_labels = {
        0.00: "EPO w=0%   (MVO + 5% pre-shrink)",
        0.10: "EPO w=10%", 0.25: "EPO w=25%",
        0.50: "EPO w=50%", 0.75: "EPO w=75%",
        0.90: "EPO w=90%", 0.99: "EPO w=99%",
        1.00: "EPO w=100%  (= Vol-Scaled anchor)",
    }
    panel_oos = subset(epo_panel, s, e)
    for w in CANDIDATE_WS:
        col = f"EPO_w_{w:.2f}"
        if col in panel_oos.columns:
            rows.append(performance_summary(panel_oos[col], w_labels[w]))

    perf = pd.DataFrame(rows).set_index("Strategy")

    print("\n" + "=" * 65)
    print(f"PERFORMANCE — OOS: {BACKTEST_START} → {END_DATE}")
    print("=" * 65)
    print(perf.to_string())

    perf.to_csv("performance_summary.csv")
    print("\nGemt: performance_summary.csv")

    # 7. Terminal statistik
    print_top_bottom_stocks(monthly_excess, ticker_to_sector, END_DATE)
    print_top_bottom_correlations(daily_prices, ticker_to_sector, n=10)

    # 8. Visualiseringer
    print("\nGenererer visualiseringer...")
    plot_visualizations(
        monthly_excess   = monthly_excess,
        daily            = daily,
        ticker_to_sector = ticker_to_sector,
        end_date         = END_DATE,
    )

    return {
        "monthly_excess":   monthly_excess,
        "xsmom":            xsmom,
        "epo_panel":        epo_panel,
        "epo_dyn_oos":      epo_dyn,
        "performance":      perf,
        "ticker_to_sector": ticker_to_sector,
    }


if __name__ == "__main__":
    results = main()