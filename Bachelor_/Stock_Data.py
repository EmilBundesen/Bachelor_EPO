import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    get_weights_at_date
)

from Get_stock_data import DAILY_RETS_PATH, DAILY_PRICES_PATH, SECTOR_PATH

from Signal_Visual import (
    plot_turnover,
    compute_turnover_series
)

START_DATE        = "2015-01-01"
END_DATE          = "2025-12-31"
BACKTEST_START    = "2020-01-01"
MAX_NAN_THRESHOLD = 0.20
N_LONG_SHORT      = 20


# ── Datahentning ──────────────────────────────────────────────
def load_data():
    """
    Indlæser gemt datasæt fra Data_Fetch.py.
    Kør Data_Fetch.py først hvis filerne ikke findes.
    """
    if not all(os.path.exists(p) for p in
               [DAILY_RETS_PATH, DAILY_PRICES_PATH, SECTOR_PATH]):
        raise FileNotFoundError(
            "Data ikke fundet — kør Data_Fetch.py først."
        )

    daily        = pd.read_parquet(DAILY_RETS_PATH)
    daily_prices = pd.read_parquet(DAILY_PRICES_PATH)
    sec_df       = pd.read_csv(SECTOR_PATH)
    ticker_to_sector = dict(zip(sec_df["ticker"], sec_df["sector"]))

    print(f"Data indlæst: {daily.shape[1]} aktier  |  "
          f"{daily.index[0].date()} → {daily.index[-1].date()}")

    return daily, daily_prices, ticker_to_sector

def get_rf_monthly(start: str, end: str) -> pd.Series:
    rf_df = get_monthly_risk()
    rf_df.index = rf_df.index.to_period("M").to_timestamp("M")
    rf = rf_df["RF"].loc[start:end].dropna()
    rf.name = "RF"
    return rf


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

def compute_tsmom_signal(monthly_ret, ticker_to_sector,
                          lookback=LOOKBACK_MONTHS):
    """
    TSMOM per aktie med asymmetrisk short-filter:
      - Long:  12m afkast > 0  → signal = +afkast / n
      - Short: 12m afkast < 0  → signal = +afkast / n  (negativt tal)
      - Nul:   12m afkast = 0  → ingen position

    Short-siden aktiveres KUN hvis aktien har negativt absolut afkast
    — ikke blot relativt dårlig inden for universet.
    Dermed shortes der ikke aktier der blot stiger langsommere end andre.
    """
    roll = monthly_ret.rolling(window=lookback, min_periods=lookback).sum()
    out  = pd.DataFrame(np.nan, index=roll.index, columns=roll.columns)

    for date, row in roll.iterrows():
        avail = row.dropna()
        if len(avail) < 2:
            continue

        # Long: positivt 12m afkast
        long_mask  = avail > 0
        # Short: negativt absolut 12m afkast (aktien har rent faktisk tabt)
        short_mask = avail < 0

        signal = pd.Series(0.0, index=avail.index)
        signal[long_mask]  =  avail[long_mask]   # positiv vægt
        signal[short_mask] =  avail[short_mask]  # negativ vægt (afkast er < 0)

        n = len(avail)
        out.loc[date, signal.index] = signal.values / n

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

    roll = (1 + monthly_ret).rolling(lookback).apply(np.prod, raw=True) - 1

    idx  = monthly_excess.index
    rets, dates = [], []

    for t in range(len(idx) - 1):
        date      = idx[t]
        next_date = idx[t + 1]

        momentum = roll.loc[date].dropna()
        if len(momentum) < n * 2:
            continue

        long = momentum.nlargest(n)
        short = momentum.nsmallest(n)

        r = monthly_excess.loc[next_date]

        weights = pd.Series(0.0, index=r.index)
        weights[long.index]  =  1.0 / n
        weights[short.index] = -1.0 / n

        aligned = pd.concat([weights, r], axis=1).dropna()
        if aligned.empty:
            continue

        ret = (aligned.iloc[:, 0] * aligned.iloc[:, 1]).sum()

        rets.append(ret)
        dates.append(next_date)

    return pd.Series(rets, index=dates,
                     name=f"Simple Momentum (L{n}/S{n})")

def backtest_simple_momentum_ew(monthly_excess, monthly_ret,
                               lookback=LOOKBACK_MONTHS):

    roll = (1 + monthly_ret).rolling(lookback).apply(np.prod, raw=True) - 1
    idx  = monthly_excess.index
    rets, dates = [], []

    for t in range(len(idx) - 1):
        date      = idx[t]
        next_date = idx[t + 1]
        momentum  = roll.loc[date].dropna()

        long_tickers  = momentum[momentum > 0].index
        short_tickers = momentum[momentum < 0].index

        if len(long_tickers) == 0 and len(short_tickers) == 0:
            continue

        r = monthly_excess.loc[next_date]

        weights = pd.Series(0.0, index=r.index)

        if len(long_tickers) > 0:
            weights[long_tickers] = 1.0 / len(long_tickers)

        if len(short_tickers) > 0:
            weights[short_tickers] = -1.0 / len(short_tickers)

        # 🔥 NORMALISÉR TIL UNIT LEVERAGE
        weights = weights / weights.abs().sum()

        aligned = pd.concat([weights, r], axis=1).dropna()
        if aligned.empty:
            continue

        ret = (aligned.iloc[:, 0] * aligned.iloc[:, 1]).sum()

        rets.append(ret)
        dates.append(next_date)

    return pd.Series(rets, index=dates, name="TSMOM EW (fixed)")

def compute_equal_weight_benchmark(monthly_excess):
    """
    1/N benchmark:
    - Lige vægt i alle tilgængelige aktier hver måned
    - Rebalanceres månedligt
    """

    rets = []
    dates = []

    for date in monthly_excess.index:
        r = monthly_excess.loc[date].dropna()

        if len(r) == 0:
            continue

        weights = pd.Series(1.0 / len(r), index=r.index)

        port_ret = (weights * r).sum()

        rets.append(port_ret)
        dates.append(date)

    return pd.Series(rets, index=dates, name="Equal Weight (1/N)")

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


# ── Gross Exposure ────────────────────────────────────────────

def compute_ge_epo_fixed_w(monthly_excess, xsmom, corr_dict, vols_dict,
                            gamma, w, start, end):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    idx = monthly_excess.loc[s:e].index
    risk_dates = set(corr_dict)
    sig_dates  = set(xsmom.index)
    ge, dates  = [], []
    for date in idx:
        if date not in risk_dates or date not in sig_dates:
            continue
        wts = epo_weights(xsmom.loc[date], corr_dict[date],
                          vols_dict[date], gamma, w)
        if len(wts) == 0:
            continue
        ge.append(wts.abs().sum())
        dates.append(date)
    return pd.Series(ge, index=dates)


def compute_ge_indmom(monthly_excess, xsmom, vols_dict, start, end):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    idx = monthly_excess.loc[s:e].index
    risk_dates = set(vols_dict)
    sig_dates  = set(xsmom.index)
    ge, dates  = [], []
    for date in idx:
        if date not in risk_dates or date not in sig_dates:
            continue
        sig = xsmom.loc[date].dropna()
        v   = vols_dict[date].reindex(sig.index).dropna()
        common = sig.reindex(v.index).dropna()
        v = v.reindex(common.index)
        raw_w = common / (v ** 2)
        pos = raw_w[raw_w > 0].sum()
        neg = raw_w[raw_w < 0].sum()
        if pos == 0 or neg == 0:
            continue
        wts = raw_w / max(pos, abs(neg))
        ge.append(wts.abs().sum())
        dates.append(date)
    return pd.Series(ge, index=dates)


def print_leverage_table(monthly_excess, xsmom, corr_shrunk, vols,
                          corr_raw, vols_raw, gamma, backtest_start, end_date):
    s, e = backtest_start, end_date
    rows = []

    # OOS Eq1 (anchor)
    ge = compute_ge_indmom(monthly_excess, xsmom, vols, s, e)
    rows.append({"Strategi": "OOS Eq1 (anchor)", "Gns. GE": ge.mean() * 100})

    # Std MVO
    ge = compute_ge_epo_fixed_w(monthly_excess, xsmom, corr_raw, vols_raw,
                                  gamma, w=0.0, start=s, end=e)
    rows.append({"Strategi": "Std MVO", "Gns. GE": ge.mean() * 100})

    # EPO alle w-værdier
    for w in CANDIDATE_WS:
        ge = compute_ge_epo_fixed_w(monthly_excess, xsmom, corr_shrunk, vols,
                                     gamma, w=w, start=s, end=e)
        rows.append({"Strategi": f"EPO w={int(round(w*100))}%",
                     "Gns. GE": ge.mean() * 100})

    print("\n" + "=" * 45)
    print(f"GEARING — OOS: {backtest_start} → {end_date}")
    print("=" * 45)
    print(f"{'Strategi':<35} {'Gns. GE':>8}")
    print("-" * 45)
    for row in rows:
        print(f"{row['Strategi']:<35} {row['Gns. GE']:>7.0f}%")
    print("=" * 45)

# EPO vægte over tid
def plot_epo_weights_over_time(monthly_excess, xsmom, corr_shrunk, vols,
                                gamma, w=0.75, start="2025-01-01",
                                end="2025-12-31", top_n=10):
    """
    Beregner EPO-vægte (w=0.75) for hver måned i perioden og viser
    de top_n long- og top_n short-positioner som linjediagrammer over tid.
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    dates = [d for d in monthly_excess.index if s <= d <= e]

    # ── Byg vægtmatrix måned for måned ───────────────────────
    weight_records = {}
    for date in dates:
        # Signal og risikomodel bruger foregående måneds data
        valid_signal = monthly_excess.index[monthly_excess.index < date]
        if len(valid_signal) == 0:
            continue
        sig_date = valid_signal[-1]

        if sig_date not in xsmom.index:
            continue
        if sig_date not in corr_shrunk:
            continue

        wts = epo_weights(
            signal=xsmom.loc[sig_date],
            corr=corr_shrunk[sig_date],
            vols=vols[sig_date],
            gamma=gamma,
            w=w
        )
        if len(wts) > 0:
            weight_records[date] = wts

    if not weight_records:
        print("Ingen vægte beregnet — tjek at data dækker perioden.")
        return

    weights_df = pd.DataFrame(weight_records).T.fillna(0)

    # ── Identificer top_n long og top_n short på tværs af 2025 ──
    mean_wts   = weights_df.mean()
    top_long   = mean_wts.nlargest(top_n).index
    top_short  = mean_wts.nsmallest(top_n).index

    # ── Plot ──────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", palette="tab10")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Long-positioner
    ax = axes[0]
    for ticker in top_long:
        ax.plot(weights_df.index, weights_df[ticker] * 100,
                marker="o", markersize=4, linewidth=1.8, label=ticker)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Top {top_n} long-positioner — EPO w={w} (2025)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Vægt (%)")
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    # Short-positioner
    ax = axes[1]
    for ticker in top_short:
        ax.plot(weights_df.index, weights_df[ticker] * 100,
                marker="o", markersize=4, linewidth=1.8, label=ticker)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Top {top_n} short-positioner — EPO w={w} (2025)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Vægt (%)")
    ax.set_xlabel("Måned")
    ax.legend(fontsize=8, ncol=2, loc="lower right")

    plt.xticks(weights_df.index,
               [d.strftime("%Y-%m") for d in weights_df.index],
               rotation=45, ha="right")
    plt.suptitle(f"EPO porteføljervægte over tid (w={w})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return weights_df

def backtest_buy_and_hold_2025(monthly_excess, xsmom, corr_shrunk, vols,
                                gamma, w=0.75,
                                signal_date="2024-12-31",
                                start="2025-01-01",
                                end="2025-12-31"):
    """
    Buy-and-hold 2025: vægte fryses ved signal_date og rebalanceres ikke.
    Afkastet beregnes ved at holde samme vægte hele året.
    """
    # Find nærmeste tilgængelige signal-dato
    sig_date = monthly_excess.index[
        monthly_excess.index <= pd.to_datetime(signal_date)
    ][-1]

    # Beregn EPO-vægte én gang
    wts = epo_weights(
        signal=xsmom.loc[sig_date],
        corr=corr_shrunk[sig_date],
        vols=vols[sig_date],
        gamma=gamma,
        w=w
    )
    if len(wts) == 0:
        raise ValueError("Ingen vægte beregnet — tjek signal og risikomodel.")

    print(f"\n  Signal-dato:        {sig_date.strftime('%Y-%m')}")
    print(f"  Antal long:         {(wts > 0).sum()}")
    print(f"  Antal short:        {(wts < 0).sum()}")
    print(f"  Brutto-eksponering: {wts.abs().sum():.2%}")

    # Buy-and-hold: samme vægte hver måned
    period = monthly_excess.loc[start:end]
    rets, dates = [], []

    for date, row in period.iterrows():
        r     = row.reindex(wts.index).dropna()
        w_aln = wts.reindex(r.index).dropna()
        if len(w_aln) == 0:
            continue
        rets.append((w_aln * r.reindex(w_aln.index)).sum())
        dates.append(date)

    s = pd.Series(rets, index=dates, name=f"Buy-and-Hold EPO w={w} (2025)")
    return s

# leverage
def backtest_leveraged_buy_and_hold_2025(monthly_excess, xsmom, corr_shrunk, vols,
                                          gamma, w=0.75,
                                          signal_date="2024-12-31",
                                          start="2025-01-01",
                                          end="2025-12-31",
                                          target_leverage=1.0):
    """
    Beregner buy-and-hold med leverage-skalerede vægte og sammenligner
    med den uskalerede buy-and-hold.
    target_leverage: ønsket brutto-eksponering (f.eks. 1.47 = anchor niveau)
    """
    sig_date = monthly_excess.index[
        monthly_excess.index <= pd.to_datetime(signal_date)
    ][-1]

    wts = epo_weights(
        signal=xsmom.loc[sig_date],
        corr=corr_shrunk[sig_date],
        vols=vols[sig_date],
        gamma=gamma,
        w=w
    )
    if len(wts) == 0:
        raise ValueError("Ingen vægte beregnet.")

    # Uskalerede vægte (original buy-and-hold)
    period = monthly_excess.loc[start:end]
    rets_unscaled, rets_scaled, dates = [], [], []

    # Skalér vægte til target leverage
    current_ge = wts.abs().sum()
    scaled_wts = wts * (target_leverage / current_ge)

    for date, row in period.iterrows():
        r = row.reindex(wts.index).dropna()

        # Uskaleret
        w_aln = wts.reindex(r.index).dropna()
        if len(w_aln) > 0:
            rets_unscaled.append((w_aln * r.reindex(w_aln.index)).sum())

        # Skaleret
        w_scl = scaled_wts.reindex(r.index).dropna()
        if len(w_scl) > 0:
            rets_scaled.append((w_scl * r.reindex(w_scl.index)).sum())

        dates.append(date)

    bah_unscaled = pd.Series(rets_unscaled, index=dates,
                              name=f"Buy-and-Hold EPO w={w}")
    bah_scaled   = pd.Series(rets_scaled,   index=dates,
                              name=f"Leveraged BaH (GE={target_leverage:.0%})")

    # Print sammenligning
    print("\n" + "=" * 65)
    print(f"BUY-AND-HOLD: USKALERET vs. LEVERAGE-SKALERET (GE={target_leverage:.0%}) — 2025")
    print("=" * 65)
    print(f"  {'Måned':<10} {'Uskaleret':>14} {f'Skaleret ({target_leverage:.0%} GE)':>20}")
    print("-" * 65)
    for date in dates:
        u = bah_unscaled.loc[date]
        s = bah_scaled.loc[date]
        print(f"  {date.strftime('%Y-%m'):<10} {u:>13.2%} {s:>19.2%}")

    cum_u = (1 + bah_unscaled).prod() - 1
    cum_s = (1 + bah_scaled).prod() - 1
    print("-" * 65)
    print(f"  {'Kumuleret':<10} {cum_u:>13.2%} {cum_s:>19.2%}")
    print("=" * 65)

    perf_u = performance_summary(bah_unscaled, f"Buy-and-Hold EPO w={w}")
    perf_s = performance_summary(bah_scaled,   f"Leveraged BaH (GE={target_leverage:.0%})")
    perf_df = pd.DataFrame([perf_u, perf_s]).set_index("Strategy")[
        ["Ann. Return", "Ann. Vol", "Sharpe"]
    ]
    print(perf_df.to_string(float_format=lambda x: f"{x:.4f}"))

    return bah_unscaled, bah_scaled

# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("EPO — Yahoo Finance enkeltaktier")
    print(f"Data: {START_DATE} → {END_DATE}")
    print(f"Backtest OOS fra: {BACKTEST_START}")
    print("=" * 65)

    # 1. Data
    daily, daily_prices, ticker_to_sector = load_data()
    monthly = to_monthly_returns(daily)
    rf = get_rf_monthly(START_DATE, END_DATE)
    monthly_excess = compute_monthly_excess(monthly, rf)
    print(f"\nAktier: {monthly_excess.shape[1]}  |  "
          f"Periode: {monthly_excess.index[0].date()} → "
          f"{monthly_excess.index[-1].date()}")

    # 2. Signal
    print("\nBeregner signal...")
    xsmom = compute_tsmom_signal(monthly_excess, ticker_to_sector, LOOKBACK_MONTHS)

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
    mom_ew_full = backtest_simple_momentum_ew(monthly_excess, monthly)
    én_over_N = compute_equal_weight_benchmark(monthly_excess)

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
        performance_summary(subset(mom_ew_full, s, e), "TSMOM EW (hele univers)"),
        performance_summary(subset(én_over_N, s, e), "1/N"),
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

    # ── Buy-and-hold vs. månedlig rebalancering 2025 ─────────────
    bah_2025 = backtest_buy_and_hold_2025(
        monthly_excess=monthly_excess,
        xsmom=xsmom,
        corr_shrunk=corr_shrunk,
        vols=vols,
        gamma=GAMMA,
        w=0.75
    )

    epo_75_2025 = subset(epo_panel["EPO_w_0.75"], "2025-01-01", "2025-12-31")

    print("\n" + "=" * 60)
    print("BUY-AND-HOLD vs. MÅNEDLIG REBALANCERING — 2025")
    print("=" * 60)
    sammenligning = pd.DataFrame([
        performance_summary(bah_2025, "Buy-and-Hold EPO w=0.75"),
        performance_summary(epo_75_2025, "Månedlig rebalancering EPO w=0.75"),
    ]).set_index("Strategy")[["Ann. Return", "Ann. Vol", "Sharpe"]]
    print(sammenligning.to_string(float_format=lambda x: f"{x:.4f}"))

    # Kumuleret afkast måned for måned
    print("\nMånedlige afkast:")
    print(f"  {'Måned':<10} {'Buy-and-Hold':>14} {'Rebalancering':>14}")
    print("-" * 42)
    for date in bah_2025.index:
        bah_r = bah_2025.loc[date]
        reb_r = epo_75_2025.loc[date] if date in epo_75_2025.index else float("nan")
        print(f"  {date.strftime('%Y-%m'):<10} {bah_r:>13.2%} {reb_r:>13.2%}")

    cum_bah = (1 + bah_2025).prod() - 1
    cum_reb = (1 + epo_75_2025).prod() - 1
    print("-" * 42)
    print(f"  {'Kumuleret':<10} {cum_bah:>13.2%} {cum_reb:>13.2%}")


    # 7. Terminal statistik
    print_top_bottom_stocks(monthly_excess, ticker_to_sector, END_DATE)
    print_top_bottom_correlations(daily_prices, ticker_to_sector, n=10)

    # 7b. Leverage tabel
    print_leverage_table(
        monthly_excess=monthly_excess,
        xsmom=xsmom,
        corr_shrunk=corr_shrunk,
        vols=vols,
        corr_raw=corr_raw,
        vols_raw=vols_raw,
        gamma=GAMMA,
        backtest_start=BACKTEST_START,
        end_date=END_DATE,
    )

    # Leverage-skalerede vægte
    backtest_leveraged_buy_and_hold_2025(
        monthly_excess=monthly_excess,
        xsmom=xsmom,
        corr_shrunk=corr_shrunk,
        vols=vols,
        gamma=GAMMA,
        w=0.75,
        signal_date="2024-12-31",
        target_leverage=1.0
    )

    # turnover
    print("\nGenererer turnover-plot...")
    plot_turnover(
        monthly_excess=monthly_excess,
        xsmom=xsmom,
        corr_shrunk=corr_shrunk,
        vols=vols,
        corr_raw=corr_raw,
        vols_raw=vols_raw,
        gamma=GAMMA,
        backtest_start=START_DATE,
        end_date=END_DATE,
        roll_window=12
    )

    ret_2025 = subset(monthly_excess, "2025-01-01", "2025-12-31")
    print(f"Gns. månedligt afkast 2025: {ret_2025.mean().mean():.4f}")
    print(f"Gns. månedlig volatilitet 2025: {ret_2025.std().mean():.4f}")

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