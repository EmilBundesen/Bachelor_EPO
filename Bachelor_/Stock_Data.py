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

START_DATE        = "2010-01-01"
END_DATE          = "2025-12-31"
BACKTEST_START    = "2020-01-01"
MAX_NAN_THRESHOLD = 0.20
N_LONG_SHORT      = 80


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

    # Beregn kumuleret afkast per sektor
    sector_cum_final = {}
    sector_cum_series = {}
    for sector, tickers in sector_groups.items():
        available = [t for t in tickers if t in full.columns]
        if not available:
            continue
        sector_ret = full[available].mean(axis=1)
        cum_series = (1 + sector_ret).cumprod() - 1
        sector_cum_final[sector] = cum_series.iloc[-1]
        sector_cum_series[sector] = cum_series

    # Top 5 og bund 5
    sorted_sectors = sorted(sector_cum_final, key=sector_cum_final.get, reverse=True)
    top5 = sorted_sectors[:5]
    bottom5 = sorted_sectors[-5:]
    show = set(top5 + bottom5)

    colors_top = sns.color_palette("Greens_d", 5)
    colors_bottom = sns.color_palette("Reds_d", 5)

    for i, sector in enumerate(top5):
        ax.plot(sector_cum_series[sector], label=f"↑ {sector.capitalize()}",
                linewidth=2, color=colors_top[i])

    for i, sector in enumerate(bottom5):
        ax.plot(sector_cum_series[sector], label=f"↓ {sector.capitalize()}",
                linewidth=2, color=colors_bottom[i], linestyle="--")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Kumuleret afkast pr. sektor — top 5 og bund 5 (hele periode, equal-weighted)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Kumuleret afkast")
    ax.set_xlabel("Dato")
    ax.legend(fontsize=10, ncol=2)
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
                                gamma, w=0.75, start="2023-01-01",
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

# debug
def visualize_date_flow_simple(monthly_excess, xsmom, vols_dict, corr_dict):
    """
    Viser tidsflow: signal → vægte → return.
    """
    print("="*90)
    print("FULL DATE FLOW VISUALIZATION (2015-2025)")
    print("="*90)
    print(f"{'Portefølje':<12} {'Signal':<12} {'Risk':<12} {'Return':<12} {'Status':<12}")
    print("-"*90)

    for port_date in monthly_excess.index:
        # Signal = sidste tilgængelige signal før port_date
        sig_date = xsmom.index[xsmom.index < port_date][-1] if any(xsmom.index < port_date) else pd.NaT

        # Risk = sidste tilgængelige risiko før port_date
        risk_dates = sorted(set(vols_dict.keys()).intersection(set(corr_dict.keys())))
        risk_date = max([d for d in risk_dates if d <= port_date], default=pd.NaT)

        # Return = næste måned efter port_date
        ret_idx = monthly_excess.index[monthly_excess.index > port_date]
        ret_date = ret_idx[0] if len(ret_idx) > 0 else pd.NaT

        # Tjek rækkefølge
        if sig_date is pd.NaT or ret_date is pd.NaT:
            status = "NA"
        elif sig_date < port_date < ret_date:
            status = "OK"
        else:
            status = "LOOK-AHEAD!"

        print(f"{port_date.strftime('%Y-%m'):<12} "
              f"{sig_date.strftime('%Y-%m') if sig_date is not pd.NaT else 'NA':<12} "
              f"{risk_date.strftime('%Y-%m') if risk_date is not pd.NaT else 'NA':<12} "
              f"{ret_date.strftime('%Y-%m') if ret_date is not pd.NaT else 'NA':<12} "
              f"{status:<12}")

## Implementering af 2023 aktieunivers

def backtest_buy_and_hold_period(monthly_excess, xsmom, corr_shrunk, vols,
                                  gamma, w=0.75,
                                  signal_date="2022-12-31",
                                  start="2023-01-01",
                                  end="2025-12-31"):
    """
    Buy-and-Hold: vægte fryses én gang ved signal_date.
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

    period = monthly_excess.loc[start:end]
    rets, dates = [], []
    for date, row in period.iterrows():
        r     = row.reindex(wts.index).dropna()
        w_aln = wts.reindex(r.index).dropna()
        if len(w_aln) == 0:
            continue
        rets.append((w_aln * r.reindex(w_aln.index)).sum())
        dates.append(date)

    return pd.Series(rets, index=dates, name=f"Buy-and-Hold EPO w={w}")


def backtest_annual_rebalance_period(monthly_excess, xsmom, corr_shrunk, vols,
                                      gamma, w=0.75,
                                      start="2023-01-01",
                                      end="2025-12-31"):
    """
    Årlig rebalancering: vægte genberegnes hvert januar
    ud fra signal fra december måneden før.
    """
    period = monthly_excess.loc[start:end]
    current_wts = None
    rets, dates = [], []

    for date, row in period.iterrows():
        # Genberegn vægte i januar hvert år
        if date.month == 1 or current_wts is None:
            valid_signals = xsmom.index[xsmom.index < date]
            if len(valid_signals) == 0:
                continue
            sig_date = valid_signals[-1]

            if sig_date not in corr_shrunk or sig_date not in vols:
                continue

            wts = epo_weights(
                signal=xsmom.loc[sig_date],
                corr=corr_shrunk[sig_date],
                vols=vols[sig_date],
                gamma=gamma,
                w=w
            )
            if len(wts) > 0:
                current_wts = wts

        if current_wts is None:
            continue

        r     = row.reindex(current_wts.index).dropna()
        w_aln = current_wts.reindex(r.index).dropna()
        if len(w_aln) == 0:
            continue
        rets.append((w_aln * r.reindex(w_aln.index)).sum())
        dates.append(date)

    return pd.Series(rets, index=dates, name=f"Årlig rebalancering EPO w={w}")


def print_monthly_comparison_table(strategies: dict[str, pd.Series],
                                    start="2023-01-01",
                                    end="2025-12-31"):
    """
    Printer månedlige afkast side om side for flere strategier.
    strategies: dict med navn → pd.Series af månedlige afkast
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)

    # Saml alle datoer
    all_dates = sorted(set().union(*[s_.loc[s:e].index for s_ in strategies.values()]))

    col_w = 16
    header = f"  {'Måned':<10}" + "".join(f"{name:>{col_w}}" for name in strategies)
    sep    = "-" * (12 + col_w * len(strategies))

    print("\n" + "=" * (12 + col_w * len(strategies)))
    print(f"MÅNEDLIGE AFKAST — {start[:7]} → {end[:7]}")
    print("=" * (12 + col_w * len(strategies)))
    print(header)
    print(sep)

    cum = {name: 1.0 for name in strategies}

    for date in all_dates:
        row_str = f"  {date.strftime('%Y-%m'):<10}"
        for name, series in strategies.items():
            if date in series.index:
                r = series.loc[date]
                cum[name] *= (1 + r)
                row_str += f"{r:>{col_w}.2%}"
            else:
                row_str += f"{'N/A':>{col_w}}"
        print(row_str)

    print(sep)
    cum_str = f"  {'Kumuleret':<10}"
    for name in strategies:
        cum_str += f"{cum[name] - 1:>{col_w}.2%}"
    print(cum_str)
    print("=" * (12 + col_w * len(strategies)))

    # Kort performance-opsummering
    print(f"\n  {'Strategi':<35} {'Ann. Ret':>10} {'Ann. Vol':>10} {'Sharpe':>8}")
    print("-" * 67)
    for name, series in strategies.items():
        s_ = series.loc[s:e]
        perf = performance_summary(s_, name)
        print(f"  {name:<35} {perf['Ann. Return']:>10.4f} "
              f"{perf['Ann. Vol']:>10.4f} {perf['Sharpe']:>8.4f}")

# Leverage justeret årligt afkast:
# ── Leverage-skaleret 2023-2025 sammenligning ─────────────────

def backtest_leverage_scaled_period(monthly_excess, xsmom, corr_shrunk, vols,
                                     gamma, w=0.75,
                                     start="2023-01-01",
                                     end="2025-12-31",
                                     target_ge=1.0):
    """
    Returnerer leverage-skalerede versioner af alle fire strategier
    i perioden 2023-2025, skaleret til target_ge brutto-eksponering.
    Skaleringen sker måned for måned baseret på den aktuelle GE.
    """

    period = monthly_excess.loc[start:end]

    # ── Buy-and-Hold (fryses ved 2022-12-31) ─────────────────
    sig_date_bah = monthly_excess.index[
        monthly_excess.index <= pd.to_datetime("2022-12-31")
    ][-1]
    wts_bah = epo_weights(xsmom.loc[sig_date_bah], corr_shrunk[sig_date_bah],
                           vols[sig_date_bah], gamma, w)
    ge_bah   = wts_bah.abs().sum()
    wts_bah_scaled = wts_bah * (target_ge / ge_bah)

    # ── Årlig rebalancering ───────────────────────────────────
    current_wts_ann = None
    current_wts_ann_scaled = None

    rets_bah, rets_mon, rets_ann, rets_ew, dates = [], [], [], [], []

    for date, row in period.iterrows():

        # Månedlig rebalancering: hent EPO-vægte fra signal måneden før
        valid_signals = xsmom.index[xsmom.index < date]
        if len(valid_signals) == 0:
            continue
        sig_date = valid_signals[-1]

        if sig_date not in corr_shrunk or sig_date not in vols:
            continue

        wts_mon = epo_weights(xsmom.loc[sig_date], corr_shrunk[sig_date],
                               vols[sig_date], gamma, w)
        if len(wts_mon) == 0:
            continue
        ge_mon = wts_mon.abs().sum()
        wts_mon_scaled = wts_mon * (target_ge / ge_mon) if ge_mon > 0 else wts_mon

        # Årlig rebalancering: opdater i januar
        if date.month == 1 or current_wts_ann is None:
            wts_ann = epo_weights(xsmom.loc[sig_date], corr_shrunk[sig_date],
                                   vols[sig_date], gamma, w)
            if len(wts_ann) > 0:
                current_wts_ann = wts_ann
                ge_ann = wts_ann.abs().sum()
                current_wts_ann_scaled = wts_ann * (target_ge / ge_ann) if ge_ann > 0 else wts_ann

        if current_wts_ann_scaled is None:
            continue

        r = row

        # Buy-and-Hold
        r_bah = r.reindex(wts_bah_scaled.index).dropna()
        w_bah = wts_bah_scaled.reindex(r_bah.index).dropna()
        ret_bah = (w_bah * r_bah.reindex(w_bah.index)).sum() if len(w_bah) > 0 else np.nan

        # Månedlig reb.
        r_mon = r.reindex(wts_mon_scaled.index).dropna()
        w_mon = wts_mon_scaled.reindex(r_mon.index).dropna()
        ret_mon = (w_mon * r_mon.reindex(w_mon.index)).sum() if len(w_mon) > 0 else np.nan

        # Årlig reb.
        r_ann = r.reindex(current_wts_ann_scaled.index).dropna()
        w_ann = current_wts_ann_scaled.reindex(r_ann.index).dropna()
        ret_ann = (w_ann * r_ann.reindex(w_ann.index)).sum() if len(w_ann) > 0 else np.nan

        # 1/N (ingen skalering nødvendig — GE = 1 per definition)
        r_ew = r.dropna()
        ret_ew = r_ew.mean() if len(r_ew) > 0 else np.nan

        rets_bah.append(ret_bah)
        rets_mon.append(ret_mon)
        rets_ann.append(ret_ann)
        rets_ew.append(ret_ew)
        dates.append(date)

    return {
        f"Buy-and-Hold (skaleret {target_ge:.0%} GE)":      pd.Series(rets_bah, index=dates),
        f"Månedlig reb. (skaleret {target_ge:.0%} GE)":     pd.Series(rets_mon, index=dates),
        f"Årlig reb. (skaleret {target_ge:.0%} GE)":        pd.Series(rets_ann, index=dates),
        "1/N (Equal Weight)":                                pd.Series(rets_ew,  index=dates),
    }


def print_scaled_annual_table(scaled_strategies: dict[str, pd.Series],
                               start="2023-01-01", end="2025-12-31"):
    """
    Printer årlige afkast for leverage-skalerede strategier side om side.
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    years = sorted({d.year for series in scaled_strategies.values()
                    for d in series.loc[s:e].index})

    names = list(scaled_strategies.keys())
    col_w = 26

    header = f"  {'År':<8}" + "".join(f"{n:>{col_w}}" for n in names)
    sep    = "-" * (10 + col_w * len(names))

    print("\n" + "=" * (10 + col_w * len(names)))
    print(f"ÅRLIGE AFKAST (LEVERAGE-SKALERET) — {start[:4]}–{end[:4]}")
    print("=" * (10 + col_w * len(names)))
    print(header)
    print(sep)

    for year in years:
        row_str = f"  {year:<8}"
        for name, series in scaled_strategies.items():
            yr_data = series.loc[s:e]
            yr_data = yr_data[yr_data.index.year == year].dropna()
            if len(yr_data) == 0:
                row_str += f"{'N/A':>{col_w}}"
            else:
                ann_ret = (1 + yr_data).prod() - 1
                row_str += f"{ann_ret:>{col_w}.2%}"
        print(row_str)

    # Kumuleret
    print(sep)
    cum_str = f"  {'Kumuleret':<8}"
    for name, series in scaled_strategies.items():
        data = series.loc[s:e].dropna()
        cum  = (1 + data).prod() - 1
        cum_str += f"{cum:>{col_w}.2%}"
    print(cum_str)

    # Performance
    print(sep)
    print(f"\n  {'Strategi':<45} {'Ann. Ret':>10} {'Ann. Vol':>10} {'Sharpe':>8}")
    print("-" * 77)
    for name, series in scaled_strategies.items():
        data = series.loc[s:e].dropna()
        perf = performance_summary(data, name)
        print(f"  {name:<45} {perf['Ann. Return']:>10.4f} "
              f"{perf['Ann. Vol']:>10.4f} {perf['Sharpe']:>8.4f}")
    print("=" * (10 + col_w * len(names)))

# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("EPO — Yahoo Finance enkeltaktier")
    print(f"Data: {START_DATE} → {END_DATE}")
    print(f"Backtest OOS fra: 2020-01-01")
    print("=" * 65)

    BACKTEST_START_NEW = "2020-01-01"

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
    én_over_N   = compute_equal_weight_benchmark(monthly_excess)

    # 5. EPO
    print("\nBygger EPO panel...")
    epo_panel = build_epo_panel(
        monthly_excess, xsmom, corr_shrunk, vols, GAMMA, CANDIDATE_WS)
    print("\nBygger dynamisk OOS EPO...")
    epo_dyn = build_dynamic_oos_epo(
        epo_panel, oos_start=BACKTEST_START_NEW, min_history=MIN_HISTORY_OOS)

    # ── 6. Performance tabel OOS 2020-2025 ───────────────────
    s, e = BACKTEST_START_NEW, END_DATE
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
        performance_summary(subset(én_over_N,   s, e), "1/N"),
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
    print(f"PERFORMANCE — OOS: {BACKTEST_START_NEW} → {END_DATE}")
    print("=" * 65)
    print(perf.to_string())

    perf.to_csv("performance_summary_2020_2025.csv")
    print("\nGemt: performance_summary_2020_2025.csv")

    # ── 7. Månedlige afkast 2023-2025: fire strategier ────────
    PERIOD_START = "2023-01-01"
    PERIOD_END   = "2025-12-31"
    W            = 0.75

    print("\nBeregner 2023-2025 strategier...")

    # Buy-and-Hold (fryses én gang ved 2022-12-31)
    bah = backtest_buy_and_hold_period(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma=GAMMA, w=W,
        signal_date="2022-12-31",
        start=PERIOD_START, end=PERIOD_END
    )

    # Månedlig rebalancering (EPO w=0.75 fra panel)
    monthly_reb = subset(epo_panel[f"EPO_w_{W:.2f}"], PERIOD_START, PERIOD_END)
    monthly_reb.name = f"Månedlig rebalancering EPO w={W}"

    # Årlig rebalancering
    annual_reb = backtest_annual_rebalance_period(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma=GAMMA, w=W,
        start=PERIOD_START, end=PERIOD_END
    )

    # 1/N
    ew = subset(én_over_N, PERIOD_START, PERIOD_END)
    ew.name = "1/N (Equal Weight)"

    # Print samlet tabel
    print_monthly_comparison_table(
        strategies={
            f"Buy-and-Hold EPO w={W}":        bah,
            f"Månedlig reb. EPO w={W}":        monthly_reb,
            f"Årlig reb. EPO w={W}":           annual_reb,
            "1/N (Equal Weight)":              ew,
        },
        start=PERIOD_START,
        end=PERIOD_END
    )

    # ── 7b. Leverage-skalerede årlige afkast 2023-2025 ────────
    # EPO w=0.75 har gns. GE = 176% → skalér til 100% GE
    TARGET_GE = 1.0  # 100% GE

    print(f"\nBeregner leverage-skalerede afkast (target GE = {TARGET_GE:.0%})...")
    scaled = backtest_leverage_scaled_period(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma=GAMMA, w=W,
        start=PERIOD_START, end=PERIOD_END,
        target_ge=TARGET_GE
    )
    print_scaled_annual_table(scaled, start=PERIOD_START, end=PERIOD_END)

    # ── 8. Øvrig output (uændret) ─────────────────────────────
    print_top_bottom_stocks(monthly_excess, ticker_to_sector, END_DATE)
    print_top_bottom_correlations(daily_prices, ticker_to_sector, n=10)

    plot_epo_weights_over_time(monthly_excess, xsmom, corr_shrunk, vols,
                               GAMMA, w=0.75, start="2023-01-01",
                               end="2025-12-31", top_n=10)

    print_leverage_table(
        monthly_excess=monthly_excess,
        xsmom=xsmom,
        corr_shrunk=corr_shrunk,
        vols=vols,
        corr_raw=corr_raw,
        vols_raw=vols_raw,
        gamma=GAMMA,
        backtest_start=BACKTEST_START_NEW,
        end_date=END_DATE,
    )

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