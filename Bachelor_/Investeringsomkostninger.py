# ── investeringsomkostninger.py ───────────────────────────────

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

from Equity_1 import epo_weights


# ── 1. Turnover-beregning ─────────────────────────────────────

def compute_turnover_series(monthly_excess, xsmom, corr_dict, vols_dict,
                             gamma, w, start, end):
    """
    Månedlig turnover = 0.5 * sum(|w_t - w_{t-1}|)
    Returnerer en pd.Series med turnover per måned.
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    idx  = monthly_excess.loc[s:e].index
    risk_dates = set(corr_dict)
    sig_dates  = set(xsmom.index)

    prev_wts = None
    turnover, dates = [], []

    for date in idx:
        if date not in risk_dates or date not in sig_dates:
            continue

        wts = epo_weights(xsmom.loc[date], corr_dict[date],
                          vols_dict[date], gamma, w)
        if len(wts) == 0:
            continue

        if prev_wts is not None:
            all_tickers = wts.index.union(prev_wts.index)
            w_cur  = wts.reindex(all_tickers).fillna(0.0)
            w_prev = prev_wts.reindex(all_tickers).fillna(0.0)
            turnover.append(0.5 * (w_cur - w_prev).abs().sum())
            dates.append(date)

        prev_wts = wts

    return pd.Series(turnover, index=dates, name=f"EPO_w{w:.2f}")


def compute_turnover_annual_rebalance(monthly_excess, xsmom, corr_dict,
                                       vols_dict, gamma, w, start, end):
    """
    Turnover for årlig rebalancering:
    Vægte genberegnes kun i januar — turnover beregnes kun i de måneder
    hvor vægtene faktisk ændres (dvs. januar hvert år).
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    period = monthly_excess.loc[s:e]

    current_wts = None
    prev_wts    = None
    turnover, dates = [], []

    for date, _ in period.iterrows():
        valid_signals = xsmom.index[xsmom.index < date]
        if len(valid_signals) == 0:
            continue
        sig_date = valid_signals[-1]

        if sig_date not in corr_dict or sig_date not in vols_dict:
            continue

        # Genberegn vægte kun i januar
        if date.month == 1 or current_wts is None:
            wts = epo_weights(xsmom.loc[sig_date], corr_dict[sig_date],
                              vols_dict[sig_date], gamma, w)
            if len(wts) > 0:
                current_wts = wts

        if current_wts is None:
            continue

        if prev_wts is not None:
            all_tickers = current_wts.index.union(prev_wts.index)
            w_cur  = current_wts.reindex(all_tickers).fillna(0.0)
            w_prev = prev_wts.reindex(all_tickers).fillna(0.0)
            diff   = 0.5 * (w_cur - w_prev).abs().sum()
            # Kun registrer turnover hvis vægtene faktisk har ændret sig
            if diff > 1e-8:
                turnover.append(diff)
                dates.append(date)

        prev_wts = current_wts.copy()

    return pd.Series(turnover, index=dates, name=f"Årlig_reb_w{w:.2f}")


# ── 2. Turnover-statistik ─────────────────────────────────────

def print_turnover_stats(monthly_excess, xsmom, corr_shrunk, vols,
                          corr_raw, vols_raw, gamma,
                          start="2023-01-01", end="2025-12-31",
                          w=0.75):
    """
    Printer antal turnovers og gennemsnitlig turnover per handel
    for månedlig og årlig rebalancering samt MVO i perioden.
    """
    print("\n" + "=" * 65)
    print(f"TURNOVER STATISTIK — {start[:7]} → {end[:7]}")
    print("=" * 65)
    print(f"  {'Strategi':<35} {'Antal TO':>10} {'Gns. TO':>10} {'Total TO':>10}")
    print("-" * 65)

    strategies = {
        f"Månedlig reb. EPO w={w}": compute_turnover_series(
            monthly_excess, xsmom, corr_shrunk, vols,
            gamma, w, start, end),
        f"Årlig reb. EPO w={w}": compute_turnover_annual_rebalance(
            monthly_excess, xsmom, corr_shrunk, vols,
            gamma, w, start, end),
        "Std MVO": compute_turnover_series(
            monthly_excess, xsmom, corr_raw, vols_raw,
            gamma, 0.0, start, end),
        "INDMOM (w=1.00)": compute_turnover_series(
            monthly_excess, xsmom, corr_shrunk, vols,
            gamma, 1.0, start, end),
    }

    for name, to_series in strategies.items():
        n      = len(to_series)
        mean   = to_series.mean() * 100 if n > 0 else 0
        total  = to_series.sum()  * 100 if n > 0 else 0
        print(f"  {name:<35} {n:>10} {mean:>9.1f}% {total:>9.1f}%")

    print("=" * 65)
    return strategies


# ── 3. Turnover-plot ──────────────────────────────────────────

def plot_turnover(monthly_excess, xsmom, corr_shrunk, vols,
                  corr_raw, vols_raw, gamma,
                  backtest_start, end_date,
                  roll_window=12):
    """
    Rullende turnover-plot for månedlig rebalancering, MVO og INDMOM.
    Bruger korrekt compute_turnover_series uden prev_wts nulstilling.
    """
    strategies = {
        "Std MVO":          (corr_raw,    vols_raw, 0.00),
        "EPO w=0%":         (corr_shrunk, vols,     0.00),
        "EPO w=25%":        (corr_shrunk, vols,     0.25),
        "EPO w=75%":        (corr_shrunk, vols,     0.75),
        "EPO w=100%":       (corr_shrunk, vols,     1.00),
    }

    colors = {
        "Std MVO":    "black",
        "EPO w=0%":   "#d62728",
        "EPO w=25%":  "#ff7f0e",
        "EPO w=75%":  "#2ca02c",
        "EPO w=100%": "#1f77b4",
    }

    fig, ax = plt.subplots(figsize=(16, 6))

    for label, (corr_d, vol_d, w) in strategies.items():
        print(f"  Beregner turnover: {label} …", flush=True)
        to_series  = compute_turnover_series(
            monthly_excess, xsmom, corr_d, vol_d,
            gamma, w, backtest_start, end_date)
        rolling_to = to_series.rolling(window=roll_window,
                                        min_periods=roll_window).mean()
        ax.plot(rolling_to.index, rolling_to * 100,
                label=label, color=colors[label], linewidth=1.5)

    ax.set_title(
        f"Rullende {roll_window}-måneders gennemsnitlig turnover\n"
        f"OOS: {backtest_start} → {end_date}",
        fontsize=13, pad=12)
    ax.set_ylabel("Månedlig turnover (%, rullende gns.)", fontsize=11)
    ax.set_xlabel("Dato", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/Users/emilbundesen/Desktop/Bachelor/Turnover_EPO.png",
        dpi=150, bbox_inches="tight")
    plt.show()


def compute_net_returns(monthly_excess, xsmom, corr_shrunk, vols,
                         corr_raw, vols_raw, gamma,
                         start="2023-01-01", end="2025-12-31",
                         w=0.75, c=0.001):
    """
    Beregner nettoafkast = bruttoafkast - transaktionsomkostninger
    TC_t = c * turnover_t
    c = 10 bps (0.001) per handel som default
    """
    from Equity_1 import build_epo_panel, subset, performance_summary

    s, e = start, end

    # ── Turnover serier ───────────────────────────────────────
    to_monthly = compute_turnover_series(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma, w, s, e)

    to_annual = compute_turnover_annual_rebalance(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma, w, s, e)

    # ── Bruttoafkast serier ───────────────────────────────────
    from Stock_Data import (
        backtest_buy_and_hold_period,
        backtest_annual_rebalance_period,
        subset,
    )
    from Equity_1 import build_epo_panel

    # Månedlig reb. bruttoafkast
    epo_panel   = build_epo_panel(
        monthly_excess, xsmom, corr_shrunk, vols, gamma, [w])
    gross_mon   = subset(epo_panel[f"EPO_w_{w:.2f}"], s, e)

    # Årlig reb. bruttoafkast
    gross_ann   = backtest_annual_rebalance_period(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma=gamma, w=w, start=s, end=e)

    # ── Nettoafkast = brutto - c * turnover ───────────────────
    tc_monthly  = (c * to_monthly).reindex(gross_mon.index).fillna(0.0)
    tc_annual   = (c * to_annual).reindex(gross_ann.index).fillna(0.0)

    net_monthly = gross_mon  - tc_monthly
    net_annual  = gross_ann  - tc_annual

    # ── Performance ───────────────────────────────────────────
    def perf(series, name):
        r       = series.dropna()
        n       = len(r)
        ann_ret = (1 + r).prod() ** (12 / n) - 1 if n > 0 else np.nan
        ann_vol = r.std() * np.sqrt(12)           if n > 0 else np.nan
        sharpe  = ann_ret / ann_vol               if ann_vol > 0 else np.nan
        cum     = (1 + r).prod() - 1
        return {"Strategi": name,
                "Ann. Ret": ann_ret,
                "Ann. Vol": ann_vol,
                "Sharpe":   sharpe,
                "Kum. afkast": cum}

    rows = [
        perf(gross_mon,  f"Månedlig reb. (brutto) w={w}"),
        perf(net_monthly, f"Månedlig reb. (netto, c={c*10000:.0f}bp) w={w}"),
        perf(gross_ann,  f"Årlig reb. (brutto) w={w}"),
        perf(net_annual,  f"Årlig reb. (netto, c={c*10000:.0f}bp) w={w}"),
    ]

    # ── Print tabel ───────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"NETTO PERFORMANCE EFTER TRANSAKTIONSOMKOSTNINGER (c = {c*10000:.0f} bp)")
    print(f"Periode: {start[:7]} → {end[:7]}")
    print("=" * 75)
    print(f"  {'Strategi':<45} {'Ann. Ret':>9} {'Ann. Vol':>9} "
          f"{'Sharpe':>8} {'Kum.':>8}")
    print("-" * 75)
    for row in rows:
        print(f"  {row['Strategi']:<45} "
              f"{row['Ann. Ret']:>9.2%} "
              f"{row['Ann. Vol']:>9.2%} "
              f"{row['Sharpe']:>8.3f} "
              f"{row['Kum. afkast']:>8.2%}")
    print("=" * 75)

    return net_monthly, net_annual

def plot_net_cumulative_vs_cost(monthly_excess, xsmom, corr_shrunk, vols,
                                 gamma, start="2023-01-01", end="2025-12-31",
                                 w=0.75):
    """
    Viser kumuleret nettoafkast som funktion af transaktionsomkostninger c
    for månedlig og årlig rebalancering.
    X-akse: c i basispoint (0 → 100)
    Y-akse: kumuleret nettoafkast
    """
    from Equity_1 import build_epo_panel
    from Stock_Data import backtest_annual_rebalance_period, subset

    s, e = start, end

    # ── Bruttoafkast ──────────────────────────────────────────
    epo_panel  = build_epo_panel(
        monthly_excess, xsmom, corr_shrunk, vols, gamma, [w])
    gross_mon  = subset(epo_panel[f"EPO_w_{w:.2f}"], s, e)
    gross_ann  = backtest_annual_rebalance_period(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma=gamma, w=w, start=s, end=e)

    # ── Turnover serier ───────────────────────────────────────
    to_mon = compute_turnover_series(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma, w, s, e)
    to_ann = compute_turnover_annual_rebalance(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma, w, s, e)

    # ── Beregn kumuleret nettoafkast for c = 0 → 100 bp ──────
    c_values = np.arange(0, 101, 1)          # 0, 1, 2, … 100 bp
    cum_mon, cum_ann = [], []

    for c_bp in c_values:
        c = c_bp / 10_000

        tc_mon  = (c * to_mon).reindex(gross_mon.index).fillna(0.0)
        net_mon = gross_mon - tc_mon
        cum_mon.append((1 + net_mon.dropna()).prod() - 1)

        tc_ann  = (c * to_ann).reindex(gross_ann.index).fillna(0.0)
        net_ann = gross_ann - tc_ann
        cum_ann.append((1 + net_ann.dropna()).prod() - 1)

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(c_values, [r * 100 for r in cum_mon],
            color="#2ca02c", linewidth=2,
            label=f"Månedlig rebalancering EPO $w={w}$")
    ax.plot(c_values, [r * 100 for r in cum_ann],
            color="#1f77b4", linewidth=2, linestyle="--",
            label=f"Årlig rebalancering EPO $w={w}$")

    # Markér hvor de to kurver krydser
    diff = np.array(cum_mon) - np.array(cum_ann)
    cross = np.where(np.diff(np.sign(diff)))[0]
    if len(cross) > 0:
        cx = c_values[cross[0]]
        cy = cum_mon[cross[0]] * 100
        ax.axvline(cx, color="red", linewidth=1, linestyle=":",
                   label=f"Breakeven: $c = {cx}$ bp")
        ax.annotate(f"{cx} bp", xy=(cx, cy),
                    xytext=(cx + 3, cy + 0.5),
                    fontsize=10, color="red")

    ax.set_title(
        f"Kumuleret nettoafkast som funktion af transaktionsomkostninger\n"
        f"Periode: {start[:7]} → {end[:7]}, EPO $w={w}$",
        fontsize=13, pad=12)
    ax.set_xlabel("Transaktionsomkostninger $c$ (basispoint)", fontsize=11)
    ax.set_ylabel("Kumuleret nettoafkast (%)", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))

    plt.tight_layout()
    plt.savefig(
        "/Users/emilbundesen/Desktop/Bachelor/Netto_vs_Omkostninger.png",
        dpi=150, bbox_inches="tight")
    plt.show()

    # Print breakeven
    if len(cross) > 0:
        print(f"\n  Breakeven ved c = {cx} bp")
        print(f"  Ved c > {cx} bp outperformer årlig rebalancering månedlig.")
    else:
        print("\n  Ingen breakeven fundet i intervallet 0-100 bp.")