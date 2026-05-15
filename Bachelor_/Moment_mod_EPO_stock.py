"""
EPO w=0.75 vs. TSMOM — Yahoo Finance aktier 2023–2025
======================================================
Genbruger præcis de serier der allerede beregnes i Equity_2.main():

  EPO w=0.75 månedlig reb. → epo_panel["EPO_w_0.75"]
  TSMOM Vol-skaleret        → backtest_vol_scaled (INDMOM-anchor)
  TSMOM EW                  → backtest_simple_momentum_ew

Ændring ift. Equity_2: signal beregnes på råafkast (monthly),
ikke merafkast — se kommentar i compute_tsmom_signal nedenfor.

Output:
  - Ét plot: kumuleret merafkast 2023–2025 med OOS-highlight
  - Én SR-tabel: fuld periode (2020–2025) + OOS (2023–2025)

Krav: køres EFTER Equity_2.main() har returneret results-dict,
eller køres selvstændigt (genberegner alt fra bunden).
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from Equity_1 import (
    compute_risk_model,
    build_epo_panel,
    get_monthly_risk,
    backtest_strategy,
    performance_summary,
    subset,
    CORR_PRESHRINK,
    GAMMA,
    RISK_WINDOW,
    LOOKBACK_MONTHS,
    CANDIDATE_WS,
)

from Best_stocks_from_industry import DAILY_RETS_PATH, SECTOR_PATH

from moment_mod_EPO_ind import (
backtest_tsmom_vol,
)

from Stock_Data import (
load_data,
to_monthly_returns,
get_rf_monthly,
compute_monthly_excess,
compute_tsmom_signal,
backtest_simple_momentum_ew,
backtest_simple_momentum
)

# ── Parametre ─────────────────────────────────────────────────────────────────

W          = 0.75
VOL_TARGET = 0.40        # 40% p.a. — Pedersen (2021)
START      = "2010-01-01"
END        = "2025-12-31"
OOS_START  = "2020-01-01"

PLOT_START = "2023-01-01"
PLOT_END   = "2025-12-31"

C_EPO    = "#1a4f8a"   # mørk blå
C_TSMVOL = "#e67e22"   # orange
C_TSMEW  = "#c0392b"   # rød
C_TSMLONGSHORT = "#006400" #mørkegrøn


# ── 1. Data ───────────────────────────────────────────────────────────────────

def load_and_prepare():
    """
    Returnerer:
      monthly        — råafkast  (til signal)
      monthly_excess — merafkast (til kovarians og backtest)
      ticker_to_sector
    """
    if not all(os.path.exists(p) for p in [DAILY_RETS_PATH, SECTOR_PATH]):
        raise FileNotFoundError("Kør Best_stocks_from_industry.py først.")

    daily = pd.read_parquet(DAILY_RETS_PATH)
    sec_df = pd.read_csv(SECTOR_PATH)
    ticker_to_sector = dict(zip(sec_df["ticker"], sec_df["sector"]))

    # Daglige → månedlige råafkast
    monthly = (1 + daily).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M").to_timestamp("M")

    # Risikofri rente
    rf_df = get_monthly_risk()
    rf_df.index = rf_df.index.to_period("M").to_timestamp("M")
    rf = rf_df["RF"].loc[START:END].dropna()

    # Månedlige merafkast
    monthly_excess = monthly.sub(rf.reindex(monthly.index).ffill(), axis=0)

    monthly        = monthly.loc[START:END]
    monthly_excess = monthly_excess.loc[START:END]

    print(f"Data: {monthly_excess.shape[1]} aktier  |  "
          f"{monthly_excess.index[0].date()} → {monthly_excess.index[-1].date()}")

    return monthly, monthly_excess, ticker_to_sector


# ── SR-tabel ───────────────────────────────────────────────────────────────

def print_sr_table(strategies: dict, full_start: str, full_end: str,
                   oos_start: str, oos_end: str) -> None:
    """Printer to blokke: fuld OOS-periode + plotperiode 2023-2025."""

    def block(start, end, title):
        s_dt = pd.to_datetime(start)
        e_dt = pd.to_datetime(end)
        rows = {
            k: performance_summary(
                strategies[k].loc[s_dt:e_dt].dropna(), k)
            for k in strategies
        }
        best = max(rows, key=lambda k: rows[k]["Sharpe"])
        print(f"\n{'═'*62}")
        print(f"  {title}")
        print(f"{'═'*62}")
        print(f"  {'Strategi':<30} {'Ann.afk':>9} {'Vol':>8} {'SR':>8}")
        print(f"  {'-'*58}")
        for k, r in rows.items():
            marker = " ◀" if k == best else ""
            print(
                f"  {k:<30} "
                f"{r['Ann. Return']:>8.2%} "
                f"{r['Ann. Vol']:>7.2%} "
                f"{r['Sharpe']:>8.3f}"
                f"{marker}"
            )
        print(f"{'═'*62}")

    block(full_start, full_end,
          f"OOS-periode  {full_start[:4]}–{full_end[:4]}")
    block(oos_start, oos_end,
          f"Plotperiode  {oos_start[:4]}–{oos_end[:4]}")


# ── 6. Plot ───────────────────────────────────────────────────────────────────

def plot(strategies: dict, plot_start: str, plot_end: str) -> None:
    s_dt = pd.to_datetime(plot_start)
    e_dt = pd.to_datetime(plot_end)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("white")

    lines = [
        (list(strategies.keys())[0], C_EPO,    2.2, "-"),
        (list(strategies.keys())[1], C_TSMVOL, 1.8, "--"),
        (list(strategies.keys())[2], C_TSMEW,  1.6, ":"),
        (list(strategies.keys())[3], C_TSMLONGSHORT, 2.2, "-"),
    ]

    for key, color, lw, ls in lines:
        s   = strategies[key].loc[s_dt:e_dt].dropna()
        cum = (1 + s).cumprod() - 1
        sr  = performance_summary(s, "")["Sharpe"]
        ax.plot(cum.index, cum * 100,
                color=color, lw=lw, ls=ls, label=f"{key}")
        ax.annotate(
            f"{cum.iloc[-1]*100:.0f}%",
            xy=(cum.index[-1], cum.iloc[-1] * 100),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color=color, fontweight="bold", va="center",
        )

    # OOS-highlight 2023-2025
    ax.axvspan(
        pd.to_datetime("2023-01-01"),
        pd.to_datetime("2025-12-31"),
        alpha=0.08, color=C_EPO, zorder=0)
    ymin, ymax = ax.get_ylim()
    ax.text(
        pd.to_datetime("2023-01-01") + pd.Timedelta(days=30),
        ymax * 0.97,
        "OOS  2023–2025",
        fontsize=8, color=C_EPO, va="top",
        style="italic", fontweight="bold",
    )

    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_ylabel("Kumuleret merafkast (%)", fontsize=10)
    ax.set_title(
        f"EPO w={W} vs. TSMOM — Yahoo Finance aktier (2020–2025)\n"
        f"Samme TSMOM-signal",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="#ccc",
              loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    plt.tight_layout()
    plt.savefig("epo_vs_tsmom_yf_2023_2025.png",
                dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\nGemt: epo_vs_tsmom_yf_2023_2025.png")
    plt.show()


# ── 7. Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print(f"EPO w={W} vs. TSMOM — Yahoo Finance aktier")
    print(f"Signal      : TSMOM på råafkast (monthly)")
    print(f"Kovarians   : beregnet på merafkast (monthly_excess)")
    print(f"Risikomodel : {RISK_WINDOW}m vindue, θ={CORR_PRESHRINK}")
    print("=" * 62)

    # 1. Data
    daily, daily_prices, ticker_to_sector = load_data()
    monthly = to_monthly_returns(daily)
    rf = get_rf_monthly(START, END)
    monthly_excess = compute_monthly_excess(monthly, rf)
    print(f"\nAktier: {monthly_excess.shape[1]}  |  "
          f"Periode: {monthly_excess.index[0].date()} → "
          f"{monthly_excess.index[-1].date()}")

    # 2. Signal
    print("\nBeregner signal...")
    tsmom = compute_tsmom_signal(monthly_excess, ticker_to_sector, LOOKBACK_MONTHS)

    # 3. Risikomodel
    print("\nBygger risikomodel (60m, 5% pre-shrinkage)...")
    corr_shrunk, vols = compute_risk_model(
        monthly_excess, window=RISK_WINDOW,
        theta=CORR_PRESHRINK, verbose=True)

    # ── EPO panel → udtræk w=0.75 ─────────────────────────────
    print("\nBygger EPO panel (alle w-værdier)...")
    epo_panel = build_epo_panel(
        monthly_excess, tsmom, corr_shrunk, vols, GAMMA, CANDIDATE_WS)
    epo_monthly_reb = epo_panel[f"EPO_w_{W:.2f}"]
    epo_monthly_reb.name = f"EPO w={W} (månedlig reb.)"

    # ── TSMOM benchmarks ──────────────────────────────────────
    print(f"Backtester TSMOM Vol (σ_target={VOL_TARGET*100:.0f}%)...")
    tsmom_vol = backtest_tsmom_vol(monthly_excess, tsmom)
    tsmom_vol.name = f"TSMOM Vol (σ_t={VOL_TARGET*100:.0f}%)"

    print("Backtester TSMOM EW...")
    tsmom_ew = backtest_simple_momentum_ew(monthly_excess, monthly)
    tsmom_ew.name = "TSMOM EW (1/N)"

    print("Backtester TSMOM long short fra stock_data")
    tsmom_stock = backtest_simple_momentum(monthly_excess, monthly, 50)
    tsmom_stock.name = "TSMOM Long/Short (50)"

    # ── Strategier samlet ─────────────────────────────────────
    strategies = {
        epo_monthly_reb.name: epo_monthly_reb,
        tsmom_vol.name:       tsmom_vol,
        tsmom_ew.name:        tsmom_ew,
        tsmom_stock.name:     tsmom_stock,
    }

    # ── Tabel ─────────────────────────────────────────────────
    print_sr_table(
        strategies,
        full_start=PLOT_START, full_end=END,
        oos_start="2023-01-01", oos_end=PLOT_END,
    )

    # ── Plot ──────────────────────────────────────────────────
    plot(strategies, plot_start=PLOT_START, plot_end=PLOT_END)


if __name__ == "__main__":
    main()