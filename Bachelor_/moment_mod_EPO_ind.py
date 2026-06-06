"""
EPO w=0.75 vs. TSMOM Vol-skaleret vs. TSMOM EW — tre delperioder
=================================================================
Bruger Fama-French 49 Industry månedlige afkast.

Formål: isoler værdien af korrelationsshrinkagen i EPO.

Alle tre strategier bruger SAMME signal — compute_tsmom_signal —
så forskellen mellem strategierne udelukkende skyldes
porteføljekonstruktionen:

  TSMOM EW  : signal → equal weight (1/N per retning)
  Vol : signal → vol-skaleret (Pedersen: 1/n × σ_target/σ_i)
  EPO w=0.75: signal → korrelationsshrinkage oven på vol-skaleringen

Sekvensen EW → Vol → EPO isolerer hvert lags bidrag.
Forskellen Vol → EPO er udelukkende korrelationsstrukturens bidrag.

Krav:
  - Equity_1.py i samme mappe
  - Data: 49_Industry_monthly.csv  +  Månedlig_rf.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Importer fra Equity_1 ─────────────────────────────────────────────────────

from Equity_1 import (
    compute_risk_model,
    compute_risk_model_ewm,
    backtest_strategy,
    backtest_epo_fixed_w,
    epo_weights,
    performance_summary,
    subset,
    get_monthly_return,
    get_monthly_risk,
    calculate_monthly_excess_returns,
    CORR_PRESHRINK,
    GAMMA,
    RISK_WINDOW,
    LOOKBACK_MONTHS,
)

# ── Parametre ─────────────────────────────────────────────────────────────────

W          = 0.75
VOL_TARGET = 0.40        # 40% p.a. — Pedersen (2021)
VOL_WINDOW = RISK_WINDOW # 24m rullende std

FULL_START = "1980-01-01"
FULL_END   = "2025-12-31"

PERIODS = [
    ("1998-01-01", "2002-12-31", "1998–2002"),
    ("2007-01-01", "2009-12-31", "2007–2009"),
    ("2020-01-01", "2022-12-31", "2020–2022"),
]

C_EPO    = "#1a4f8a"  # mørk blå  — EPO w=0.75 (rullende)
C_EPOEWM = "#6c3483"  # lilla     — EPO w=0.75 (EWMA)
C_TSMVOL = "#e67e22"  # orange    — TSMOM vol-skaleret
C_TSMEW  = "#c0392b"  # rød       — TSMOM equal-weighted


# ── Fælles TSMOM signal ───────────────────────────────────────────────────────

def compute_tsmom_signal(
    excess: pd.DataFrame,
    lookback: int = LOOKBACK_MONTHS,
) -> pd.DataFrame:
    """
    TSMOM med asymmetrisk short-filter — bruges af ALLE tre strategier:
      - Long:  12m afkast > 0  → signal = +afkast / n
      - Short: 12m afkast < 0  → signal = +afkast / n  (negativt tal)
      - Nul:   12m afkast = 0  → ingen position

    Short-siden aktiveres KUN hvis industrien har negativt absolut afkast.
    Dermed shortes der ikke industrier der blot stiger langsommere end andre.

    Returnerer en DataFrame med signalværdier klar til brug i:
      - backtest_epo_fixed_w  (som mu-input)
      - backtest_tsmom_ew     (retning bruges til equal-weighting)
      - backtest_tsmom_vol    (retning bruges til vol-skalering)
    """
    roll = excess.rolling(window=lookback, min_periods=lookback).sum()
    out  = pd.DataFrame(np.nan, index=roll.index, columns=roll.columns)

    for date, row in roll.iterrows():
        avail = row.dropna()
        if len(avail) < 2:
            continue

        long_mask  = avail > 0
        short_mask = avail < 0

        signal = pd.Series(0.0, index=avail.index)
        signal[long_mask]  = avail[long_mask]
        signal[short_mask] = avail[short_mask]

        n = len(avail)
        out.loc[date, signal.index] = signal.values / n

    return out


# ── TSMOM Equal-Weighted ──────────────────────────────────────────────────────

def backtest_tsmom_ew(
    excess: pd.DataFrame,
    tsmom:  pd.DataFrame,
) -> pd.Series:
    """
    TSMOM Equal-Weighted:
      Retningen fra TSMOM-signalet → equal weight 1/N per industri.
      Alle long-industrier får +1/N, alle short-industrier -1/N.
      N = antal industrier med gyldigt signal denne måned.

    Ingen risikojustering — isolerer reningssignalets bidrag alene.
    """
    def weight_fn(date: pd.Timestamp) -> pd.Series:
        if date not in tsmom.index:
            return pd.Series(dtype=float)
        sig = tsmom.loc[date].dropna()
        sig = sig[sig != 0]
        if sig.empty:
            return pd.Series(dtype=float)

        n   = len(sig)
        wts = pd.Series(0.0, index=sig.index)
        wts[sig > 0] =  1.0 / n
        wts[sig < 0] = -1.0 / n
        return wts[wts != 0]

    return backtest_strategy(excess, weight_fn, name="TSMOM EW")


# ── TSMOM Volatilitetsskaleret ────────────────────────────────────────────────

def backtest_tsmom_vol(
    excess:     pd.DataFrame,
    tsmom:      pd.DataFrame,
    vol_window: int   = VOL_WINDOW,
    vol_target: float = VOL_TARGET,
) -> pd.Series:
    """
    TSMOM Volatilitetsskaleret — Pedersens equal-volatility-weighted formel:

      w_i = (1/n_t) × (σ_target / √12) / σ_i × sign(signal_i)

    Retningen kommer fra compute_tsmom_signal (samme som EPO og EW).
    σ_i = 24m rullende månedlig standardafvigelse for industri i.
    1/n_t begrænser naturligt den samlede leverage.

    Kontrollerer for individuelle risikoforskelle — men ignorerer stadig
    korrelationsstrukturen. Det er præcis hvad EPO tilføjer oven på dette.
    """
    roll_vol       = excess.rolling(window=vol_window, min_periods=vol_window).std()
    target_monthly = vol_target / np.sqrt(12)

    def weight_fn(date: pd.Timestamp) -> pd.Series:
        if date not in tsmom.index:
            return pd.Series(dtype=float)

        sig     = tsmom.loc[date].dropna()
        sig     = sig[sig != 0]
        vol_row = roll_vol.loc[date].dropna() if date in roll_vol.index else pd.Series(dtype=float)

        common = sig.index.intersection(vol_row.index)
        if len(common) < 2:
            return pd.Series(dtype=float)

        sig_c = sig[common]
        vol_c = vol_row[common].replace(0, np.nan).dropna()
        common = sig_c.index.intersection(vol_c.index)
        if len(common) < 2:
            return pd.Series(dtype=float)

        sig_c = sig_c[common]
        vol_c = vol_c[common]

        # Pedersen: (1/n_t) × (σ_target_monthly / σ_i) × sign(signal)
        n_t = len(common)
        wts = (1.0 / n_t) * (target_monthly / vol_c) * np.sign(sig_c)
        wts = wts[wts != 0]

        if wts.empty:
            return pd.Series(dtype=float)
        return wts

    return backtest_strategy(excess, weight_fn, name="TSMOM Vol")


# ── Hjælpefunktion ────────────────────────────────────────────────────────────

def max_drawdown(series: pd.Series) -> float:
    """Maksimalt kumuleret drawdown (negativ værdi)."""
    s   = series.dropna()
    cum = (1 + s).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()
    return dd.min()


def win_rate(series: pd.Series) -> float:
    """Andel af måneder med positivt afkast."""
    s = series.dropna()
    return (s > 0).sum() / len(s)


def summarise(series: pd.Series, label: str) -> dict:
    d = performance_summary(series, label)
    d["Cum"]     = (1 + series.dropna()).prod() - 1
    d["WinRate"] = win_rate(series)
    d["MaxDD"]   = max_drawdown(series)
    return d


# ── Long-Only EPO ─────────────────────────────────────────────────────────────

def epo_weights_long_only(signal, corr, vols, gamma, w):
    raw = epo_weights(signal, corr, vols, gamma, w)
    if len(raw) == 0:
        return pd.Series(dtype=float)
    long_only = raw.clip(lower=0)
    total = long_only.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return long_only / total


def backtest_epo_long_only(excess, tsmom, corr_dict, vols_dict, gamma, w):
    risk_dates = set(corr_dict)
    sig_dates  = set(tsmom.index)

    def weight_fn(date):
        if date not in risk_dates or date not in sig_dates:
            return pd.Series(dtype=float)
        return epo_weights_long_only(
            tsmom.loc[date], corr_dict[date], vols_dict[date], gamma, w)

    return backtest_strategy(excess, weight_fn, name=f"EPO Long-Only w={w}")


# ── Industribidrag og eksponering ──────────────────────────────────────────────

def print_industry_contribution_table(excess, tsmom, corr_dict, vols_dict,
                                       gamma, w=0.75,
                                       start="1980-01-01", end="2025-12-31",
                                       long_only=False):
    """
    Gennemsnitligt årligt afkastbidrag per industri (weight × return)
    over hele perioden. Total vises som geometrisk kumuleret afkast.
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    idx  = excess.index

    risk_dates = set(corr_dict)
    sig_dates  = set(tsmom.index)

    contrib   = {}
    port_rets = []

    for t in range(len(idx) - 1):
        date     = idx[t]
        nxt_date = idx[t + 1]
        if nxt_date < s or nxt_date > e:
            continue
        if date not in risk_dates or date not in sig_dates:
            continue

        wts = (epo_weights_long_only(tsmom.loc[date], corr_dict[date],
                                     vols_dict[date], gamma, w)
               if long_only else
               epo_weights(tsmom.loc[date], corr_dict[date],
                           vols_dict[date], gamma, w))
        if len(wts) == 0:
            continue

        r     = excess.loc[nxt_date].reindex(wts.index).dropna()
        w_aln = wts.reindex(r.index).dropna()
        r_aln = r.reindex(w_aln.index)

        port_rets.append((w_aln * r_aln).sum())
        for ind in w_aln.index:
            contrib[ind] = contrib.get(ind, 0.0) + w_aln[ind] * r_aln[ind]

    n_months  = len(port_rets)
    n_years   = n_months / 12.0
    geo_cum   = np.prod([1 + r for r in port_rets]) - 1 if port_rets else np.nan
    geo_ann   = (1 + geo_cum) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    # Aritmetisk bidrag per industri (sommer til total)
    contrib_ann = {ind: v / n_years for ind, v in contrib.items()}
    total_arith_ann = sum(contrib_ann.values())
    total_arith_cum = sum(contrib.values())
    inds = sorted(contrib_ann, key=contrib_ann.get, reverse=True)

    label = f"EPO LONG-ONLY w={int(w*100)}%" if long_only else f"EPO w={int(w*100)}%"
    col_w = 18
    width = 24 + col_w * 2
    print("\n" + "=" * width)
    print(f"GNS. INDUSTRIBIDRAG — {label}  ({start[:4]}–{end[:4]})")
    print(f"Bidrag er aritmetiske (Σ w·r) og summer til total.")
    print("=" * width)
    print(f"  {'Industri':<22} {'Gns. årligt bidrag':>{col_w}} {'Aritm. kum. bidrag':>{col_w}}")
    print("-" * width)
    for ind in inds:
        print(f"  {ind:<22} {contrib_ann[ind]:>{col_w}.4%} {contrib[ind]:>{col_w}.2%}")
    print("-" * width)
    print(f"  {'Sum (aritm.)':<22} {total_arith_ann:>{col_w}.4%} {total_arith_cum:>{col_w}.2%}")
    print(f"  {'Porto. geo. ann.':<22} {geo_ann:>{col_w}.4%} {geo_cum:>{col_w}.2%}")
    print("=" * width)


def print_industry_exposure_table(excess, tsmom, corr_dict, vols_dict,
                                   gamma, w=0.75,
                                   start="1980-01-01", end="2025-12-31",
                                   long_only=False):
    """
    Gennemsnitlig brutto-eksponering per industri over hele perioden
    (|vægt| gennemsnittet over alle måneder).
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    idx  = excess.index

    risk_dates = set(corr_dict)
    sig_dates  = set(tsmom.index)
    exposure   = {}
    n_months   = 0

    for t in range(len(idx) - 1):
        date     = idx[t]
        nxt_date = idx[t + 1]
        if nxt_date < s or nxt_date > e:
            continue
        if date not in risk_dates or date not in sig_dates:
            continue

        wts = (epo_weights_long_only(tsmom.loc[date], corr_dict[date],
                                     vols_dict[date], gamma, w)
               if long_only else
               epo_weights(tsmom.loc[date], corr_dict[date],
                           vols_dict[date], gamma, w))
        if len(wts) == 0:
            continue

        n_months += 1
        for ind, weight in wts.items():
            exposure[ind] = exposure.get(ind, 0.0) + abs(weight)

    if n_months == 0:
        print("Ingen data.")
        return

    exp_mean = {ind: v / n_months for ind, v in exposure.items()}
    inds = sorted(exp_mean, key=exp_mean.get, reverse=True)
    ge   = sum(exp_mean.values())

    label = f"EPO LONG-ONLY w={int(w*100)}%" if long_only else f"EPO w={int(w*100)}%"
    col_w = 20
    width = 24 + col_w
    print("\n" + "=" * width)
    print(f"GNS. INDUSTRIEKSPONERING (BRUTTO) — {label}  ({start[:4]}–{end[:4]})")
    print("=" * width)
    print(f"  {'Industri':<22} {'Gns. eksponering':>{col_w}}")
    print("-" * width)
    for ind in inds:
        print(f"  {ind:<22} {exp_mean[ind]:>{col_w}.4%}")
    print("-" * width)
    print(f"  {'Total (GE)':<22} {ge:>{col_w}.4%}")
    print("=" * width)


# ── Visualisering ─────────────────────────────────────────────────────────────

def plot_subperiods(all_epo, all_epo_ewm, all_ew, all_vol, labels):
    n = len(labels)
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#f8f9fa")

    gs = gridspec.GridSpec(
        2, n, figure=fig,
        height_ratios=[2.2, 1.5],
        hspace=0.50, wspace=0.28,
        top=0.88, bottom=0.08, left=0.06, right=0.97,
    )

    for col, (epo, epo_ewm, ew, vol, label) in enumerate(
        zip(all_epo, all_epo_ewm, all_ew, all_vol, labels)
    ):
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.set_facecolor("white")

        for s, color, lw, ls in [
            (epo,     C_EPO,    2.2, "-"),
            (epo_ewm, C_EPOEWM, 2.0, "-."),
            (vol,     C_TSMVOL, 1.8, "--"),
            (ew,      C_TSMEW,  1.6, ":"),
        ]:
            cum = (1 + s.dropna()).cumprod() - 1
            ax_top.plot(cum.index, cum * 100, color=color, lw=lw, ls=ls)
            ax_top.annotate(
                f"{cum.iloc[-1]*100:.0f}%",
                xy=(cum.index[-1], cum.iloc[-1] * 100),
                xytext=(5, 0), textcoords="offset points",
                fontsize=7.5, color=color, fontweight="bold", va="center",
            )

        ax_top.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax_top.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax_top.set_title(label, fontsize=12, fontweight="bold", pad=6)
        if col == 0:
            ax_top.set_ylabel("Kumuleret merafkast", fontsize=9)
        ax_top.tick_params(axis="x", rotation=30, labelsize=8)
        ax_top.tick_params(axis="y", labelsize=8)
        ax_top.spines[["top", "right"]].set_visible(False)
        ax_top.grid(axis="y", alpha=0.3, lw=0.6)

        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.set_facecolor("white")

        for s, color, lw, ls in [
            (epo,     C_EPO,    2.0, "-"),
            (epo_ewm, C_EPOEWM, 1.8, "-."),
            (vol,     C_TSMVOL, 1.6, "--"),
            (ew,      C_TSMEW,  1.4, ":"),
        ]:
            roll_sr = (
                s.rolling(12).mean() * 12
                / (s.rolling(12).std() * np.sqrt(12))
            ).dropna()
            ax_bot.plot(roll_sr.index, roll_sr,
                        color=color, lw=lw, ls=ls, alpha=0.9)

        ax_bot.axhline(0, color="black", lw=0.6, alpha=0.4)
        if col == 0:
            ax_bot.set_ylabel("SR (rullende 12m)", fontsize=9)
        ax_bot.tick_params(axis="x", rotation=30, labelsize=8)
        ax_bot.tick_params(axis="y", labelsize=8)
        ax_bot.spines[["top", "right"]].set_visible(False)
        ax_bot.grid(axis="y", alpha=0.3, lw=0.6)

    legend_elements = [
        Line2D([0], [0], color=C_EPO,    lw=2.2, ls="-",
               label=f"EPO w={W} (rullende {RISK_WINDOW}m kovarians)"),
        Line2D([0], [0], color=C_EPOEWM, lw=2.0, ls="-.",
               label=f"EPO w={W} (EWMA kovarians, span={RISK_WINDOW}m)"),
        Line2D([0], [0], color=C_TSMVOL, lw=1.8, ls="--",
               label=f"TSMOM Vol-skaleret (1/n × σ_target/σ_i, ingen korr.)"),
        Line2D([0], [0], color=C_TSMEW,  lw=1.6, ls=":",
               label="TSMOM Equal-Weighted (1/N, ingen risikojust.)"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=2,
        fontsize=9, frameon=True, framealpha=0.9,
        edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.975),
    )
    fig.suptitle(
        f"EPO w={W} (rullende vs. EWMA) vs. TSMOM — Tre delperioder\n"
        "Fama-French 49 industrier  |  Samme signal — forskellig porteføljekonstruktion",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.savefig("epo_vs_tsmom_subperiods.png",
                dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Figur gemt: epo_vs_tsmom_subperiods.png")
    plt.show()


def plot_combined_cumulative(epo_full, epo_ewm_full, ew_full, vol_full):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("white")

    for s, color, lw, ls, label in [
        (epo_full,     C_EPO,    2.2, "-",  f"EPO w={W} (rullende {RISK_WINDOW}m kovarians)"),
        (epo_ewm_full, C_EPOEWM, 2.0, "-.", f"EPO w={W} (EWMA kovarians, span={RISK_WINDOW}m)"),
        (vol_full,     C_TSMVOL, 1.8, "--", f"TSMOM Vol-skaleret (σ_target={VOL_TARGET*100:.0f}%)"),
        (ew_full,      C_TSMEW,  1.6, ":",  "TSMOM Equal-Weighted (1/N)"),
    ]:
        cum = (1 + s.dropna()).cumprod() - 1
        ax.plot(cum.index, cum * 100, color=color, lw=lw, ls=ls, label=label)

    colors_band = ["#d4e6f1", "#d5f5e3", "#fdebd0"]
    ymax = ax.get_ylim()[1]
    for (start, end, lbl), bc in zip(PERIODS, colors_band):
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                   alpha=0.25, color=bc, zorder=0)
        ax.text(
            pd.to_datetime(start) + (pd.to_datetime(end) - pd.to_datetime(start)) / 2,
            ymax * 0.97,
            lbl, ha="center", va="top", fontsize=8,
            color="#444", style="italic",
        )

    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_ylabel("Kumuleret merafkast (%)", fontsize=10)
    ax.set_title(
        f"EPO w={W} (rullende vs. EWMA) vs. TSMOM — Fuld periode\n"
        f"Fama-French 49 industrier "
        f"({pd.to_datetime(FULL_START).year}–{pd.to_datetime(FULL_END).year})  "
        f"|  Rullende = {RISK_WINDOW}m vindue  |  EWMA span={RISK_WINDOW}m",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="#ccc")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    plt.tight_layout()
    plt.savefig("epo_vs_tsmom_full.png",
                dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Figur gemt: epo_vs_tsmom_full.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(f"EPO w={W} vs. TSMOM Vol vs. TSMOM EW")
    print(f"Formål     : isoler værdien af korrelationsshrinkage")
    print(f"Signal     : identisk TSMOM-signal for alle tre strategier")
    print(f"Datasæt    : Fama-French 49 industrier")
    print(f"Risikomodel: {RISK_WINDOW}m vindue, θ={CORR_PRESHRINK}  (Equity_1)")
    print(f"Vol-target : {VOL_TARGET*100:.0f}% p.a. | Vol-vindue: {VOL_WINDOW}m")
    print("=" * 65)

    # ── 1. Data ────────────────────────────────────────────────
    monthly_returns = get_monthly_return()
    rf_monthly      = get_monthly_risk()
    excess          = calculate_monthly_excess_returns(monthly_returns, rf_monthly)

    print(f"\nData: {excess.shape[1]} industrier  |  "
          f"{excess.index[0].date()} → {excess.index[-1].date()}")

    excess = excess.loc[FULL_START:FULL_END]

    # ── 2. Fælles TSMOM signal ─────────────────────────────────
    print("\nBeregner TSMOM-signal (deles af alle tre strategier)...")
    tsmom = compute_tsmom_signal(excess, lookback=LOOKBACK_MONTHS)

    # ── 3. Risikomodeller ──────────────────────────────────────
    print(f"\nBygger rullende risikomodel ({RISK_WINDOW}m, θ={CORR_PRESHRINK})...")
    corr_shrunk, vols = compute_risk_model(
        excess, window=RISK_WINDOW,
        theta=CORR_PRESHRINK, verbose=True
    )

    print(f"\nBygger EWMA risikomodel (span={RISK_WINDOW}m, θ={CORR_PRESHRINK})...")
    corr_ewm, vols_ewm = compute_risk_model_ewm(
        excess, span=RISK_WINDOW,
        min_periods=max(12, RISK_WINDOW // 2),
        theta=CORR_PRESHRINK, verbose=True
    )

    # ── 4. Backtest ────────────────────────────────────────────
    print(f"\nBacktester EPO w={W} (rullende kovarians)...")
    epo_full = backtest_epo_fixed_w(
        excess, tsmom, corr_shrunk, vols, GAMMA, w=W
    )
    epo_full.name = f"EPO w={W}"

    print(f"Backtester EPO w={W} (EWMA kovarians)...")
    epo_ewm_full = backtest_epo_fixed_w(
        excess, tsmom, corr_ewm, vols_ewm, GAMMA, w=W
    )
    epo_ewm_full.name = f"EPO w={W} EWMA"

    print(f"Backtester EPO Long-Only w={W}...")
    epo_lo_full = backtest_epo_long_only(excess, tsmom, corr_shrunk, vols, GAMMA, w=W)

    print("Backtester TSMOM Equal-Weighted (samme signal, 1/N vægte)...")
    ew_full = backtest_tsmom_ew(excess, tsmom)

    print(f"Backtester TSMOM Vol-skaleret "
          f"(samme signal, σ_target={VOL_TARGET*100:.0f}%)...")
    vol_full = backtest_tsmom_vol(
        excess, tsmom,
        vol_window=VOL_WINDOW,
        vol_target=VOL_TARGET,
    )

    # ── 5. Delperioder ─────────────────────────────────────────
    all_epo, all_epo_ewm, all_ew, all_vol, labels = [], [], [], [], []

    for start, end, label in PERIODS:
        epo_sub     = subset(epo_full,     start, end).dropna()
        epo_ewm_sub = subset(epo_ewm_full, start, end).dropna()
        epo_lo_sub  = subset(epo_lo_full,  start, end).dropna()
        ew_sub      = subset(ew_full,      start, end).dropna()
        vol_sub     = subset(vol_full,     start, end).dropna()

        all_epo.append(epo_sub)
        all_epo_ewm.append(epo_ewm_sub)
        all_ew.append(ew_sub)
        all_vol.append(vol_sub)
        labels.append(label)

        print(f"\n{'═'*75}")
        print(f"  {label}")
        print(f"{'═'*75}")
        print(f"  {'Strategi':<42} {'Ann.afk':>9} {'Vol':>8} "
              f"{'SR':>7} {'Kum.':>9} {'WinRate':>8} {'MaxDD':>8}")
        print(f"  {'-'*83}")
        for s, lbl in [
            (epo_sub,     f"EPO Long-Short w={W} (rullende)"),
            (epo_ewm_sub, f"EPO Long-Short w={W} (EWMA)"),
            (epo_lo_sub,  f"EPO Long-Only  w={W}"),
            (vol_sub,     f"TSMOM Vol (σ_t={VOL_TARGET*100:.0f}%)"),
            (ew_sub,      "TSMOM EW (1/N)"),
        ]:
            r = summarise(s, lbl)
            print(
                f"  {r['Strategy']:<42} "
                f"{r['Ann. Return']:>8.2%} "
                f"{r['Ann. Vol']:>7.2%} "
                f"{r['Sharpe']:>7.3f} "
                f"{r['Cum']:>8.1%} "
                f"{r['WinRate']:>7.1%} "
                f"{r['MaxDD']:>8.1%}"
            )

    # ── 6. Samlet periode ──────────────────────────────────────
    print(f"\n{'═'*75}")
    print(f"  Samlet periode ({FULL_START[:4]}–{FULL_END[:4]})")
    print(f"{'═'*75}")
    print(f"  {'Strategi':<42} {'Ann.afk':>9} {'Vol':>8} "
          f"{'SR':>7} {'Kum.':>9} {'WinRate':>8} {'MaxDD':>8}")
    print(f"  {'-'*83}")
    for s, lbl in [
        (epo_full,     f"EPO Long-Short w={W} (rullende)"),
        (epo_ewm_full, f"EPO Long-Short w={W} (EWMA)"),
        (epo_lo_full,  f"EPO Long-Only  w={W}"),
        (vol_full,     f"TSMOM Vol (σ_t={VOL_TARGET*100:.0f}%)"),
        (ew_full,      "TSMOM EW (1/N)"),
    ]:
        r = summarise(s.dropna(), lbl)
        print(
            f"  {r['Strategy']:<42} "
            f"{r['Ann. Return']:>8.2%} "
            f"{r['Ann. Vol']:>7.2%} "
            f"{r['Sharpe']:>7.3f} "
            f"{r['Cum']:>8.1%} "
            f"{r['WinRate']:>7.1%} "
            f"{r['MaxDD']:>8.1%}"
        )

    # ── 7. Industribidrag og eksponering 1980-2025 ─────────────
    print(f"\nBeregner industribidrag og eksponering ({FULL_START[:4]}–{FULL_END[:4]})...")

    for lo in [False, True]:
        print_industry_contribution_table(
            excess, tsmom, corr_shrunk, vols, GAMMA, w=W,
            start=FULL_START, end=FULL_END, long_only=lo)
        print_industry_exposure_table(
            excess, tsmom, corr_shrunk, vols, GAMMA, w=W,
            start=FULL_START, end=FULL_END, long_only=lo)

    # ── 7. Figurer ─────────────────────────────────────────────
    print("\nGenererer figurer...")
    plot_subperiods(all_epo, all_epo_ewm, all_ew, all_vol, labels)
    plot_combined_cumulative(epo_full, epo_ewm_full, ew_full, vol_full)
    print("\nFærdig.")


if __name__ == "__main__":
    main()