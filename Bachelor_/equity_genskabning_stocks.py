"""
equity_genskabning_stocks.py
════════════════════════════
Genskaber 'equity_genskabning.py' på aktieuniverset (Yahoo Finance).

Kører 8 konfigurationer (varierende risikovindue, signalvindue, signaltype
og forankret EPO) og producerer en samlet Sharpe-tabel til sammenligning.

Data   : Yahoo Finance enkeltaktier (via Best_stocks_from_industry.py)
Periode: 2010-2025  |  OOS-backtest: 2020-2025
Signal : TSMOM (standard for aktieuniverset) + XSMOM til robusthedstjek
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Equity_1 import (
    compute_xsmom,
    compute_risk_model,
    compute_risk_model_ewm,
    backtest_equal_weight,
    backtest_indmom,
    backtest_mvo_no_shrink,
    build_epo_panel,
    build_dynamic_oos_epo,
    sharpe_ratio,
    subset,
    epo_weights,
    backtest_strategy,
    GAMMA,
    CANDIDATE_WS,
)

from Stock_Data import (
    load_data,
    to_monthly_returns,
    compute_monthly_excess,
    get_rf_monthly,
    compute_tsmom_signal,
)

from equity_genskabning import (
    epo_anchored_weights,
    backtest_epo_anchored,
    build_anchored_epo_panel,
)

# ── Konstanter ────────────────────────────────────────────────
DATA_START      = "2010-01-01"
DATA_END        = "2025-12-31"
BACKTEST_START  = "2020-01-01"
BACKTEST_END    = "2025-12-31"
CORR_PRESHRINK  = 0.05
MIN_HISTORY_OOS = 1

EQUITY_CONFIGS = [
    {"name": "Equity 1", "risk_window": 60, "signal_window": 12, "signal_type": "TSMOM", "ewm": False},
    {"name": "Equity 2", "risk_window": 36, "signal_window": 12, "signal_type": "TSMOM", "ewm": False},
    {"name": "Equity 3", "risk_window": 24, "signal_window": 12, "signal_type": "TSMOM", "ewm": False},
    {"name": "Equity 4", "risk_window": 24, "signal_window": 24, "signal_type": "TSMOM", "ewm": False},
    {"name": "Equity 5", "risk_window": 24, "signal_window":  6, "signal_type": "TSMOM", "ewm": False},
    {"name": "Equity 6", "risk_window": 24, "signal_window":  3, "signal_type": "TSMOM", "ewm": False},
    {"name": "Equity 7", "risk_window": 24, "signal_window": 12, "signal_type": "XSMOM", "ewm": False},
    {"name": "Equity 8", "risk_window": 24, "signal_window": 12, "signal_type": "TSMOM", "ewm": False},  # forankret EPO
    {"name": "Equity 9", "risk_window": 60, "signal_window": 12, "signal_type": "TSMOM", "ewm": True},   # EWMA kovarians
]


# ── Performance ───────────────────────────────────────────────

def performance_summary(r: pd.Series, name: str) -> dict:
    r = r.dropna()
    return {
        "Strategy": name,
        "Sharpe":   round(sharpe_ratio(r), 3),
    }


# ── Enkelt konfiguration ──────────────────────────────────────

def run_single_equity(monthly_excess, ticker_to_sector, config) -> tuple:
    name          = config["name"]
    risk_window   = config["risk_window"]
    signal_window = config["signal_window"]
    signal_type   = config["signal_type"]
    use_ewm       = config.get("ewm", False)
    is_anchored   = (name == "Equity 8")

    print(f"\n{'='*65}")
    anchor_label = " [FORANKRET EPO → 1/N]" if is_anchored else ""
    ewm_label    = " [EWMA kovarians]" if use_ewm else ""
    print(f"Kører {name} | Risikovindue: {risk_window}m | "
          f"Signal: {signal_type} {signal_window}m{anchor_label}{ewm_label}")
    print(f"{'='*65}")

    if signal_type == "XSMOM":
        signal = compute_xsmom(monthly_excess, signal_window)
    elif signal_type == "TSMOM":
        signal = compute_tsmom_signal(monthly_excess, ticker_to_sector,
                                      signal_window)
    else:
        raise ValueError(f"Ukendt signaltype: {signal_type}")

    if use_ewm:
        corr_shrunk, vols = compute_risk_model_ewm(
            monthly_excess, span=risk_window,
            min_periods=max(12, risk_window // 2),
            theta=CORR_PRESHRINK, verbose=True)
        corr_raw, vols_raw = compute_risk_model_ewm(
            monthly_excess, span=risk_window,
            min_periods=max(12, risk_window // 2),
            theta=0.0, verbose=False)
    else:
        corr_shrunk, vols = compute_risk_model(
            monthly_excess, window=risk_window,
            theta=CORR_PRESHRINK, verbose=True)
        corr_raw, vols_raw = compute_risk_model(
            monthly_excess, window=risk_window,
            theta=0.0, verbose=False)

    ew_full     = backtest_equal_weight(monthly_excess)
    indmom_full = backtest_indmom(monthly_excess, signal)
    mvo_full    = backtest_mvo_no_shrink(
        monthly_excess, signal, corr_raw, vols_raw, GAMMA)

    s, e = BACKTEST_START, BACKTEST_END

    if is_anchored:
        epo_panel = build_anchored_epo_panel(
            monthly_excess, signal, corr_shrunk, vols, GAMMA, CANDIDATE_WS)
        epo_dyn = build_dynamic_oos_epo(
            epo_panel, oos_start=BACKTEST_START,
            min_history=MIN_HISTORY_OOS)

        rows = [
            performance_summary(subset(ew_full,     s, e), "1/N"),
            performance_summary(subset(indmom_full, s, e), "INDMOM"),
            performance_summary(subset(mvo_full,    s, e), "MVO (no shrinkage)"),
            performance_summary(subset(epo_dyn,     s, e), "EPO: out-of-sample"),
        ]
        for w in CANDIDATE_WS:
            col = f"EPO_anchored_1N_w_{w:.2f}"
            if col in epo_panel.columns:
                rows.append(performance_summary(
                    subset(epo_panel[col], s, e), f"EPO w={w:.0%}"))
    else:
        epo_panel = build_epo_panel(
            monthly_excess, signal, corr_shrunk, vols, GAMMA, CANDIDATE_WS)
        epo_dyn = build_dynamic_oos_epo(
            epo_panel, oos_start=BACKTEST_START,
            min_history=MIN_HISTORY_OOS)

        rows = [
            performance_summary(subset(ew_full,     s, e), "1/N"),
            performance_summary(subset(indmom_full, s, e), "INDMOM"),
            performance_summary(subset(mvo_full,    s, e), "MVO (no shrinkage)"),
            performance_summary(subset(epo_dyn,     s, e), "EPO: out-of-sample"),
        ]
        for w in CANDIDATE_WS:
            col = f"EPO_w_{w:.2f}"
            if col in epo_panel.columns:
                rows.append(performance_summary(
                    subset(epo_panel[col], s, e), f"EPO w={w:.0%}"))

    perf = pd.DataFrame(rows).set_index("Strategy")
    perf.columns = pd.MultiIndex.from_tuples([(name, c) for c in perf.columns])
    return perf, corr_shrunk, vols


# ── IC-plot ───────────────────────────────────────────────────

def plot_rolling_ic(monthly_excess, ticker_to_sector, configs, window=12):
    """Rullende IC for Equity 3-6 med fast y-akse [-0.2, 0.3]."""
    signal_configs = [c for c in configs if c["name"] in
                      ["Equity 3", "Equity 4", "Equity 5", "Equity 6"]]

    fig, axes = plt.subplots(len(signal_configs), 1,
                             figsize=(12, 3 * len(signal_configs)),
                             sharex=True)

    excess_oos   = subset(monthly_excess, BACKTEST_START, BACKTEST_END)
    forward_rets = excess_oos.shift(-1)

    for ax, config in zip(axes, signal_configs):
        name          = config["name"]
        signal_type   = config["signal_type"]
        signal_window = config["signal_window"]

        if signal_type == "XSMOM":
            signal = compute_xsmom(monthly_excess, signal_window)
        elif signal_type == "TSMOM":
            signal = compute_tsmom_signal(monthly_excess, ticker_to_sector,
                                          signal_window)

        signal_oos = subset(signal, BACKTEST_START, BACKTEST_END)

        ic_scores = {}
        for date in signal_oos.index[:-1]:
            if date not in forward_rets.index:
                continue
            s_ = signal_oos.loc[date].dropna()
            r_ = forward_rets.loc[date].dropna()
            common = s_.index.intersection(r_.index)
            if len(common) < 5:
                continue
            ic_scores[date] = s_[common].corr(r_[common])

        ic_series  = pd.Series(ic_scores).sort_index()
        ic_series  = subset(ic_series, BACKTEST_START, BACKTEST_END)
        rolling_ic = ic_series.rolling(window=window).mean()

        ax.plot(rolling_ic.index, rolling_ic.values,
                label=f"Rullende IC ({window}m)", color="steelblue")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(ic_series.mean(), color="red", linewidth=0.8, linestyle=":",
                   label=f"Gns. IC: {ic_series.mean():.3f}")
        ax.fill_between(rolling_ic.index, rolling_ic, 0,
                        where=rolling_ic > 0, alpha=0.2, color="green")
        ax.fill_between(rolling_ic.index, rolling_ic, 0,
                        where=rolling_ic < 0, alpha=0.2, color="red")
        ax.set_ylim(-0.2, 0.3)
        ax.set_title(f"{name} | {signal_type} {signal_window}m")
        ax.set_xlim(pd.Timestamp(BACKTEST_START), pd.Timestamp(BACKTEST_END))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.xlabel("Dato")
    plt.tight_layout()
    plt.show()


def plot_risk_model_diagnostics(monthly_excess, configs, corr_results):
    """Rullende gennemsnitlig korrelation for Equity 1-3."""
    risk_configs = [c for c in configs if c["name"] in
                    ["Equity 1", "Equity 2", "Equity 3"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"Equity 1": "steelblue",
              "Equity 2": "darkorange",
              "Equity 3": "green"}

    for config in risk_configs:
        name        = config["name"]
        risk_window = config["risk_window"]
        color       = colors[name]

        corr_dict, _ = corr_results[name]

        avg_corr_series = {}
        for date, C in corr_dict.items():
            vals = C.values
            n    = len(vals)
            off  = vals[np.triu_indices(n, k=1)]
            avg_corr_series[date] = off.mean()

        avg_corr = pd.Series(avg_corr_series).sort_index()
        avg_corr = subset(avg_corr, BACKTEST_START, BACKTEST_END)

        ax.plot(avg_corr.index, avg_corr.values,
                label=f"{name} (vindue={risk_window}m)",
                color=color, linewidth=1.5)

    ax.set_title("Rullende gennemsnitlig korrelation (OOS 2020-2025) — aktieuniverset")
    ax.set_ylabel("Gns. korrelation")
    ax.set_xlabel("Dato")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(pd.Timestamp(BACKTEST_START), pd.Timestamp(BACKTEST_END))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("EQUITY GENSKABNING — Aktieuniverset (Yahoo Finance)")
    print(f"Data: {DATA_START} → {DATA_END}")
    print(f"OOS backtest: {BACKTEST_START} → {BACKTEST_END}")
    print("=" * 65)

    # Data
    daily, _, ticker_to_sector = load_data()
    monthly        = to_monthly_returns(daily)
    rf             = get_rf_monthly(DATA_START, DATA_END)
    monthly_excess = compute_monthly_excess(monthly, rf)

    print(f"\nAktier: {monthly_excess.shape[1]}  |  "
          f"{monthly_excess.index[0].date()} → {monthly_excess.index[-1].date()}")

    # Kør alle konfigurationer
    all_results  = []
    corr_results = {}
    for config in EQUITY_CONFIGS:
        perf, corr_shrunk, vols = run_single_equity(
            monthly_excess, ticker_to_sector, config)
        corr_results[config["name"]] = (corr_shrunk, vols)
        all_results.append(perf)

    # Samlet tabel
    combined = pd.concat(all_results, axis=1)
    print("\n" + "=" * 65)
    print(f"SAMLET SHARPE — OOS {BACKTEST_START[:4]}–{BACKTEST_END[:4]} — aktieuniverset")
    print("=" * 65)
    print(combined.to_string())

    # Figurer
    plot_rolling_ic(monthly_excess, ticker_to_sector, EQUITY_CONFIGS)
    plot_risk_model_diagnostics(monthly_excess, EQUITY_CONFIGS, corr_results)


if __name__ == "__main__":
    main()
