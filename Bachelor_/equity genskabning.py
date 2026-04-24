import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Konstanter
DATA_START_DATE     = "1985-01-01"
BACKTEST_START_DATE = "2010-01-01"
BACKTEST_END_DATE   = "2025-12-31"

CORR_PRESHRINK   = 0.05  # θ: 5%
MIN_VOL          = 1e-8  # Floor on vol to avoid division by zero

CANDIDATE_WS     = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]
GAMMA = 3
MIN_HISTORY_OOS  = 1
PERCENT_TO_DECIMAL   = 100.0
Missing_values = [-99.99, -999]

from Equity_1 import (
    get_monthly_return,
    get_monthly_risk,
    calculate_monthly_excess_returns,
    compute_xsmom, # dette er ligning 24-25 fra artiklen.
    compute_risk_model,
    backtest_equal_weight,
    backtest_indmom,
    backtest_mvo_no_shrink,
    build_epo_panel,
    build_dynamic_oos_epo,
    sharpe_ratio,
    subset,
    epo_weights
)

from Stock_Data import compute_tsmom_signal

def performance_summary(r: pd.Series, name: str) -> dict:
    r    = r.dropna()
    return {
        "Strategy":    name,
        "Sharpe":      round(sharpe_ratio(r), 3),
    }

# Konstanter
EQUITY_CONFIGS = [
    {"name": "Equity 1", "risk_window": 60, "signal_window": 12, "signal_type": "XSMOM"},
    {"name": "Equity 2", "risk_window": 36, "signal_window": 12, "signal_type": "XSMOM"},
    {"name": "Equity 3", "risk_window": 24, "signal_window": 12, "signal_type": "XSMOM"},
    {"name": "Equity 4", "risk_window": 24, "signal_window": 24, "signal_type": "XSMOM"},
    {"name": "Equity 5", "risk_window": 24, "signal_window": 12, "signal_type": "XSMOM"},
    {"name": "Equity 6", "risk_window": 24, "signal_window":  3, "signal_type": "XSMOM"},
    {"name": "Equity 7", "risk_window": 24, "signal_window": 12, "signal_type": "TSMOM"},
]

def run_single_equity(monthly_excess, config) -> pd.DataFrame:
    name          = config["name"]
    risk_window   = config["risk_window"]
    signal_window = config["signal_window"]
    signal_type   = config["signal_type"]

    print(f"\n{'='*65}")
    print(f"Kører {name} | Risikovindue: {risk_window}m | Signal: {signal_type} {signal_window}m")
    print(f"{'='*65}")

    # Signal
    if signal_type == "XSMOM":
        signal = compute_xsmom(monthly_excess, signal_window)
    elif signal_type == "TSMOM":
        signal = compute_tsmom_signal(monthly_excess, signal_window)  # skal importeres
    else:
        raise ValueError(f"Ukendt signaltype: {signal_type}")

    # Risikomodel
    corr_shrunk, vols = compute_risk_model(
        monthly_excess, window=risk_window, theta=CORR_PRESHRINK, verbose=True)
    corr_raw, vols_raw = compute_risk_model(
        monthly_excess, window=risk_window, theta=0.0, verbose=False)

    # Backtests
    ew_full     = backtest_equal_weight(monthly_excess)
    indmom_full = backtest_indmom(monthly_excess, signal)
    mvo_full    = backtest_mvo_no_shrink(monthly_excess, signal, corr_raw, vols_raw, GAMMA)
    epo_panel   = build_epo_panel(monthly_excess, signal, corr_shrunk, vols, GAMMA, CANDIDATE_WS)
    epo_dyn     = build_dynamic_oos_epo(epo_panel, oos_start=BACKTEST_START_DATE, min_history=MIN_HISTORY_OOS)

    # OOS
    s, e = BACKTEST_START_DATE, BACKTEST_END_DATE
    rows = [
        performance_summary(subset(ew_full,     s, e), "1/N"),
        performance_summary(subset(indmom_full, s, e), "INDMOM"),
        performance_summary(subset(mvo_full,    s, e), "MVO (no shrinkage)"),
        performance_summary(subset(epo_dyn,     s, e), "EPO: out-of-sample"),
    ]
    for w in CANDIDATE_WS:
        col = f"EPO_w_{w:.2f}"
        if col in epo_panel.columns:
            rows.append(performance_summary(subset(epo_panel[col], s, e), f"EPO w={w:.0%}"))

    perf = pd.DataFrame(rows).set_index("Strategy")
    perf.columns = pd.MultiIndex.from_tuples([(name, c) for c in perf.columns])
    return perf, corr_shrunk, vols


def plot_rolling_ic(monthly_excess, configs, window=12):
    """Viser rullende IC for Equity 4-7 (signal varierer)."""

    signal_configs = [c for c in configs if c["name"] in
                      ["Equity 4", "Equity 5", "Equity 6", "Equity 7"]]

    fig, axes = plt.subplots(len(signal_configs), 1,
                             figsize=(12, 3 * len(signal_configs)),
                             sharex=True)

    # Begræns excess returns til OOS-perioden
    excess_oos   = subset(monthly_excess, BACKTEST_START_DATE, BACKTEST_END_DATE)
    forward_rets = excess_oos.shift(-1)

    for ax, config in zip(axes, signal_configs):
        name          = config["name"]
        signal_type   = config["signal_type"]
        signal_window = config["signal_window"]

        # Signal beregnes på hele historikken
        if signal_type == "XSMOM":
            signal = compute_xsmom(monthly_excess, signal_window)
        elif signal_type == "TSMOM":
            signal = compute_tsmom_signal(monthly_excess, signal_window)

        # Begræns signal til OOS-perioden
        signal_oos = subset(signal, BACKTEST_START_DATE, BACKTEST_END_DATE)

        # Beregn IC for hver dato i OOS-perioden
        ic_scores = {}
        for date in signal_oos.index[:-1]:
            if date not in forward_rets.index:
                continue
            s      = signal_oos.loc[date].dropna()
            r      = forward_rets.loc[date].dropna()
            common = s.index.intersection(r.index)
            if len(common) < 5:
                continue
            ic_scores[date] = s[common].corr(r[common])

        ic_series  = pd.Series(ic_scores).sort_index()
        ic_series  = subset(ic_series, BACKTEST_START_DATE, BACKTEST_END_DATE)
        rolling_ic = ic_series.rolling(window=window).mean()

        ax.plot(rolling_ic.index, rolling_ic.values,
                label=f"Rullende IC ({window}m)", color="steelblue")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(ic_series.mean(), color="red",
                   linewidth=0.8, linestyle=":",
                   label=f"Gns. IC: {ic_series.mean():.3f}")
        ax.fill_between(rolling_ic.index, rolling_ic, 0,
                        where=rolling_ic > 0, alpha=0.2, color="green")
        ax.fill_between(rolling_ic.index, rolling_ic, 0,
                        where=rolling_ic < 0, alpha=0.2, color="red")
        ax.set_title(f"{name} | {signal_type} {signal_window}m")
        ax.set_xlim(pd.Timestamp(BACKTEST_START_DATE),
                    pd.Timestamp(BACKTEST_END_DATE))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.xlabel("Dato")
    plt.tight_layout()
    plt.show()


def plot_risk_model_diagnostics(monthly_excess, configs, corr_results):
    """
    Viser rullende gennemsnitlig korrelation og porteføljevægtenes
    stabilitet for Equity 1-3 (risikovindue varierer).
    """
    risk_configs = [c for c in configs if c["name"] in
                    ["Equity 1", "Equity 2", "Equity 3"]]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    colors = {"Equity 1": "steelblue",
              "Equity 2": "darkorange",
              "Equity 3": "green"}

    for config in risk_configs:
        name          = config["name"]
        risk_window   = config["risk_window"]
        signal_window = config["signal_window"]
        color         = colors[name]

        corr_dict, vols_dict = corr_results[name]

        # ── Plot 1: Rullende gennemsnitlig korrelation ────────────────
        avg_corr_series = {}
        for date, C in corr_dict.items():
            vals = C.values
            n    = len(vals)
            off  = vals[np.triu_indices(n, k=1)]
            avg_corr_series[date] = off.mean()

        avg_corr = (pd.Series(avg_corr_series)
                    .sort_index())
        avg_corr = subset(avg_corr, BACKTEST_START_DATE, BACKTEST_END_DATE)

        axes[0].plot(avg_corr.index, avg_corr.values,
                     label=f"{name} (vindue={risk_window}m)",
                     color=color, linewidth=1.5)

        # ── Plot 2: Porteføljevægtenes stabilitet ─────────────────────
        signal     = compute_xsmom(monthly_excess, signal_window)
        signal_oos = subset(signal, BACKTEST_START_DATE, BACKTEST_END_DATE)
        weights    = {}

        for date in signal_oos.index:
            if date not in corr_dict:
                continue
            w = epo_weights(signal.loc[date], corr_dict[date],
                            vols_dict[date], GAMMA, w=0.0)
            if len(w) > 0:
                weights[date] = w

        if weights:
            weights_df  = pd.DataFrame(weights).T.sort_index()
            weights_df  = subset(weights_df, BACKTEST_START_DATE, BACKTEST_END_DATE)
            rolling_std = weights_df.std(axis=1).rolling(window=12).mean()

            axes[1].plot(rolling_std.index, rolling_std.values,
                         label=f"{name} (vindue={risk_window}m)",
                         color=color, linewidth=1.5)

    # ── Formatering ───────────────────────────────────────────────────
    start = pd.Timestamp(BACKTEST_START_DATE)
    end   = pd.Timestamp(BACKTEST_END_DATE)

    axes[0].set_title("Rullende gennemsnitlig korrelation (OOS periode 2010-2025)")
    axes[0].set_ylabel("Gns. korrelation")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlim(start, end)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Porteføljevægtenes stabilitet — rullende std. (12m, OOS periode 2010-2025)")
    axes[1].set_ylabel("Gns. std. af vægte")
    axes[1].set_xlim(start, end)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.xlabel("Dato")
    plt.tight_layout()
    plt.show()


def main():
    # Data indlæsning
    monthly_ret   = get_monthly_return()
    rf            = get_monthly_risk()
    monthly_excess = calculate_monthly_excess_returns(monthly_ret, rf)

    # Kør alle equity konfigurationer
    all_results = []
    corr_results = {}
    for config in EQUITY_CONFIGS:
        perf, corr_shrunk, vols = run_single_equity(monthly_excess, config)
        corr_results[config["name"]] = (corr_shrunk, vols)
        all_results.append(perf)

    # Saml resultater
    combined = pd.concat(all_results, axis=1)
    print("\n" + "="*65)
    print("SAMLET PERFORMANCE — alle equity konfigurationer")
    print("="*65)
    print(combined.to_string())

    #Signal diagnostik
    plot_rolling_ic(monthly_excess, EQUITY_CONFIGS)
    plot_risk_model_diagnostics(monthly_excess, EQUITY_CONFIGS,corr_results)

if __name__ == "__main__":
    main()