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
    epo_weights,
    backtest_strategy
)

from Stock_Data import compute_tsmom_signal

def performance_summary(r: pd.Series, name: str) -> dict:
    r    = r.dropna()
    return {
        "Strategy":    name,
        "Sharpe":      round(sharpe_ratio(r), 3),
    }

# ── Konstanter ────────────────────────────────────────────────
EQUITY_CONFIGS = [
    {"name": "Equity 1", "risk_window": 60, "signal_window": 12, "signal_type": "XSMOM"},
    {"name": "Equity 2", "risk_window": 36, "signal_window": 12, "signal_type": "XSMOM"},
    {"name": "Equity 3", "risk_window": 24, "signal_window": 12, "signal_type": "XSMOM"},
    {"name": "Equity 4", "risk_window": 24, "signal_window": 24, "signal_type": "XSMOM"},
    {"name": "Equity 5", "risk_window": 24, "signal_window":  6, "signal_type": "XSMOM"},
    {"name": "Equity 6", "risk_window": 24, "signal_window":  3, "signal_type": "XSMOM"},
    {"name": "Equity 7", "risk_window": 24, "signal_window": 12, "signal_type": "TSMOM"},
    {"name": "Equity 8", "risk_window": 24, "signal_window": 12, "signal_type": "XSMOM"},  # forankret EPO
]


# ── Forankret EPO (Eq. i opgaven) ─────────────────────────────
def epo_anchored_weights(signal: pd.Series,
                          anchor: pd.Series,
                          corr: pd.DataFrame,
                          vols: pd.Series,
                          gamma: float,
                          w: float,
                          min_vol: float = MIN_VOL) -> pd.Series:
    """
    EPO_a(w) = Σ_w^{-1} [ (1-w) · κ · s  +  w · V · a ]

    hvor κ = sqrt(a' Σ̃ a) / sqrt(s' Σ_w^{-1} Σ̃ Σ_w^{-1} s)
    skalerer signalet til samme enhed som ankeret.

    anchor : 1/N vægte (lige vægt i alle tilgængelige aktier)
    V      : diag(Σ) — diagonal kovariansmatrix
    """
    common = (signal.dropna().index
              .intersection(corr.index)
              .intersection(vols.index)
              .intersection(anchor.index))
    if len(common) < 2:
        return pd.Series(dtype=float)

    s_vec = signal.loc[common]
    v     = vols.loc[common]
    C     = corr.loc[common, common]
    a_vec = anchor.loc[common]

    ok    = v >= min_vol
    s_vec, v, C, a_vec = s_vec[ok], v[ok], C.loc[ok, ok], a_vec[ok]
    if len(s_vec) < 2:
        return pd.Series(dtype=float)

    v_a   = v.values
    s_a   = s_vec.values
    C_a   = C.values
    a_a   = a_vec.values
    n     = len(s_a)

    # Σ_w = D · [(1-w)C + w·I] · D
    C_w     = (1.0 - w) * C_a + w * np.eye(n)
    D       = np.diag(v_a)
    Sigma_w = D @ C_w @ D

    # Σ̃ = diag(Σ_w)  (V i formlen)
    Sigma_tilde = np.diag(np.diag(Sigma_w))

    try:
        Sigma_w_inv = np.linalg.inv(Sigma_w)
    except np.linalg.LinAlgError:
        Sigma_w_inv = np.linalg.pinv(Sigma_w)

    # Skalering κ
    num   = np.sqrt(a_a @ Sigma_tilde @ a_a)
    inner = s_a @ Sigma_w_inv @ Sigma_tilde @ Sigma_w_inv @ s_a
    denom = np.sqrt(inner) if inner > 0 else 1.0
    kappa = num / denom if denom > 0 else 1.0

    # V · a  (diagonal matrix gange vektor = element-vis)
    Va = np.diag(Sigma_tilde) * a_a

    # Kombineret signal
    combined = (1.0 - w) * kappa * s_a + w * Va

    # EPO_a = Σ_w^{-1} · combined
    x_raw = Sigma_w_inv @ combined
    x_s   = pd.Series(x_raw, index=s_vec.index)

    # Unit-leverage normalisering
    pos = x_s[x_s > 0].sum()
    neg = x_s[x_s < 0].sum()

    if pos == 0 and neg == 0:
        return pd.Series(dtype=float)

    if neg == 0:
        return x_s / pos
    elif pos == 0:
        return x_s / abs(neg)
    else:
        return x_s / max(pos, abs(neg))

def backtest_epo_anchored(monthly_excess, signals, corr_dict, vols_dict,
                           gamma, w) -> pd.Series:
    """Backtest af forankret EPO med 1/N som anker."""
    risk_dates = set(corr_dict)
    sig_dates  = set(signals.index)

    def weight_fn(date):
        if date not in risk_dates or date not in sig_dates:
            return pd.Series(dtype=float)

        # 1/N anker: lige vægt i alle aktier med signal og risikomodel
        sig  = signals.loc[date].dropna()
        v    = vols_dict[date].reindex(sig.index).dropna()
        common = sig.index.intersection(v.index)
        if len(common) < 2:
            return pd.Series(dtype=float)

        n_assets = len(common)
        anchor   = pd.Series(1.0 / n_assets, index=common)

        return epo_anchored_weights(
            signal=sig,
            anchor=anchor,
            corr=corr_dict[date],
            vols=vols_dict[date],
            gamma=gamma,
            w=w,
        )

    return backtest_strategy(monthly_excess, weight_fn,
                             name=f"EPO_anchored_1N_w_{w:.2f}")


def build_anchored_epo_panel(monthly_excess, signals, corr_dict, vols_dict,
                              gamma, candidate_ws) -> pd.DataFrame:
    parts = []
    for w in candidate_ws:
        print(f"  Anchored EPO w = {w:.2f} …", flush=True)
        parts.append(backtest_epo_anchored(
            monthly_excess, signals, corr_dict, vols_dict, gamma, w))
    return pd.concat(parts, axis=1).sort_index()


# ── run_single_equity (opdateret til Equity 8) ───────────────

def run_single_equity(monthly_excess, config) -> pd.DataFrame:
    name          = config["name"]
    risk_window   = config["risk_window"]
    signal_window = config["signal_window"]
    signal_type   = config["signal_type"]
    is_anchored   = (name == "Equity 8")

    print(f"\n{'='*65}")
    anchor_label = " [FORANKRET EPO → 1/N]" if is_anchored else ""
    print(f"Kører {name} | Risikovindue: {risk_window}m | "
          f"Signal: {signal_type} {signal_window}m{anchor_label}")
    print(f"{'='*65}")

    if signal_type == "XSMOM":
        signal = compute_xsmom(monthly_excess, signal_window)
    elif signal_type == "TSMOM":
        signal = compute_tsmom_signal(monthly_excess, signal_window)
    else:
        raise ValueError(f"Ukendt signaltype: {signal_type}")

    corr_shrunk, vols = compute_risk_model(
        monthly_excess, window=risk_window, theta=CORR_PRESHRINK, verbose=True)
    corr_raw, vols_raw = compute_risk_model(
        monthly_excess, window=risk_window, theta=0.0, verbose=False)

    ew_full     = backtest_equal_weight(monthly_excess)
    indmom_full = backtest_indmom(monthly_excess, signal)
    mvo_full    = backtest_mvo_no_shrink(
        monthly_excess, signal, corr_raw, vols_raw, GAMMA)

    s, e = BACKTEST_START_DATE, BACKTEST_END_DATE

    if is_anchored:
        epo_panel = build_anchored_epo_panel(
            monthly_excess, signal, corr_shrunk, vols, GAMMA, CANDIDATE_WS)
        epo_dyn = build_dynamic_oos_epo(
            epo_panel, oos_start=BACKTEST_START_DATE,
            min_history=MIN_HISTORY_OOS)

        rows = [
            performance_summary(subset(ew_full,     s, e), "1/N"),
            performance_summary(subset(indmom_full, s, e), "INDMOM"),
            performance_summary(subset(mvo_full,    s, e), "MVO (no shrinkage)"),
            # ← samme navn som Equity 1-7
            performance_summary(subset(epo_dyn,     s, e), "EPO: out-of-sample"),
        ]
        for w in CANDIDATE_WS:
            col = f"EPO_anchored_1N_w_{w:.2f}"
            if col in epo_panel.columns:
                # ← samme navnemønster som Equity 1-7
                rows.append(performance_summary(
                    subset(epo_panel[col], s, e), f"EPO w={w:.0%}"))
    else:
        epo_panel = build_epo_panel(
            monthly_excess, signal, corr_shrunk, vols, GAMMA, CANDIDATE_WS)
        epo_dyn = build_dynamic_oos_epo(
            epo_panel, oos_start=BACKTEST_START_DATE,
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


# ── IC plot (fast y-akse) ─────────────────────────────────────

def plot_rolling_ic(monthly_excess, configs, window=12):
    """Viser rullende IC for Equity 3-6 med fast y-akse [-0.2, 0.3]."""

    signal_configs = [c for c in configs if c["name"] in
                      ["Equity 3", "Equity 4", "Equity 5", "Equity 6"]]

    fig, axes = plt.subplots(len(signal_configs), 1,
                             figsize=(12, 3 * len(signal_configs)),
                             sharex=True)

    excess_oos   = subset(monthly_excess, BACKTEST_START_DATE, BACKTEST_END_DATE)
    forward_rets = excess_oos.shift(-1)

    for ax, config in zip(axes, signal_configs):
        name          = config["name"]
        signal_type   = config["signal_type"]
        signal_window = config["signal_window"]

        if signal_type == "XSMOM":
            signal = compute_xsmom(monthly_excess, signal_window)
        elif signal_type == "TSMOM":
            signal = compute_tsmom_signal(monthly_excess, signal_window)

        signal_oos = subset(signal, BACKTEST_START_DATE, BACKTEST_END_DATE)

        ic_scores = {}
        for date in signal_oos.index[:-1]:
            if date not in forward_rets.index:
                continue
            s_     = signal_oos.loc[date].dropna()
            r_     = forward_rets.loc[date].dropna()
            common = s_.index.intersection(r_.index)
            if len(common) < 5:
                continue
            ic_scores[date] = s_[common].corr(r_[common])

        ic_series  = pd.Series(ic_scores).sort_index()
        ic_series  = subset(ic_series, BACKTEST_START_DATE, BACKTEST_END_DATE)
        rolling_ic = ic_series.rolling(window=window).mean()

        ax.plot(rolling_ic.index, rolling_ic.values,
                label=f"Rullende IC ({window}m)", color="steelblue")
        ax.axhline(0,              color="black", linewidth=0.8, linestyle="--")
        ax.axhline(ic_series.mean(), color="red", linewidth=0.8, linestyle=":",
                   label=f"Gns. IC: {ic_series.mean():.3f}")
        ax.fill_between(rolling_ic.index, rolling_ic, 0,
                        where=rolling_ic > 0, alpha=0.2, color="green")
        ax.fill_between(rolling_ic.index, rolling_ic, 0,
                        where=rolling_ic < 0, alpha=0.2, color="red")

        # Fast y-akse
        ax.set_ylim(-0.2, 0.3)

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
    Viser rullende gennemsnitlig korrelation for Equity 1-3 (risikovindue varierer).
    """
    risk_configs = [c for c in configs if c["name"] in
                    ["Equity 1", "Equity 2", "Equity 3"]]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

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
        avg_corr = subset(avg_corr, BACKTEST_START_DATE, BACKTEST_END_DATE)

        ax.plot(avg_corr.index, avg_corr.values,
                label=f"{name} (vindue={risk_window}m)",
                color=color, linewidth=1.5)

    start = pd.Timestamp(BACKTEST_START_DATE)
    end   = pd.Timestamp(BACKTEST_END_DATE)

    ax.set_title("Rullende gennemsnitlig korrelation (OOS periode 2010-2025)")
    ax.set_ylabel("Gns. korrelation")
    ax.set_xlabel("Dato")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(start, end)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

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