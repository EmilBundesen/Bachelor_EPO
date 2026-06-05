"""
PC_Volatility_Analysis.py

Replicates Figure 1 (A & B) from Pedersen's EPO article.

Two plots, each comparing MVO (no pre-shrinkage) vs. EPO w=0.75:

  Figure 1A — Volatility by PC rank
    X-axis : Principal component rank (1 = largest eigenvalue)
    Y-axis : Annualised volatility (%)
    Lines  : MVO ex ante | MVO realized | EPO ex ante | EPO realized

  Figure 1B — Expected return by PC rank
    X-axis : Principal component rank
    Y-axis : Annualised Sharpe ratio  (signal·vol⁻¹ vs. realised)
    Lines  : MVO ex ante | MVO realized | EPO ex ante | EPO realized

Risk models
  MVO : rolling 60-month covariance, θ=0 (no pre-shrinkage), w=0
  EPO : rolling 60-month covariance, θ=0.05 pre-shrinkage,   w=0.75
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from Equity_1 import (
    get_monthly_return,
    get_monthly_risk,
    calculate_monthly_excess_returns,
    compute_xsmom,
    compute_risk_model,
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    RISK_WINDOW,
    LOOKBACK_MONTHS,
    CORR_PRESHRINK,
)

OOS_START = BACKTEST_START_DATE
OOS_END   = BACKTEST_END_DATE
MAX_PCS   = 49

# Colours: MVO = blue family, EPO = red family
C_MVO_EA  = "#1f77b4"   # blue  — MVO ex ante
C_MVO_R   = "#aec7e8"   # light blue — MVO realized
C_EPO_EA  = "#d62728"   # red   — EPO ex ante
C_EPO_R   = "#f7b6b6"   # light red  — EPO realized


# ── Covariance helper ──────────────────────────────────────────────────────────

def build_sigma_w(corr: np.ndarray, vol: np.ndarray, w: float) -> np.ndarray:
    """Σ_w = (1-w)·Σ + w·diag(Σ)."""
    Sigma = corr * np.outer(vol, vol)
    return (1.0 - w) * Sigma + w * np.diag(np.diag(Sigma))


# ── Per-date PC decomposition ──────────────────────────────────────────────────

def compute_pc_series(
    w: float,
    corr_dict: dict,
    vols_dict: dict,
    signals: pd.DataFrame,
    monthly_excess: pd.DataFrame,
    oos_start: str,
    oos_end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every OOS date t with a risk model and signal, eigen-decompose Σ_w(t)
    and record per PC rank:
      - ex ante vol   : sqrt(λ_k) × sqrt(12)
      - ex ante Sharpe: (v_k^T s_t) / sqrt(λ_k) × sqrt(12)   [sign-aligned]
      - realized ret  : sign-aligned v_k^T r_{t+1}             [monthly]

    Returns (vol_mat, ea_sharpe_mat, realized_mat) each shape (T, MAX_PCS).
    """
    t0 = pd.Timestamp(oos_start)
    t1 = pd.Timestamp(oos_end)

    risk_dates  = sorted(d for d in corr_dict if t0 <= d <= t1)
    excess_idx  = monthly_excess.index
    excess_vals = monthly_excess.values

    vol_rows, ea_rows, r_rows = [], [], []

    for date in risk_dates:
        if date not in signals.index:
            continue
        try:
            t_loc = excess_idx.get_loc(date)
        except KeyError:
            continue
        if t_loc + 1 >= len(excess_idx):
            continue

        corr_df = corr_dict[date]
        vol_s   = vols_dict[date]
        sig_s   = signals.loc[date].dropna()

        common = (corr_df.index
                  .intersection(vol_s.index)
                  .intersection(sig_s.index))
        if len(common) < 2:
            continue

        C = corr_df.loc[common, common].values
        v = vol_s.loc[common].values
        s = sig_s.loc[common].values

        col_pos = [monthly_excess.columns.get_loc(c)
                   for c in common if c in monthly_excess.columns]
        if len(col_pos) != len(common):
            continue
        r_next = excess_vals[t_loc + 1, col_pos]
        if np.any(np.isnan(r_next)):
            continue

        Sigma_w = build_sigma_w(C, v, w)
        try:
            eigvals, eigvecs = np.linalg.eigh(Sigma_w)
        except np.linalg.LinAlgError:
            continue

        # Descending order
        order   = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        # Sign-align to signal direction
        proj  = eigvecs.T @ s
        signs = np.where(proj >= 0, 1.0, -1.0)
        eigvecs = eigvecs * signs

        n_pcs = len(eigvals)

        ex_ante_vol    = np.sqrt(eigvals) * np.sqrt(12)              # annualised
        ea_signal      = np.abs(eigvecs.T @ s)                       # ≥ 0 after align
        with np.errstate(divide="ignore", invalid="ignore"):
            ea_sharpe  = np.where(eigvals > 0,
                                  ea_signal / np.sqrt(eigvals), 0.0) * np.sqrt(12)
        r_pc = eigvecs.T @ r_next                                    # monthly

        def _pad(arr):
            row = np.full(MAX_PCS, np.nan)
            row[:min(n_pcs, MAX_PCS)] = arr[:MAX_PCS]
            return row

        vol_rows.append(_pad(ex_ante_vol))
        ea_rows.append(_pad(ea_sharpe))
        r_rows.append(_pad(r_pc))

    if not vol_rows:
        raise RuntimeError(f"No observations for w={w}.")

    return (np.array(vol_rows),
            np.array(ea_rows),
            np.array(r_rows))


def aggregate(vol_mat, ea_mat, r_mat):
    """
    Collapse time dimension → per PC rank statistics.

    Returns
    -------
    ea_vol    : mean ex ante annualised vol
    real_vol  : std of realized PC returns × sqrt(12)  (realized vol)
    ea_sharpe : mean ex ante signal-to-vol (annualised)
    real_sr   : mean(r_pc) / std(r_pc) × sqrt(12)     (realized Sharpe)
    n_obs     : valid observation count
    """
    with np.errstate(all="ignore"):
        ea_vol   = np.nanmean(vol_mat, axis=0)
        real_vol = np.nanstd(r_mat, axis=0, ddof=1) * np.sqrt(12)

        ea_sharpe = np.nanmean(ea_mat, axis=0)
        r_mean    = np.nanmean(r_mat, axis=0)
        r_std     = np.nanstd(r_mat, axis=0, ddof=1)
        real_sr   = np.where(r_std > 0, r_mean / r_std * np.sqrt(12), np.nan)

        n_obs = np.sum(~np.isnan(vol_mat), axis=0)

    return ea_vol, real_vol, ea_sharpe, real_sr, n_obs


# ── Plotting ───────────────────────────────────────────────────────────────────

def _mask(arr, n_obs, min_obs=12):
    mask = (n_obs >= min_obs) & np.isfinite(arr)
    ranks = np.where(mask)[0] + 1
    return ranks, arr[mask]


def plot_volatility(mvo_stats, epo_stats, oos_start, oos_end, save_path=None):
    """
    Figure 1A — Volatility by PC rank.
    Each dataset = (ea_vol, real_vol, ea_sharpe, real_sr, n_obs).
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ea_vol_m, real_vol_m, _, _, n_m = mvo_stats
    ea_vol_e, real_vol_e, _, _, n_e = epo_stats

    r_m, ea_m  = _mask(ea_vol_m * 100,   n_m)
    r_mr, rv_m = _mask(real_vol_m * 100, n_m)
    r_e, ea_e  = _mask(ea_vol_e * 100,   n_e)
    r_er, rv_e = _mask(real_vol_e * 100, n_e)

    ax.plot(r_m,  ea_m,  color=C_MVO_EA, lw=2.0, marker="o", ms=3,
            label="MVO — ex ante vol")
    ax.plot(r_mr, rv_m,  color=C_MVO_R,  lw=2.0, marker="o", ms=3,
            linestyle="--", label="MVO — realized vol")
    ax.plot(r_e,  ea_e,  color=C_EPO_EA, lw=2.0, marker="s", ms=3,
            label=r"EPO $w=0.75$ — ex ante vol")
    ax.plot(r_er, rv_e,  color=C_EPO_R,  lw=2.0, marker="s", ms=3,
            linestyle="--", label=r"EPO $w=0.75$ — realized vol")

    ax.set_xlabel("Principal component rank  (1 = largest eigenvalue)", fontsize=10)
    ax.set_ylabel("Annualised volatility (%)", fontsize=10)
    ax.set_title(
        f"Figure 1A — Ex Ante vs. Realized Volatility by PC\n"
        f"OOS: {oos_start[:7]} – {oos_end[:7]}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", lw=0.5, alpha=0.5)
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    return fig


def plot_expected_returns(mvo_stats, epo_stats, oos_start, oos_end, save_path=None):
    """
    Figure 1B — Expected return (Sharpe) by PC rank.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    _, _, ea_sr_m, real_sr_m, n_m = mvo_stats
    _, _, ea_sr_e, real_sr_e, n_e = epo_stats

    r_m,  ea_m  = _mask(ea_sr_m,   n_m)
    r_mr, rv_m  = _mask(real_sr_m, n_m)
    r_e,  ea_e  = _mask(ea_sr_e,   n_e)
    r_er, rv_e  = _mask(real_sr_e, n_e)

    ax.axhline(0, color="black", lw=0.8, alpha=0.4)

    ax.plot(r_m,  ea_m,  color=C_MVO_EA, lw=2.0, marker="o", ms=3,
            label="MVO — ex ante Sharpe")
    ax.plot(r_mr, rv_m,  color=C_MVO_R,  lw=2.0, marker="o", ms=3,
            linestyle="--", label="MVO — realized Sharpe")
    ax.plot(r_e,  ea_e,  color=C_EPO_EA, lw=2.0, marker="s", ms=3,
            label=r"EPO $w=0.75$ — ex ante Sharpe")
    ax.plot(r_er, rv_e,  color=C_EPO_R,  lw=2.0, marker="s", ms=3,
            linestyle="--", label=r"EPO $w=0.75$ — realized Sharpe")

    ax.set_xlabel("Principal component rank  (1 = largest eigenvalue)", fontsize=10)
    ax.set_ylabel("Annualised Sharpe ratio", fontsize=10)
    ax.set_title(
        f"Figure 1B — Ex Ante vs. Realized Expected Return by PC\n"
        f"OOS: {oos_start[:7]} – {oos_end[:7]}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", lw=0.5, alpha=0.5)
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("Figure 1A & 1B — Ex Ante vs. Realized by PC")
    print(f"MVO (θ=0, w=0)  vs.  EPO (θ={CORR_PRESHRINK}, w=0.75)")
    print(f"OOS: {OOS_START} → {OOS_END}")
    print("=" * 65 + "\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data …")
    monthly_ret    = get_monthly_return()
    rf             = get_monthly_risk()
    monthly_excess = calculate_monthly_excess_returns(monthly_ret, rf)
    signals        = compute_xsmom(monthly_excess, LOOKBACK_MONTHS)

    # ── Risk models ────────────────────────────────────────────────────────────
    print(f"\nBuilding MVO risk model (θ=0) …")
    corr_raw, vols_raw = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=0.0, verbose=False)

    print(f"Building EPO risk model (θ={CORR_PRESHRINK}) …")
    corr_shrunk, vols_shrunk = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=CORR_PRESHRINK, verbose=True)

    # ── Compute PC series ──────────────────────────────────────────────────────
    print("\nMVO (w=0, θ=0) …", flush=True)
    mvo_vols, mvo_ea, mvo_r = compute_pc_series(
        0.0, corr_raw, vols_raw, signals, monthly_excess, OOS_START, OOS_END)
    mvo_stats = aggregate(mvo_vols, mvo_ea, mvo_r)

    print("EPO (w=0.75, θ={}) …".format(CORR_PRESHRINK), flush=True)
    epo_vols, epo_ea, epo_r = compute_pc_series(
        0.75, corr_shrunk, vols_shrunk, signals, monthly_excess, OOS_START, OOS_END)
    epo_stats = aggregate(epo_vols, epo_ea, epo_r)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nPlotting Figure 1A (volatility) …")
    plot_volatility(mvo_stats, epo_stats, OOS_START, OOS_END,
                    save_path="Figure1A_Volatility.png")

    print("Plotting Figure 1B (expected return) …")
    plot_expected_returns(mvo_stats, epo_stats, OOS_START, OOS_END,
                          save_path="Figure1B_ExpectedReturn.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
