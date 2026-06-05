"""
PC_Volatility_Analysis.py

Produces 4 figures replicating the spirit of Figure 1 (A-D) in Pedersen's
EPO article, split by model and metric:

  Figure 1A — MVO        : ex ante vol vs. realized vol by PC rank
  Figure 1B — EPO w=0.75 : ex ante vol vs. realized vol by PC rank
  Figure 1C — MVO        : ex ante Sharpe vs. realized Sharpe by PC rank
  Figure 1D — EPO w=0.75 : ex ante Sharpe vs. realized Sharpe by PC rank

Risk models
  MVO : 60-month rolling covariance, θ=0 (no pre-shrinkage), w=0
  EPO : 60-month rolling covariance, θ=0.05 (pre-shrinkage),  w=0.75

Definitions (all annualised)
  Ex ante vol of PC k     = sqrt(λ_k) × sqrt(12)
    where λ_k is the k-th eigenvalue of Σ_w(t), averaged over OOS dates.

  Realized vol of PC k    = std_t( v_k(t)ᵀ r_{t+1} ) × sqrt(12)
    The time-series std of the returns projected on the k-th eigenvector.

  Ex ante Sharpe of PC k  = mean_t( |v_k(t)ᵀ s_t| / sqrt(λ_k(t)) ) × sqrt(12)
    Signal projected on PC k, normalised by the model's predicted vol.
    Since s_t is unit-leverage (same units as excess returns), this is a
    proper Sharpe ratio analogue.

  Realized Sharpe of PC k = mean_t(r̃_k) / std_t(r̃_k) × sqrt(12)
    where r̃_k(t) = sign(v_k(t)ᵀ s_t) × v_k(t)ᵀ r_{t+1}
    (sign-aligned so positive = signal predicts correctly).
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

OOS_START = BACKTEST_START_DATE   # "1942-01-01"
OOS_END   = BACKTEST_END_DATE     # "2018-12-31"
MAX_PCS   = 49                    # maximum number of principal components
MIN_OBS   = 60                    # minimum time-series obs to plot a PC rank

COLOUR_EXANTE   = "#1f77b4"   # blue
COLOUR_REALIZED = "#d62728"   # red


# ── Build Σ_w from stored correlation + vol ────────────────────────────────────

def build_sigma_w(corr: np.ndarray, vol: np.ndarray, w: float) -> np.ndarray:
    """
    Reconstruct Σ_w = (1-w)·Σ + w·diag(Σ)
    where Σ = diag(vol) · corr · diag(vol).
    corr and vol are already pre-shrunk if θ>0 was used in compute_risk_model.
    """
    Sigma   = corr * np.outer(vol, vol)
    Sigma_w = (1.0 - w) * Sigma + w * np.diag(np.diag(Sigma))
    return Sigma_w


# ── Per-date computation ───────────────────────────────────────────────────────

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
    Loop over every OOS date t that has a risk model entry.
    For each date, eigendecompose Σ_w(t) and record:

      vol_mat[t, k]    — ex ante annualised vol of PC rank k  = sqrt(λ_k) × sqrt(12)
      ea_sh_mat[t, k]  — ex ante Sharpe of PC rank k
                         = |v_k(t)ᵀ s_t| / sqrt(λ_k(t)) × sqrt(12)
      r_mat[t, k]      — sign-aligned realised return of PC rank k at t
                         = sign(v_k(t)ᵀ s_t) × v_k(t)ᵀ r_{t+1}   [monthly, decimal]

    All three matrices have shape (n_dates, MAX_PCS) with NaN padding.

    NaN handling: instead of dropping an entire date when some industry has a
    NaN next-period return, we restrict the common asset set to those with
    valid r_{t+1}.  This keeps far more observations.
    """
    t0 = pd.Timestamp(oos_start)
    t1 = pd.Timestamp(oos_end)

    risk_dates  = sorted(d for d in corr_dict if t0 <= d <= t1)
    excess_idx  = monthly_excess.index
    excess_arr  = monthly_excess.values        # shape (T, N), decimal returns

    vol_rows, ea_rows, r_rows = [], [], []

    for date in risk_dates:
        # ── locate date in the return series ──────────────────────────────────
        try:
            t_loc = excess_idx.get_loc(date)
        except KeyError:
            continue
        if t_loc + 1 >= len(excess_idx):
            continue

        # ── find assets that are valid at t (in risk model) AND have r_{t+1} ──
        corr_df = corr_dict[date]
        vol_s   = vols_dict[date]

        r_next_all  = excess_arr[t_loc + 1]                 # (N,) may have NaN
        valid_ret   = ~np.isnan(r_next_all)
        valid_names = monthly_excess.columns[valid_ret]

        # Intersect: in risk model, has signal (if signal date exists), has r_{t+1}
        common = corr_df.index.intersection(vol_s.index).intersection(valid_names)

        # Only use signal if it exists for this date (some early dates have none)
        has_signal = date in signals.index
        if has_signal:
            sig_s  = signals.loc[date].dropna()
            common = common.intersection(sig_s.index)
        else:
            sig_s = None

        if len(common) < 2:
            continue

        C = corr_df.loc[common, common].values              # (n, n) correlation
        v = vol_s.loc[common].values                        # (n,)   monthly std

        # Column positions for r_{t+1}
        col_pos = [monthly_excess.columns.get_loc(c) for c in common]
        r_next  = r_next_all[col_pos]                       # (n,) no NaN by construction

        # ── build Σ_w and eigendecompose ──────────────────────────────────────
        Sigma_w = build_sigma_w(C, v, w)
        try:
            eigvals, eigvecs = np.linalg.eigh(Sigma_w)     # ascending order
        except np.linalg.LinAlgError:
            continue

        # Sort descending (largest eigenvalue = PC 1)
        order   = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)           # clamp numerical noise
        eigvecs = eigvecs[:, order]                         # (n, n_pcs)

        n_pcs = len(eigvals)

        # ── ex ante vol ────────────────────────────────────────────────────────
        # sqrt(λ_k) is monthly std in decimal; ×sqrt(12) → annualised
        ea_vol = np.sqrt(eigvals) * np.sqrt(12)             # (n_pcs,)

        # ── sign-align + ex ante Sharpe ────────────────────────────────────────
        if has_signal:
            s   = sig_s.loc[common].values                  # (n,) signal weights
            # Project signal onto each PC; flip PC if projection is negative
            proj  = eigvecs.T @ s                           # (n_pcs,)
            signs = np.where(proj >= 0.0, 1.0, -1.0)
            eigvecs_aligned = eigvecs * signs               # columns flipped

            # |v_k^T s| (always ≥ 0 after sign alignment)
            ea_signal = eigvecs_aligned.T @ s               # (n_pcs,)

            # Sharpe = signal_projection / predicted_vol
            with np.errstate(divide="ignore", invalid="ignore"):
                ea_sharpe = np.where(
                    eigvals > 0.0,
                    ea_signal / np.sqrt(eigvals),
                    0.0,
                ) * np.sqrt(12)                             # annualised

            # Realised return: sign-aligned so positive = signal was right
            r_pc = eigvecs_aligned.T @ r_next              # (n_pcs,) monthly decimal
        else:
            # No signal available — cannot compute Sharpe; only record vol
            ea_sharpe = np.full(n_pcs, np.nan)
            r_pc      = eigvecs.T @ r_next                 # unsigned

        # ── pad and store ──────────────────────────────────────────────────────
        def _pad(arr):
            row = np.full(MAX_PCS, np.nan)
            fill = min(n_pcs, MAX_PCS)
            row[:fill] = arr[:fill]
            return row

        vol_rows.append(_pad(ea_vol))
        ea_rows.append(_pad(ea_sharpe))
        r_rows.append(_pad(r_pc))

    if not vol_rows:
        raise RuntimeError(f"No valid OOS dates found for w={w}.")

    return (np.array(vol_rows),      # (T, MAX_PCS)
            np.array(ea_rows),       # (T, MAX_PCS)
            np.array(r_rows))        # (T, MAX_PCS)


# ── Aggregate across time ──────────────────────────────────────────────────────

def aggregate(vol_mat, ea_mat, r_mat):
    """
    Collapse the time dimension into per-PC-rank statistics.

    Returns
    -------
    avg_ea_vol  : mean ex ante annualised vol per rank  (decimal, ×100 → %)
    real_vol    : std of realised PC returns ×sqrt(12)  (decimal, ×100 → %)
    avg_ea_sh   : mean ex ante Sharpe per rank
    real_sh     : realised Sharpe = mean(r̃_k)/std(r̃_k) ×sqrt(12)
    n_obs       : number of valid (non-NaN) observations per rank
    """
    with np.errstate(all="ignore"):
        avg_ea_vol = np.nanmean(vol_mat, axis=0)

        # Realized vol: std of the sign-aligned PC returns across time
        real_vol = np.nanstd(r_mat, axis=0, ddof=1) * np.sqrt(12)

        avg_ea_sh = np.nanmean(ea_mat, axis=0)

        r_mean = np.nanmean(r_mat, axis=0)
        r_std  = np.nanstd(r_mat,  axis=0, ddof=1)
        real_sh = np.where(r_std > 0,
                           r_mean / r_std * np.sqrt(12),
                           np.nan)

        n_obs = np.sum(~np.isnan(vol_mat), axis=0)

    return avg_ea_vol, real_vol, avg_ea_sh, real_sh, n_obs


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _valid_range(arr, n_obs, min_obs=MIN_OBS):
    """Return (ranks, values) for PC ranks with enough observations."""
    mask  = (n_obs >= min_obs) & np.isfinite(arr)
    ranks = np.where(mask)[0] + 1   # 1-based
    return ranks, arr[mask]


def _plot_two_lines(ax, ranks_ea, vals_ea, ranks_r, vals_r,
                    label_ea, label_r, ylabel, title):
    ax.plot(ranks_ea, vals_ea,
            color=COLOUR_EXANTE, lw=2.0, marker="o", ms=3, label=label_ea)
    ax.plot(ranks_r,  vals_r,
            color=COLOUR_REALIZED, lw=2.0, marker="s", ms=3,
            linestyle="--", label=label_r)
    ax.axhline(0, color="black", lw=0.7, alpha=0.35)
    ax.set_xlabel("Principal component rank  (1 = largest eigenvalue)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", lw=0.5, alpha=0.5)
    ax.tick_params(labelsize=9)


# ── Four individual figures ────────────────────────────────────────────────────

def make_figure(stats, fig_label, model_name, metric, oos_start, oos_end,
                save_path=None):
    """
    stats     : (avg_ea_vol, real_vol, avg_ea_sh, real_sh, n_obs)
    metric    : "vol" or "sharpe"
    """
    avg_ea_vol, real_vol, avg_ea_sh, real_sh, n_obs = stats

    fig, ax = plt.subplots(figsize=(9, 5))

    if metric == "vol":
        r_ea, v_ea = _valid_range(avg_ea_vol * 100, n_obs)   # decimal → %
        r_r,  v_r  = _valid_range(real_vol   * 100, n_obs)
        _plot_two_lines(
            ax,
            r_ea, v_ea, r_r, v_r,
            label_ea=f"{model_name} — ex ante vol",
            label_r =f"{model_name} — realized vol",
            ylabel="Annualised volatility (%)",
            title=(f"Figure 1{fig_label} — {model_name}: "
                   f"Ex Ante vs. Realized Volatility by PC\n"
                   f"OOS: {oos_start[:7]} – {oos_end[:7]}"),
        )
    else:  # sharpe
        r_ea, v_ea = _valid_range(avg_ea_sh, n_obs)
        r_r,  v_r  = _valid_range(real_sh,   n_obs)
        _plot_two_lines(
            ax,
            r_ea, v_ea, r_r, v_r,
            label_ea=f"{model_name} — ex ante Sharpe",
            label_r =f"{model_name} — realized Sharpe",
            ylabel="Annualised Sharpe ratio",
            title=(f"Figure 1{fig_label} — {model_name}: "
                   f"Ex Ante vs. Realized Expected Return by PC\n"
                   f"OOS: {oos_start[:7]} – {oos_end[:7]}"),
        )

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
    print("Figure 1 (A–D)  —  Ex Ante vs. Realized by Principal Component")
    print(f"  MVO : θ=0   (no pre-shrinkage), w=0")
    print(f"  EPO : θ={CORR_PRESHRINK} (pre-shrinkage),  w=0.75")
    print(f"  OOS : {OOS_START} → {OOS_END}")
    print("=" * 65 + "\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading monthly returns and risk-free rate …")
    monthly_ret    = get_monthly_return()
    rf             = get_monthly_risk()
    monthly_excess = calculate_monthly_excess_returns(monthly_ret, rf)

    print("Computing XSMOM signal …")
    signals = compute_xsmom(monthly_excess, LOOKBACK_MONTHS)

    # ── Build two risk models ──────────────────────────────────────────────────
    print(f"\nBuilding MVO risk model  (θ=0, no pre-shrinkage) …")
    corr_raw, vols_raw = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=0.0, verbose=False)
    print(f"  Dates in risk model: {len(corr_raw)}")

    print(f"\nBuilding EPO risk model  (θ={CORR_PRESHRINK}) …")
    corr_shrunk, vols_shrunk = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=CORR_PRESHRINK, verbose=True)
    print(f"  Dates in risk model: {len(corr_shrunk)}")

    # ── Compute PC time series ─────────────────────────────────────────────────
    print("\nComputing PC decomposition for MVO (w=0, θ=0) …", flush=True)
    mvo_vol, mvo_ea, mvo_r = compute_pc_series(
        0.0, corr_raw, vols_raw, signals, monthly_excess, OOS_START, OOS_END)
    mvo_stats = aggregate(mvo_vol, mvo_ea, mvo_r)
    n_dates_mvo = mvo_vol.shape[0]
    print(f"  OOS dates used: {n_dates_mvo}")

    print(f"\nComputing PC decomposition for EPO (w=0.75, θ={CORR_PRESHRINK}) …",
          flush=True)
    epo_vol, epo_ea, epo_r = compute_pc_series(
        0.75, corr_shrunk, vols_shrunk, signals, monthly_excess, OOS_START, OOS_END)
    epo_stats = aggregate(epo_vol, epo_ea, epo_r)
    n_dates_epo = epo_vol.shape[0]
    print(f"  OOS dates used: {n_dates_epo}")

    # ── Quick sanity check ─────────────────────────────────────────────────────
    for name, stats in [("MVO", mvo_stats), ("EPO", epo_stats)]:
        ea_v, r_v, ea_s, r_s, n = stats
        ok = n >= MIN_OBS
        print(f"\n  {name}: {ok.sum()} PC ranks with ≥{MIN_OBS} obs")
        if ok.sum() > 0:
            print(f"    PC1 ex ante vol   : {ea_v[ok][0]*100:.2f}%")
            print(f"    PC1 realized vol  : {r_v[ok][0]*100:.2f}%")
            print(f"    PC1 ex ante Sharpe: {ea_s[ok][0]:.3f}")
            print(f"    PC1 realized Sharpe:{r_s[ok][0]:.3f}")

    # ── Four figures ───────────────────────────────────────────────────────────
    print("\nPlotting …")

    make_figure(mvo_stats, "A", "MVO", "vol",
                OOS_START, OOS_END, "Figure1A_MVO_Volatility.png")

    make_figure(epo_stats, "B", r"EPO $w$=0.75", "vol",
                OOS_START, OOS_END, "Figure1B_EPO_Volatility.png")

    make_figure(mvo_stats, "C", "MVO", "sharpe",
                OOS_START, OOS_END, "Figure1C_MVO_ExpectedReturn.png")

    make_figure(epo_stats, "D", r"EPO $w$=0.75", "sharpe",
                OOS_START, OOS_END, "Figure1D_EPO_ExpectedReturn.png")

    print("\nDone.  Produced:")
    print("  Figure1A_MVO_Volatility.png")
    print("  Figure1B_EPO_Volatility.png")
    print("  Figure1C_MVO_ExpectedReturn.png")
    print("  Figure1D_EPO_ExpectedReturn.png")


if __name__ == "__main__":
    main()
