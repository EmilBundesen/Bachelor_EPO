"""
PC_Volatility_Analysis.py

Replicates Figure 1 (A, B, C, D) from Pedersen's EPO article:
  "Ex ante vs. realized expected return by principal component."

Panel layout (matching the MVO vs. EPO theme):
  A — MVO        : w=0, no pre-shrinkage (θ=0)   ← raw risk model
  B — EPO w=0    : w=0, with pre-shrinkage (θ=0.05)
  C — EPO w=0.75 : w=0.75, with pre-shrinkage (θ=0.05)
  D — EPO w=1    : w=1 (diagonal), with pre-shrinkage (θ=0.05)

For each panel and each PC rank k at time t:
  1. Eigendecompose Σ_w(t) → eigenvectors V, eigenvalues Λ (descending)
  2. Sign-align v_k so that v_k^T s_t ≥ 0  (signal direction)
  3. Ex ante (expected) return of PC k:
       α_k(t) = v_k^T s_t / sqrt(λ_k)  ×  sqrt(12)   [annualised Sharpe contribution]
  4. Realized return of PC k:
       r_k(t) = v_k^T r_{t+1}  × sqrt(12)  averaged over t  / std(r_k) × sqrt(12)
     (i.e., realized Sharpe ratio of the sign-aligned PC portfolio)

  The line chart shows, for each PC rank:
    • Blue line  — average ex ante signal-to-vol (α_k / sqrt(λ_k), annualised)
    • Red line   — realized Sharpe ratio of the PC portfolio (annualised)

  Under ideal conditions the two lines coincide.  MVO amplifies α for the
  top PCs (noise), while EPO dampens this, producing better calibration.
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

# ── Panel configuration ────────────────────────────────────────────────────────
# Each panel: (w, use_preshrink, label, title)
PANELS = [
    (0.00, False, "A", r"MVO  ($w=0$, no pre-shrinkage)"),
    (0.00, True,  "B", r"EPO  $w=0$  (with pre-shrinkage)"),
    (0.75, True,  "C", r"EPO  $w=0.75$  (with pre-shrinkage)"),
    (1.00, True,  "D", r"EPO  $w=1$  (diagonal, with pre-shrinkage)"),
]

OOS_START = BACKTEST_START_DATE
OOS_END   = BACKTEST_END_DATE
MAX_PCS   = 49

COLOUR_EXANTE   = "#1f77b4"   # blue — ex ante / expected
COLOUR_REALIZED = "#d62728"   # red  — realized


# ── Core computation ──────────────────────────────────────────────────────────

def build_sigma_w(corr: np.ndarray, vol: np.ndarray, w: float) -> np.ndarray:
    """Σ_w = (1-w)·Σ + w·diag(Σ)  from pre-computed correlation + vol."""
    Sigma   = corr * np.outer(vol, vol)
    Sigma_w = (1.0 - w) * Sigma + w * np.diag(np.diag(Sigma))
    return Sigma_w


def compute_pc_returns(
    w: float,
    corr_dict: dict,
    vols_dict: dict,
    signals: pd.DataFrame,
    monthly_excess: pd.DataFrame,
    oos_start: str,
    oos_end: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each OOS date t with a risk model and a signal:

      1. Build Σ_w, eigendecompose (descending eigenvalue order).
      2. Sign-align each eigenvector so v_k^T s_t >= 0.
      3. Ex ante contribution of PC k:
           ea_k(t) = (v_k^T s_t) / sqrt(λ_k)  * sqrt(12)   [annualised Sharpe]
      4. Realized return of PC k:
           r_k(t)  = v_k^T r_{t+1}                          [monthly, sign-aligned]

    Returns
    -------
    ex_ante_mat   : (T, MAX_PCS)  — ex ante signal-to-vol per PC rank per date
    realized_mat  : (T, MAX_PCS)  — realized sign-aligned monthly PC return
    """
    t0 = pd.Timestamp(oos_start)
    t1 = pd.Timestamp(oos_end)

    risk_dates  = sorted(d for d in corr_dict if t0 <= d <= t1)
    excess_idx  = monthly_excess.index
    excess_vals = monthly_excess.values
    sig_idx     = signals.index

    ex_ante_rows   = []
    realized_rows  = []

    for date in risk_dates:
        # Signal must exist at date t
        if date not in sig_idx:
            continue

        # Need t+1 return
        try:
            t_loc = excess_idx.get_loc(date)
        except KeyError:
            continue
        if t_loc + 1 >= len(excess_idx):
            continue

        corr_df = corr_dict[date]
        vol_s   = vols_dict[date]
        sig_s   = signals.loc[date].dropna()

        # Align assets across risk model, signal, and next return
        common = (corr_df.index
                  .intersection(vol_s.index)
                  .intersection(sig_s.index))
        if len(common) < 2:
            continue

        C = corr_df.loc[common, common].values
        v = vol_s.loc[common].values
        s = sig_s.loc[common].values

        # Next-month excess returns for same assets
        col_locs = [excess_idx.get_loc(date) if False else None]  # placeholder
        col_positions = [monthly_excess.columns.get_loc(c)
                         for c in common if c in monthly_excess.columns]
        if len(col_positions) != len(common):
            continue
        r_next = excess_vals[t_loc + 1, col_positions]
        if np.any(np.isnan(r_next)):
            continue

        # Build Σ_w and eigendecompose
        Sigma_w = build_sigma_w(C, v, w)
        try:
            eigvals, eigvecs = np.linalg.eigh(Sigma_w)
        except np.linalg.LinAlgError:
            continue

        # Sort descending
        order   = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]           # columns = PCs

        n_pcs = len(eigvals)

        # Sign-align: flip v_k so that v_k^T s >= 0
        projections = eigvecs.T @ s           # shape (n_pcs,)
        signs = np.where(projections >= 0, 1.0, -1.0)
        eigvecs = eigvecs * signs             # broadcast over rows

        # Ex ante: |v_k^T s| / sqrt(λ_k)  ×  sqrt(12)
        ea = np.abs(eigvecs.T @ s)            # already sign-aligned so all ≥ 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ea_sharpe = np.where(eigvals > 0, ea / np.sqrt(eigvals), 0.0) * np.sqrt(12)

        # Realized: sign-aligned v_k^T r_{t+1}  (monthly)
        r_pc = eigvecs.T @ r_next

        # Store padded to MAX_PCS
        ea_row  = np.full(MAX_PCS, np.nan)
        r_row   = np.full(MAX_PCS, np.nan)
        ea_row[:n_pcs]  = ea_sharpe[:MAX_PCS]
        r_row[:n_pcs]   = r_pc[:MAX_PCS]

        ex_ante_rows.append(ea_row)
        realized_rows.append(r_row)

    if not ex_ante_rows:
        raise RuntimeError(f"No observations computed for w={w}.")

    return np.array(ex_ante_rows), np.array(realized_rows)


def aggregate(ex_ante_mat: np.ndarray,
              realized_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per PC rank:
      ex_ante_avg  : mean ex ante signal-to-vol  (annualised)
      realized_sr  : annualised realized Sharpe of the PC return series
      n_obs        : valid observation count
    """
    with np.errstate(all="ignore"):
        ex_ante_avg = np.nanmean(ex_ante_mat, axis=0)

        r_mean = np.nanmean(realized_mat, axis=0)
        r_std  = np.nanstd(realized_mat, axis=0, ddof=1)
        realized_sr = np.where(r_std > 0, r_mean / r_std * np.sqrt(12), np.nan)

        n_obs = np.sum(~np.isnan(ex_ante_mat), axis=0)

    return ex_ante_avg, realized_sr, n_obs


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_figure1(
    panel_data: list[tuple],
    oos_start: str = OOS_START,
    oos_end:   str = OOS_END,
    save_path: str | None = None,
):
    """
    Four-panel line chart.
    panel_data: list of (ex_ante_avg, realized_sr, n_obs, label, title)
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    axes = axes.flatten()

    fig.suptitle(
        "Figure 1  —  Ex Ante vs. Realized Return by Principal Component\n"
        f"OOS: {oos_start[:7]} – {oos_end[:7]}",
        fontsize=13, fontweight="bold",
    )

    for ax, (ea, sr, n_obs, lbl, title) in zip(axes, panel_data):
        mask  = (n_obs >= 12) & np.isfinite(ea) & np.isfinite(sr)
        ranks = np.where(mask)[0] + 1   # 1-based
        ea_m  = ea[mask]
        sr_m  = sr[mask]

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

        ax.plot(ranks, ea_m,
                color=COLOUR_EXANTE, linewidth=1.8, marker="o", markersize=3,
                label="Ex ante (signal / vol, annualised)")
        ax.plot(ranks, sr_m,
                color=COLOUR_REALIZED, linewidth=1.8, marker="s", markersize=3,
                label="Realized Sharpe (annualised)")

        ax.set_xlabel("Principal component rank  (1 = largest eigenvalue)", fontsize=9)
        ax.set_ylabel("Annualised Sharpe ratio", fontsize=9)
        ax.set_title(f"({lbl})  {title}", fontsize=10, fontweight="bold", loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.tick_params(labelsize=8)

        # Annotate first two PCs with ratio
        for r, e, s in zip(ranks[:3], ea_m[:3], sr_m[:3]):
            ratio = e / s if s != 0 else np.nan
            if np.isfinite(ratio):
                ax.annotate(f"×{ratio:.1f}",
                            xy=(r, max(e, s)),
                            xytext=(0, 6), textcoords="offset points",
                            fontsize=6, ha="center", color="#555555")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("Figure 1 — Ex Ante vs. Realized Return by PC")
    print(f"OOS: {OOS_START} → {OOS_END}")
    print("=" * 65 + "\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading monthly returns …")
    monthly_ret    = get_monthly_return()
    rf             = get_monthly_risk()
    monthly_excess = calculate_monthly_excess_returns(monthly_ret, rf)

    print("Computing XSMOM signal …")
    signals = compute_xsmom(monthly_excess, LOOKBACK_MONTHS)

    # ── Build both risk models ─────────────────────────────────────────────────
    print(f"\nBuilding pre-shrunk risk model (θ={CORR_PRESHRINK}) …")
    corr_shrunk, vols_shrunk = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=CORR_PRESHRINK, verbose=True)

    print("\nBuilding raw (θ=0) risk model for MVO panel …")
    corr_raw, vols_raw = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=0.0, verbose=False)

    print(f"  Pre-shrunk dates : {len(corr_shrunk)}")
    print(f"  Raw dates        : {len(corr_raw)}\n")

    # ── Compute per-panel statistics ───────────────────────────────────────────
    panel_data = []
    for w, use_preshrink, lbl, title in PANELS:
        corr_d = corr_shrunk if use_preshrink else corr_raw
        vols_d = vols_shrunk if use_preshrink else vols_raw
        shrink_label = f"θ={CORR_PRESHRINK}" if use_preshrink else "θ=0 (no pre-shrink)"

        print(f"Panel {lbl}: w={w}, {shrink_label} …", flush=True)
        ea_mat, r_mat = compute_pc_returns(
            w, corr_d, vols_d, signals, monthly_excess, OOS_START, OOS_END)
        ea_avg, sr_avg, n_obs = aggregate(ea_mat, r_mat)

        # Diagnostic
        valid = n_obs >= 12
        ratio = np.nanmean(ea_avg[valid] / np.where(np.abs(sr_avg[valid]) > 0,
                                                     np.abs(sr_avg[valid]), np.nan))
        print(f"  {valid.sum()} PCs — mean ex ante / |realized SR| = {ratio:.2f}")

        panel_data.append((ea_avg, sr_avg, n_obs, lbl, title))

    # ── Plot ───────────────────────────────────────────────────────────────────
    print("\nPlotting Figure 1 …")
    plot_figure1(
        panel_data,
        oos_start=OOS_START,
        oos_end=OOS_END,
        save_path="Figure1.png",
    )
    print("Done.")


if __name__ == "__main__":
    main()
