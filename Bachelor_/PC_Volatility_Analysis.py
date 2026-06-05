"""
PC_Volatility_Analysis.py

Replicates Figure 1 (A, B, C, D) from Pedersen's EPO article:
  "Ex ante volatility vs. realized volatility by principal component."

For each shrinkage level w in {0.0, 0.5, 0.75, 1.0}:
  - At each date t in the OOS period, build Σ_w and eigendecompose it
  - Ex ante vol of PC k  = sqrt(λ_k(t)) * sqrt(12)   [annualised]
  - Realized return of PC k at t = v_k(t)^T * r_{t+1}
  - Scatter plot: average ex ante vol (x) vs realised vol (y) per PC rank
  - Includes 45° calibration line

Each panel (A–D) uses a different w so the reader can see how diagonal
shrinkage corrects the systematic overestimation of top-PC volatility.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Allow importing from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from Equity_1 import (
    get_monthly_return,
    get_monthly_risk,
    calculate_monthly_excess_returns,
    compute_risk_model,
    DATA_START_DATE,
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    RISK_WINDOW,
    CORR_PRESHRINK,
    MIN_VOL,
)

# ── Figure 1 panel configuration ──────────────────────────────────────────────
PANEL_WS     = [0.0, 0.5, 0.75, 1.0]          # one panel per w value
PANEL_LABELS = ["A", "B", "C", "D"]
PANEL_TITLES = [
    r"$w = 0$  (MVO)",
    r"$w = 0.5$",
    r"$w = 0.75$",
    r"$w = 1$  (Diagonal)",
]

OOS_START = BACKTEST_START_DATE
OOS_END   = BACKTEST_END_DATE

# Colour palette matching Pedersen-style figures
COLOUR_EXANTE   = "#1f77b4"   # blue
COLOUR_REALIZED = "#d62728"   # red
COLOUR_DIAGONAL = "#888888"   # grey 45° line

MAX_PCS = 49  # cap at universe size


# ── Core computation ──────────────────────────────────────────────────────────

def build_sigma_w(corr: np.ndarray, vol: np.ndarray, w: float) -> np.ndarray:
    """Reconstruct Σ_w = (1-w)·Σ + w·diag(Σ) from correlation + vol arrays."""
    Sigma   = corr * np.outer(vol, vol)
    Sigma_w = (1.0 - w) * Sigma + w * np.diag(np.diag(Sigma))
    return Sigma_w


def compute_pc_stats(
    w: float,
    corr_dict: dict,
    vols_dict: dict,
    monthly_excess: pd.DataFrame,
    oos_start: str = OOS_START,
    oos_end:   str = OOS_END,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For every date t in [oos_start, oos_end] that has a risk model:
      1. Build Σ_w(t), eigen-decompose (sorted descending by eigenvalue).
      2. Record ex ante vol for each PC rank k: sqrt(λ_k) * sqrt(12).
      3. Project next-month excess return onto each eigenvector.

    Returns
    -------
    ex_ante_by_rank   : shape (n_dates, max_rank)  – annualised ex ante vol
    realized_ret_by_rank : shape (n_dates, max_rank)  – projected return (monthly)
    """
    t0 = pd.Timestamp(oos_start)
    t1 = pd.Timestamp(oos_end)

    risk_dates = sorted([d for d in corr_dict if t0 <= d <= t1])

    ex_ante_rows    = []
    realized_ret_rows = []

    excess_idx  = monthly_excess.index
    excess_vals = monthly_excess.values

    for date in risk_dates:
        # Find index of date in excess returns
        try:
            t_loc = excess_idx.get_loc(date)
        except KeyError:
            continue

        # Need next-month return
        if t_loc + 1 >= len(excess_idx):
            continue

        corr_df = corr_dict[date]
        vol_s   = vols_dict[date]

        common = corr_df.index.intersection(vol_s.index)
        if len(common) < 2:
            continue

        C = corr_df.loc[common, common].values
        v = vol_s.loc[common].values

        # Build Σ_w and eigendecompose (eigh for symmetric matrices)
        Sigma_w = build_sigma_w(C, v, w)
        try:
            eigvals, eigvecs = np.linalg.eigh(Sigma_w)
        except np.linalg.LinAlgError:
            continue

        # Sort descending by eigenvalue
        order   = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]          # columns are PCs

        # Clamp to non-negative (numerical noise)
        eigvals = np.maximum(eigvals, 0.0)

        ex_ante_vol = np.sqrt(eigvals) * np.sqrt(12)   # annualised

        # Next-month returns for the common industries
        nxt_loc   = t_loc + 1
        r_next_all = excess_vals[nxt_loc]
        col_idx   = [monthly_excess.columns.get_loc(c) for c in common
                     if c in monthly_excess.columns]
        if len(col_idx) != len(common):
            continue

        r_next = r_next_all[col_idx]
        if np.any(np.isnan(r_next)):
            continue

        # Project r_{t+1} onto each eigenvector (PC return = v_k^T r)
        pc_returns = eigvecs.T @ r_next        # shape (n_pcs,)

        n_pcs = len(eigvals)
        # Pad rows to MAX_PCS with NaN so we can stack into a matrix
        ea_row  = np.full(MAX_PCS, np.nan)
        ret_row = np.full(MAX_PCS, np.nan)
        ea_row[:n_pcs]  = ex_ante_vol[:MAX_PCS]
        ret_row[:n_pcs] = pc_returns[:MAX_PCS]

        ex_ante_rows.append(ea_row)
        realized_ret_rows.append(ret_row)

    if len(ex_ante_rows) == 0:
        raise RuntimeError(f"No observations found for w={w} in OOS period.")

    ex_ante_mat    = np.array(ex_ante_rows)       # (T, MAX_PCS)
    realized_mat   = np.array(realized_ret_rows)  # (T, MAX_PCS)

    return ex_ante_mat, realized_mat


def aggregate_pc_stats(
    ex_ante_mat: np.ndarray,
    realized_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate across time to get per-rank statistics.

    Returns
    -------
    avg_ex_ante  : mean ex ante vol per PC rank  (annualised)
    real_vol     : std of PC returns per rank × sqrt(12) (annualised realized vol)
    n_obs        : number of valid observations per rank
    """
    with np.errstate(all="ignore"):
        avg_ex_ante = np.nanmean(ex_ante_mat, axis=0)
        real_vol    = np.nanstd(realized_mat, axis=0, ddof=1) * np.sqrt(12)
        n_obs       = np.sum(~np.isnan(ex_ante_mat), axis=0)

    return avg_ex_ante, real_vol, n_obs


# ── Plotting ──────────────────────────────────────────────────────────────────

def _style_axes(ax, title: str, panel_label: str):
    ax.set_xlabel("Ex ante volatility (annualised, %)", fontsize=9)
    ax.set_ylabel("Realized volatility (annualised, %)", fontsize=9)
    ax.set_title(f"({panel_label})  {title}", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")


def plot_figure1(
    panel_results: list[tuple],
    oos_start: str = OOS_START,
    oos_end:   str = OOS_END,
    save_path: str | None = None,
):
    """
    panel_results : list of (avg_ex_ante, real_vol, n_obs) for each panel w.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    fig.suptitle(
        "Figure 1  —  Ex Ante vs. Realized Volatility by Principal Component\n"
        f"OOS period: {oos_start[:7]} – {oos_end[:7]}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for idx, ((avg_ea, real_vol, n_obs), label, title, w) in enumerate(
        zip(panel_results, PANEL_LABELS, PANEL_TITLES, PANEL_WS)
    ):
        ax = axes[idx]

        # Only use ranks with sufficient observations
        mask = (n_obs >= 12) & np.isfinite(avg_ea) & np.isfinite(real_vol)
        ea  = avg_ea[mask] * 100     # → percent
        rv  = real_vol[mask] * 100   # → percent
        n   = n_obs[mask]
        ranks = np.where(mask)[0] + 1  # 1-based PC rank

        # 45° line range
        lo = min(ea.min(), rv.min()) * 0.9
        hi = max(ea.max(), rv.max()) * 1.1
        diag = np.linspace(lo, hi, 100)
        ax.plot(diag, diag, color=COLOUR_DIAGONAL, linewidth=1.2,
                linestyle="--", zorder=1, label="45° line")

        # Scatter coloured by PC rank (dark = low rank = large PC)
        sc = ax.scatter(
            ea, rv,
            c=ranks,
            cmap="viridis_r",
            s=40,
            edgecolors="k",
            linewidths=0.3,
            zorder=3,
        )
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("PC rank", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # Annotate first few and last PC
        for i, (x, y, r) in enumerate(zip(ea, rv, ranks)):
            if r <= 3 or r == ranks[-1]:
                ax.annotate(
                    f"PC{r}",
                    xy=(x, y), xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6, color="#333333",
                )

        # Regression line through origin to show bias
        if len(ea) >= 2:
            slope = np.dot(ea, rv) / np.dot(ea, ea)
            x_fit = np.linspace(ea.min(), ea.max(), 100)
            ax.plot(x_fit, slope * x_fit, color=COLOUR_REALIZED,
                    linewidth=1.0, linestyle="-", alpha=0.7,
                    label=f"Fit slope={slope:.2f}")

        _style_axes(ax, title, label)
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()

    return fig


def plot_figure1_line(
    panel_results: list[tuple],
    oos_start: str = OOS_START,
    oos_end:   str = OOS_END,
    save_path: str | None = None,
):
    """
    Alternative line-chart version of Figure 1:
    X-axis = PC rank (1 = largest eigenvalue), Y-axis = vol.
    Two lines per panel: ex ante and realized.
    Matches a common Pedersen-style presentation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    fig.suptitle(
        "Figure 1  —  Ex Ante vs. Realized Volatility by Principal Component\n"
        f"OOS period: {oos_start[:7]} – {oos_end[:7]}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for idx, ((avg_ea, real_vol, n_obs), label, title, w) in enumerate(
        zip(panel_results, PANEL_LABELS, PANEL_TITLES, PANEL_WS)
    ):
        ax = axes[idx]

        mask  = (n_obs >= 12) & np.isfinite(avg_ea) & np.isfinite(real_vol)
        ranks = np.where(mask)[0] + 1
        ea    = avg_ea[mask] * 100
        rv    = real_vol[mask] * 100

        ax.plot(ranks, ea, color=COLOUR_EXANTE,   linewidth=1.8,
                marker="o", markersize=3, label="Ex ante vol")
        ax.plot(ranks, rv, color=COLOUR_REALIZED, linewidth=1.8,
                marker="s", markersize=3, label="Realized vol")

        ax.set_xlabel("Principal component rank", fontsize=9)
        ax.set_ylabel("Annualised volatility (%)", fontsize=9)
        ax.set_title(f"({label})  {title}", fontsize=10,
                     fontweight="bold", loc="left")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.tick_params(labelsize=8)

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
    print("Figure 1 — Ex Ante vs. Realized Volatility by PC")
    print(f"OOS: {OOS_START} → {OOS_END}")
    print("=" * 65 + "\n")

    # ── Load data ──
    print("Loading monthly returns …")
    monthly_ret = get_monthly_return()
    rf          = get_monthly_risk()
    monthly_excess = calculate_monthly_excess_returns(monthly_ret, rf)

    # ── Build risk model (pre-shrunk, same as Equity_1) ──
    print(f"\nBuilding {RISK_WINDOW}-month rolling risk model "
          f"(θ={CORR_PRESHRINK}) …")
    corr_dict, vols_dict = compute_risk_model(
        monthly_excess, window=RISK_WINDOW,
        theta=CORR_PRESHRINK, verbose=True,
    )
    print(f"  Risk model dates: {len(corr_dict)}\n")

    # ── Compute PC stats for each panel w ──
    panel_results = []
    for w, lbl in zip(PANEL_WS, PANEL_LABELS):
        print(f"Panel {lbl}: computing PC stats for w = {w} …", flush=True)
        ex_ante_mat, realized_mat = compute_pc_stats(
            w, corr_dict, vols_dict, monthly_excess, OOS_START, OOS_END
        )
        avg_ea, real_vol, n_obs = aggregate_pc_stats(ex_ante_mat, realized_mat)

        # Quick diagnostic
        valid = n_obs >= 12
        n_valid = valid.sum()
        ea_pct = avg_ea[valid] * 100
        rv_pct = real_vol[valid] * 100
        ratio  = np.nanmean(ea_pct / rv_pct)
        print(f"  {n_valid} PCs — mean(ex ante)/mean(realized) = {ratio:.3f}")

        panel_results.append((avg_ea, real_vol, n_obs))

    # ── Plot scatter version (primary — matches paper) ──
    print("\nPlotting Figure 1 (scatter) …")
    plot_figure1(
        panel_results,
        oos_start=OOS_START,
        oos_end=OOS_END,
        save_path="Figure1_scatter.png",
    )

    # ── Plot line version (alternative presentation) ──
    print("Plotting Figure 1 (line chart) …")
    plot_figure1_line(
        panel_results,
        oos_start=OOS_START,
        oos_end=OOS_END,
        save_path="Figure1_line.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
