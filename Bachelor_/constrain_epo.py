"""
weight_violation_diagnostics.py
────────────────────────────────
Analyserer hvor mange EPO-vægte der bryder ±7,5% grænsen i aktieuniverset.
Kald denne funktion fra Stock_Data.py's main() uden at ændre selve backtests.
"""

import pandas as pd
import numpy as np
from Equity_1 import epo_weights, backtest_strategy, GAMMA, CANDIDATE_WS


WEIGHT_BOUND = 0.075  # ±7,5 pct.


def analyze_weight_violations(monthly_excess, xsmom, corr_shrunk, vols,
                                gamma=GAMMA,
                                candidate_ws=None,
                                bound=WEIGHT_BOUND,
                                start="2020-01-01",
                                end="2025-12-31") -> pd.DataFrame:
    """
    For hver w-værdi: beregn EPO-vægte hver måned og tæl violations.

    Returnerer en DataFrame med kolonner:
        w | Måneder | Mdr. med violation | Pct. mdr. | Gns. violations/mdr |
          | Gns. max vægt | Max vægt ever | Gns. violation-størrelse
    """
    if candidate_ws is None:
        candidate_ws = CANDIDATE_WS

    s, e = pd.to_datetime(start), pd.to_datetime(end)
    rows = []

    for w in candidate_ws:
        n_months        = 0
        n_violation_mdr = 0
        total_viols     = 0
        max_wt_ever     = 0.0
        max_wts         = []
        viol_sizes      = []

        for date in monthly_excess.loc[s:e].index:
            if date not in corr_shrunk or date not in xsmom.index:
                continue

            wts = epo_weights(
                xsmom.loc[date], corr_shrunk[date],
                vols[date], gamma, w,
            )
            if len(wts) == 0:
                continue

            n_months += 1
            abs_wts = wts.abs()
            max_wt  = abs_wts.max()
            max_wts.append(max_wt)
            max_wt_ever = max(max_wt_ever, max_wt)

            viols = abs_wts[abs_wts > bound]
            if len(viols) > 0:
                n_violation_mdr += 1
                total_viols     += len(viols)
                viol_sizes.extend((viols - bound).tolist())

        rows.append({
            "w":                      f"{w:.0%}",
            "Måneder total":          n_months,
            "Mdr. med violation":     n_violation_mdr,
            "Pct. mdr. med viol.":    f"{n_violation_mdr / n_months:.1%}" if n_months > 0 else "N/A",
            "Gns. viol./mdr.":        round(total_viols / n_months, 2) if n_months > 0 else 0,
            "Gns. max vægt":          f"{np.mean(max_wts):.2%}" if max_wts else "N/A",
            "Max vægt ever":          f"{max_wt_ever:.2%}",
            "Gns. overskridelse":     f"{np.mean(viol_sizes):.2%}" if viol_sizes else "0.00%",
        })

    return pd.DataFrame(rows)


def print_violation_report(monthly_excess, xsmom, corr_shrunk, vols,
                            gamma=GAMMA,
                            candidate_ws=None,
                            bound=WEIGHT_BOUND,
                            start="2020-01-01",
                            end="2025-12-31"):
    """
    Printer en læsbar tabel over violations.
    Kald denne direkte fra main() i Stock_Data.py.
    """
    df = analyze_weight_violations(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma=gamma, candidate_ws=candidate_ws,
        bound=bound, start=start, end=end,
    )

    print(f"\n{'='*90}")
    print(f"WEIGHT VIOLATIONS — EPO-vægte der overskrider ±{bound:.1%}")
    print(f"Periode: {start[:7]} → {end[:7]}  |  Aktiver: {monthly_excess.shape[1]}")
    print(f"{'='*90}")
    print(f"  {'w':<8} {'Mdr.':<8} {'Mdr. m. viol.':<15} {'Pct.':<10} "
          f"{'Gns. viol./mdr.':<17} {'Gns. max vægt':<15} "
          f"{'Max vægt ever':<15} {'Gns. overskr.'}")
    print("-" * 90)
    for _, row in df.iterrows():
        print(f"  {row['w']:<8} {row['Måneder total']:<8} "
              f"{row['Mdr. med violation']:<15} {row['Pct. mdr. med viol.']:<10} "
              f"{row['Gns. viol./mdr.']:<17} {row['Gns. max vægt']:<15} "
              f"{row['Max vægt ever']:<15} {row['Gns. overskridelse']}")
    print("=" * 90)

    return df


def plot_violation_heatmap(monthly_excess, xsmom, corr_shrunk, vols,
                            gamma=GAMMA, w=0.75,
                            bound=WEIGHT_BOUND,
                            start="2020-01-01",
                            end="2025-12-31",
                            top_n=20):
    """
    Heatmap: x=måned, y=aktie (top_n mest hyppige violators).
    Cellens farve = absolut vægt (rød hvis > bound).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    s, e = pd.to_datetime(start), pd.to_datetime(end)
    records = {}

    for date in monthly_excess.loc[s:e].index:
        if date not in corr_shrunk or date not in xsmom.index:
            continue
        wts = epo_weights(xsmom.loc[date], corr_shrunk[date],
                           vols[date], gamma, w)
        if len(wts) > 0:
            records[date] = wts.abs()

    if not records:
        print("Ingen data.")
        return

    wt_df = pd.DataFrame(records).T.fillna(0)

    # Find de top_n aktier med flest violations
    violation_counts = (wt_df > bound).sum().sort_values(ascending=False)
    top_violators    = violation_counts.head(top_n).index

    plot_df = wt_df[top_violators].T * 100  # til pct.

    fig, ax = plt.subplots(figsize=(16, 8))

    # Colormap: grøn → gul → rød, med grænse ved 7,5%
    cmap = plt.cm.RdYlGn_r
    vmax = max(plot_df.values.max(), bound * 100 + 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=bound * 100, vmax=vmax)

    im = ax.imshow(plot_df.values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_yticks(range(len(top_violators)))
    ax.set_yticklabels(top_violators, fontsize=8)
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in plot_df.columns],
                        rotation=90, fontsize=7)

    plt.colorbar(im, ax=ax, label="Absolut vægt (%)")
    ax.set_title(
        f"EPO-vægte (w={w}) — {top_n} mest hyppige violators  "
        f"[rød = over {bound:.1%} grænse]\n"
        f"Periode: {start[:7]} → {end[:7]}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()