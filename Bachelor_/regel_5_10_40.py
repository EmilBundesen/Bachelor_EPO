"""
regel_5_10_40.py
────────────────────────────────────────────────────────────────
Analyserer om EPO-porteføljerne i aktieuniverset overholder
UCITS 5-10-40-reglen:

  5%-grænse : Ingen enkelt aktie må udgøre mere end 5% af porteføljen.
 10%-grænse : Ingen enkelt aktie må udgøre mere end 10% af porteføljen.
 40%-grænse : Summen af alle aktier med en absolut vægt over 5%
              må ikke overstige 40% af porteføljen.

Reglerne anvendes på absolutte (brutto) vægte hver måned.
Kald funktionerne direkte eller kør filen som __main__ efter at
have indlæst data via Stock_Data.load_data().
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from Equity_1 import epo_weights, GAMMA, CANDIDATE_WS

# ── Grænser ──────────────────────────────────────────────────────
LIMIT_5  = 0.05   # 5%-grænse per aktie
LIMIT_10 = 0.10   # 10%-grænse per aktie
LIMIT_40 = 0.40   # 40%-grænse: sum af positioner > 5%


# ── Kernefunktion ────────────────────────────────────────────────

def check_5_10_40(weights: pd.Series) -> dict:
    """
    Tjekker 5-10-40-reglen for ét sæt absolutte portef øljevægte.

    Returnerer en dict med:
      rule_5_ok   : bool — ingen aktie over 5%
      rule_10_ok  : bool — ingen aktie over 10%
      rule_40_ok  : bool — sum af positioner > 5% ≤ 40%
      max_weight  : float — størst absolutte vægt
      sum_above_5 : float — sum af |w| > 5%
      n_above_5   : int   — antal aktier over 5%
      n_above_10  : int   — antal aktier over 10%
    """
    abs_w = weights.abs()
    max_w = abs_w.max() if len(abs_w) > 0 else 0.0
    above_5  = abs_w[abs_w > LIMIT_5]
    above_10 = abs_w[abs_w > LIMIT_10]
    sum_above_5 = above_5.sum()

    return {
        "rule_5_ok":   max_w <= LIMIT_5,
        "rule_10_ok":  max_w <= LIMIT_10,
        "rule_40_ok":  sum_above_5 <= LIMIT_40,
        "max_weight":  max_w,
        "sum_above_5": sum_above_5,
        "n_above_5":   len(above_5),
        "n_above_10":  len(above_10),
    }


# ── Analyse over tid for alle w-værdier ─────────────────────────

def analyze_5_10_40(monthly_excess, xsmom, corr_shrunk, vols,
                    gamma=GAMMA,
                    candidate_ws=None,
                    start="2020-01-01",
                    end="2025-12-31") -> pd.DataFrame:
    """
    Beregner EPO-vægte måned for måned og opsummerer 5-10-40-status
    for hvert w-niveau.

    Returnerer en DataFrame med én række per w-værdi og kolonnerne:
      w | Måneder | 5%-brud (mdr.) | 10%-brud (mdr.) | 40%-brud (mdr.) |
        | Pct. 5%-brud | Pct. 10%-brud | Pct. 40%-brud |
        | Gns. max vægt | Max vægt ever | Gns. sum>5%
    """
    if candidate_ws is None:
        candidate_ws = CANDIDATE_WS

    s, e = pd.to_datetime(start), pd.to_datetime(end)
    rows = []

    for w in candidate_ws:
        n_months      = 0
        n_break_5     = 0
        n_break_10    = 0
        n_break_40    = 0
        max_wt_ever   = 0.0
        max_wts       = []
        sums_above_5  = []

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
            result = check_5_10_40(wts)

            if not result["rule_5_ok"]:
                n_break_5 += 1
            if not result["rule_10_ok"]:
                n_break_10 += 1
            if not result["rule_40_ok"]:
                n_break_40 += 1

            max_wts.append(result["max_weight"])
            sums_above_5.append(result["sum_above_5"])
            max_wt_ever = max(max_wt_ever, result["max_weight"])

        pct = lambda k: f"{k / n_months:.1%}" if n_months > 0 else "N/A"

        rows.append({
            "w":                    f"{w:.0%}",
            "Måneder total":        n_months,
            "5%-brud (mdr.)":       n_break_5,
            "10%-brud (mdr.)":      n_break_10,
            "40%-brud (mdr.)":      n_break_40,
            "Pct. 5%-brud":         pct(n_break_5),
            "Pct. 10%-brud":        pct(n_break_10),
            "Pct. 40%-brud":        pct(n_break_40),
            "Gns. max vægt":        f"{np.mean(max_wts):.2%}" if max_wts else "N/A",
            "Max vægt ever":        f"{max_wt_ever:.2%}",
            "Gns. sum>5%":          f"{np.mean(sums_above_5):.2%}" if sums_above_5 else "N/A",
        })

    return pd.DataFrame(rows)


# ── Tekst-rapport ────────────────────────────────────────────────

def print_5_10_40_report(monthly_excess, xsmom, corr_shrunk, vols,
                          gamma=GAMMA,
                          candidate_ws=None,
                          start="2020-01-01",
                          end="2025-12-31") -> pd.DataFrame:
    """
    Udskriver en læsbar tabel over 5-10-40-overholdelse for alle w-værdier.
    """
    df = analyze_5_10_40(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma=gamma, candidate_ws=candidate_ws,
        start=start, end=end,
    )

    n_assets = monthly_excess.shape[1]
    w = 105

    print(f"\n{'='*w}")
    print("5-10-40-REGLEN — UCITS overholdelse i aktieuniverset")
    print(f"Periode: {start[:7]} → {end[:7]}  |  Aktiver: {n_assets}")
    print(f"  5%-grænse: Ingen enkelt aktie må overstige 5% absolut vægt")
    print(f" 10%-grænse: Ingen enkelt aktie må overstige 10% absolut vægt")
    print(f" 40%-grænse: Summen af positioner over 5% må ikke overstige 40%")
    print(f"{'='*w}")
    print(
        f"  {'w':<8} {'Mdr.':<7}"
        f" {'5%-brud':>9} {'Pct.':>8}"
        f" {'10%-brud':>10} {'Pct.':>8}"
        f" {'40%-brud':>10} {'Pct.':>8}"
        f" {'Gns. max':>10} {'Max ever':>10} {'Gns. sum>5%':>13}"
    )
    print("-" * w)
    for _, row in df.iterrows():
        print(
            f"  {row['w']:<8} {row['Måneder total']:<7}"
            f" {row['5%-brud (mdr.)']:>9} {row['Pct. 5%-brud']:>8}"
            f" {row['10%-brud (mdr.)']:>10} {row['Pct. 10%-brud']:>8}"
            f" {row['40%-brud (mdr.)']:>10} {row['Pct. 40%-brud']:>8}"
            f" {row['Gns. max vægt']:>10} {row['Max vægt ever']:>10}"
            f" {row['Gns. sum>5%']:>13}"
        )
    print("=" * w)

    return df


# ── Tidsserieplot per w-værdi ─────────────────────────────────────

def plot_5_10_40_over_time(monthly_excess, xsmom, corr_shrunk, vols,
                            gamma=GAMMA,
                            w=0.75,
                            start="2020-01-01",
                            end="2025-12-31"):
    """
    Tre paneler over tid for ét fast w-niveau:
      1. Størst absolut vægt per måned (vandret linje ved 5% og 10%)
      2. Antal aktier over 5%-grænsen per måned
      3. Sum af positioner over 5% per måned (vandret linje ved 40%)
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)

    dates, max_wts, n_above_5, sums_above_5 = [], [], [], []

    for date in monthly_excess.loc[s:e].index:
        if date not in corr_shrunk or date not in xsmom.index:
            continue
        wts = epo_weights(xsmom.loc[date], corr_shrunk[date],
                           vols[date], gamma, w)
        if len(wts) == 0:
            continue
        result = check_5_10_40(wts)
        dates.append(date)
        max_wts.append(result["max_weight"] * 100)
        n_above_5.append(result["n_above_5"])
        sums_above_5.append(result["sum_above_5"] * 100)

    if not dates:
        print("Ingen data til plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # Panel 1: Maksimal absolut vægt
    ax = axes[0]
    ax.plot(dates, max_wts, color="steelblue", linewidth=1.5, label="Max |vægt|")
    ax.axhline(LIMIT_5  * 100, color="orange", linestyle="--",
               linewidth=1.2, label="5%-grænse")
    ax.axhline(LIMIT_10 * 100, color="red",    linestyle="--",
               linewidth=1.2, label="10%-grænse")
    ax.fill_between(dates, max_wts, LIMIT_5 * 100,
                    where=[v > LIMIT_5 * 100 for v in max_wts],
                    alpha=0.2, color="orange")
    ax.fill_between(dates, max_wts, LIMIT_10 * 100,
                    where=[v > LIMIT_10 * 100 for v in max_wts],
                    alpha=0.3, color="red")
    ax.set_ylabel("Absolut vægt (%)")
    ax.set_title(f"Størst enkeltposition — EPO w={w:.0%}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Panel 2: Antal aktier over 5%
    ax = axes[1]
    ax.bar(dates, n_above_5, color="orange", alpha=0.7, width=20,
           label="Aktier over 5%")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Antal aktier")
    ax.set_title("Antal aktier der overskrider 5%-grænsen", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 3: Sum af positioner over 5%
    ax = axes[2]
    ax.plot(dates, sums_above_5, color="darkorange", linewidth=1.5,
            label="Sum af positioner > 5%")
    ax.axhline(LIMIT_40 * 100, color="red", linestyle="--",
               linewidth=1.2, label="40%-grænse")
    ax.fill_between(dates, sums_above_5, LIMIT_40 * 100,
                    where=[v > LIMIT_40 * 100 for v in sums_above_5],
                    alpha=0.3, color="red")
    ax.set_ylabel("Samlet vægt (%)")
    ax.set_title("Sum af positioner over 5% (40%-grænse)", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlabel("Måned")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.suptitle(
        f"5-10-40-Reglen — EPO w={w:.0%} | Aktieuniverset | {start[:7]} → {end[:7]}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


# ── Heatmap: brud per aktie over tid ────────────────────────────

def plot_5_10_40_heatmap(monthly_excess, xsmom, corr_shrunk, vols,
                          gamma=GAMMA,
                          w=0.75,
                          start="2020-01-01",
                          end="2025-12-31",
                          top_n=20):
    """
    Heatmap (x=måned, y=aktie) for de top_n aktier der oftest bryder
    5%-grænsen. Cellens farve = absolut vægt; rød = over 5%-grænsen.
    """
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
        print("Ingen data til heatmap.")
        return

    wt_df = pd.DataFrame(records).T.fillna(0)

    violation_counts = (wt_df > LIMIT_5).sum().sort_values(ascending=False)
    top_violators    = violation_counts.head(top_n).index
    plot_df          = wt_df[top_violators].T * 100

    fig, ax = plt.subplots(figsize=(16, 8))

    vmax = max(plot_df.values.max(), LIMIT_5 * 100 + 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=LIMIT_5 * 100, vmax=vmax)
    im   = ax.imshow(plot_df.values, aspect="auto",
                     cmap=plt.cm.RdYlGn_r, norm=norm)

    ax.set_yticks(range(len(top_violators)))
    ax.set_yticklabels(top_violators, fontsize=8)
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in plot_df.columns],
                        rotation=90, fontsize=7)

    plt.colorbar(im, ax=ax, label="Absolut vægt (%)")
    ax.set_title(
        f"5-10-40 Heatmap — EPO w={w:.0%}  "
        f"[rød = over {LIMIT_5:.0%} grænse]\n"
        f"Top {top_n} hyppigste 5%-brud | Periode: {start[:7]} → {end[:7]}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ── Månedlig overholdelseslog ─────────────────────────────────────

def print_monthly_compliance_log(monthly_excess, xsmom, corr_shrunk, vols,
                                  gamma=GAMMA,
                                  w=0.75,
                                  start="2020-01-01",
                                  end="2025-12-31"):
    """
    Udskriver en linje per måned med status for alle tre regler.
    Bruges til at identificere præcist hvornår brud opstår.
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)

    print(f"\n{'='*90}")
    print(f"MÅNEDLIG 5-10-40 COMPLIANCE LOG — EPO w={w:.0%}")
    print(f"Periode: {start[:7]} → {end[:7]}")
    print(f"{'='*90}")
    print(f"  {'Måned':<10} {'5%-regel':>10} {'10%-regel':>11} "
          f"{'40%-regel':>11} {'Max |w|':>10} {'Sum>5%':>10} {'Aktier>5%':>12}")
    print("-" * 90)

    n_ok = 0
    n_total = 0

    for date in monthly_excess.loc[s:e].index:
        if date not in corr_shrunk or date not in xsmom.index:
            continue
        wts = epo_weights(xsmom.loc[date], corr_shrunk[date],
                           vols[date], gamma, w)
        if len(wts) == 0:
            continue

        r = check_5_10_40(wts)
        n_total += 1
        all_ok = r["rule_5_ok"] and r["rule_10_ok"] and r["rule_40_ok"]
        if all_ok:
            n_ok += 1

        status_5  = "OK" if r["rule_5_ok"]  else "BRUD"
        status_10 = "OK" if r["rule_10_ok"] else "BRUD"
        status_40 = "OK" if r["rule_40_ok"] else "BRUD"

        flag = "" if all_ok else "  ◄"
        print(
            f"  {date.strftime('%Y-%m'):<10} "
            f"{status_5:>10} {status_10:>11} {status_40:>11} "
            f"{r['max_weight']:>9.2%} {r['sum_above_5']:>9.2%} "
            f"{r['n_above_5']:>12}{flag}"
        )

    pct_ok = n_ok / n_total if n_total > 0 else 0
    print("=" * 90)
    print(f"  Fuld overholdelse: {n_ok}/{n_total} måneder ({pct_ok:.1%})")
    print("=" * 90)


# ── Sammenligning på tværs af alle w-værdier (plot) ──────────────

def plot_compliance_by_w(monthly_excess, xsmom, corr_shrunk, vols,
                          gamma=GAMMA,
                          candidate_ws=None,
                          start="2020-01-01",
                          end="2025-12-31"):
    """
    Søjlediagram: pct. måneder med brud på 5%, 10% og 40%-reglen
    for hvert w-niveau.
    """
    if candidate_ws is None:
        candidate_ws = CANDIDATE_WS

    df = analyze_5_10_40(
        monthly_excess, xsmom, corr_shrunk, vols,
        gamma=gamma, candidate_ws=candidate_ws,
        start=start, end=end,
    )

    # Konvertér pct.-strenge tilbage til float
    def to_float(s):
        try:
            return float(s.strip("%")) / 100
        except (ValueError, AttributeError):
            return 0.0

    w_labels  = df["w"].tolist()
    pct_5     = [to_float(v) for v in df["Pct. 5%-brud"]]
    pct_10    = [to_float(v) for v in df["Pct. 10%-brud"]]
    pct_40    = [to_float(v) for v in df["Pct. 40%-brud"]]

    x   = np.arange(len(w_labels))
    bw  = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - bw,   pct_5,  width=bw, label="5%-brud",  color="orange",    alpha=0.85)
    ax.bar(x,        pct_10, width=bw, label="10%-brud", color="tomato",     alpha=0.85)
    ax.bar(x + bw,   pct_40, width=bw, label="40%-brud", color="firebrick",  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(w_labels, fontsize=10)
    ax.set_xlabel("EPO w-niveau")
    ax.set_ylabel("Andel måneder med brud")
    ax.set_title(
        f"5-10-40-Reglen — Andel måneder med brud pr. w-niveau\n"
        f"Aktieuniverset | {start[:7]} → {end[:7]}",
        fontsize=12, fontweight="bold"
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6)
    plt.tight_layout()
    plt.show()


# ── Standalone kørsel ─────────────────────────────────────────────

if __name__ == "__main__":
    from Stock_Data import (
        load_data, to_monthly_returns, get_rf_monthly,
        compute_monthly_excess, compute_tsmom_signal,
        START_DATE, END_DATE, BACKTEST_START,
    )
    from Equity_1 import compute_risk_model, RISK_WINDOW, CORR_PRESHRINK, LOOKBACK_MONTHS

    print("Indlæser data...")
    daily, daily_prices, ticker_to_sector = load_data()
    monthly        = to_monthly_returns(daily)
    rf             = get_rf_monthly(START_DATE, END_DATE)
    monthly_excess = compute_monthly_excess(monthly, rf)

    print("Beregner signal...")
    xsmom = compute_tsmom_signal(monthly_excess, ticker_to_sector, LOOKBACK_MONTHS)

    print("Bygger risikomodel...")
    corr_shrunk, vols = compute_risk_model(
        monthly_excess, window=RISK_WINDOW,
        theta=CORR_PRESHRINK, verbose=True,
    )

    START = BACKTEST_START
    END   = END_DATE

    # ── Samlet rapport (alle w-værdier) ──────────────────────────
    print_5_10_40_report(
        monthly_excess, xsmom, corr_shrunk, vols,
        start=START, end=END,
    )

    # ── Månedlig log for w=0.75 ───────────────────────────────────
    print_monthly_compliance_log(
        monthly_excess, xsmom, corr_shrunk, vols,
        w=0.75, start=START, end=END,
    )

    # ── Plots ─────────────────────────────────────────────────────
    plot_5_10_40_over_time(
        monthly_excess, xsmom, corr_shrunk, vols,
        w=0.75, start=START, end=END,
    )

    plot_compliance_by_w(
        monthly_excess, xsmom, corr_shrunk, vols,
        start=START, end=END,
    )

    plot_5_10_40_heatmap(
        monthly_excess, xsmom, corr_shrunk, vols,
        w=0.75, start=START, end=END, top_n=20,
    )
