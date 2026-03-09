"""
EPO Industry Momentum — Equity 1
Pedersen, Babu & Levine (2021), Table 1 / Table 5

Equity 1 spec (Table 1):
  - Signal      : XSMOM (Eq. 24–25), 12-month lookback
  - Risk model  : 60-month equal-weighted rolling window on MONTHLY returns
                  5% pre-shrinkage of correlation toward identity  (Eq. 10)
  - Method      : Simple EPO  EPO^s(w) = (1/γ) Σ_w^{-1} s          (Eq. 20)
                  Σ_w = (1−w)Σ + w·diag(Σ)                          (Eq. 19)

Special case:
  w = 1.0  →  EPO^s(w=1) = (1/γ) · s_t^i / (σ_t^i)^2
  (signal-weighted, vol-scaled; equivalent to INDMOM up to normalisation)

Data:
  - Monthly industry returns from Kenneth French 49-industry file
  - Monthly RF from Fama-French factors file
  - NO daily data used for Equity 1
"""

import numpy as np
import pandas as pd

# Konstanter
DATA_START_DATE     = "1927-01-01"
BACKTEST_START_DATE = "1942-01-01"
BACKTEST_END_DATE   = "2018-12-31"

LOOKBACK_MONTHS  = 12    # XSMOM signal lookback (Eq. 24)
RISK_WINDOW      = 60    # Rolling window for covariance estimation (months)
CORR_PRESHRINK   = 0.05  # θ: 5% pre-shrinkage toward I (Table 1, Equity 1)
GAMMA            = 3     # Risk aversion γ (cancels in Sharpe ratio)
MIN_VOL          = 1e-4  # Floor on vol to avoid division by zero

CANDIDATE_WS     = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]
MIN_HISTORY_OOS  = 180 #15 år
PERCENT_TO_DECIMAL   = 100.0
Missing_values = [-99.99, -999]

#Dataindlæsning - månedlig
def get_monthly_return() -> pd.DataFrame:
    df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/49_Industry_monthly.csv",
        sep=",",
        header=6
    )

    df = df.iloc[:1194]  # begræns periode
    df = df[df[df.columns[0]].astype(str).str.match(r"^\d{6}$", na=False)]
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date")

    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace(Missing_values, np.nan, inplace=True)
    df = df.sort_index()
    df = df.loc[DATA_START_DATE:BACKTEST_END_DATE]
    df = df[~df.index.duplicated(keep="first")]

    return df

def get_monthly_risk() -> pd.DataFrame:
    rf_df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/Månedlig_rf.csv",
        sep=",",
        skiprows=3
    )
    rf_df = rf_df[rf_df.iloc[:, 0].astype(str).str.match(r"^\d{6,8}$", na=False)]
    rf_df = rf_df.rename(columns={rf_df.columns[0]: "Date"})
    rf_df["Date"] = pd.to_datetime(rf_df["Date"].astype(str), format="%Y%m")
    rf_df = rf_df.set_index("Date")

    rf_df["RF"] = pd.to_numeric(rf_df["RF"], errors="coerce") / PERCENT_TO_DECIMAL
    rf_df = rf_df.loc[DATA_START_DATE:BACKTEST_END_DATE]

    return rf_df

def calculate_monthly_excess_returns(returns_df: pd.DataFrame,
                                    rf_monthly: pd.DataFrame) -> pd.DataFrame:
    returns_decimal = returns_df / PERCENT_TO_DECIMAL
    excess = returns_decimal.subtract(rf_monthly["RF"], axis=0)
    return excess

# Dataindlæsning daglig
def convert_monthly_rf_to_daily(rf_monthly: pd.DataFrame,
                                trading_dates: pd.DatetimeIndex) -> pd.DataFrame:

    rf_daily = rf_monthly.reindex(trading_dates, method='ffill')
    rf_daily["year_month"] = rf_daily.index.to_period("M")
    trading_days_per_month = rf_daily.groupby("year_month").size()
    rf_daily["trading_days"] = rf_daily["year_month"].map(trading_days_per_month)
    rf_daily["RF"] = (1 + rf_daily["RF"]) ** (1 / rf_daily["trading_days"]) - 1
    rf_daily = rf_daily.drop(columns=["year_month", "trading_days"])
    return rf_daily

def get_daily_return() -> pd.DataFrame:
    df = pd.read_csv(
        "/Users/emilbundesen/Desktop/Bachelor/Data/49_Industry_Portfolios_Daily.csv",
        sep=",",
        header=5,
        low_memory=False
    )

    midpoint = len(df) // 2
    df = df.iloc[:midpoint]  # tager kun value-weigthed daglig afkast

    df = df[df[df.columns[0]].astype(str).str.match(r"^\d{8}$", na=False)]  # Behold kun rækker med faktiske datoer

    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")  # konvertere til date-format
    df = df.set_index("Date")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # konvertere daglig afkast til numerisk

    df.replace(Missing_values, np.nan, inplace=True)  # fjerner missing values

    df = df.sort_index()
    df_clean = df.loc[DATA_START_DATE:BACKTEST_END_DATE]
    df_clean = df_clean[~df_clean.index.duplicated(keep='first')]

    return df_clean

def calculate_daily_excess_returns(returns_df: pd.DataFrame,
                                  rf_daily: pd.DataFrame) -> pd.DataFrame:
    returns_decimal = returns_df / PERCENT_TO_DECIMAL
    excess = returns_decimal.subtract(rf_daily["RF"], axis=0)
    return excess


# XSMOM - Signal - (lign. 24–25).

def compute_xsmom(monthly_ret: pd.DataFrame,
                  lookback: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    roll = (monthly_ret
            .rolling(window=lookback, min_periods=lookback)
            .sum())

    out = pd.DataFrame(np.nan, index=roll.index, columns=roll.columns)

    for date, row in roll.iterrows():
        avail = row.dropna()
        if len(avail) < 2:
            continue

        demeaned = avail - avail.mean()        # cross-sectional demean

        pos = demeaned[demeaned > 0].sum()
        neg = demeaned[demeaned < 0].sum()
        if pos == 0 or neg == 0:
            continue

        c_t = 1.0 / max(pos, abs(neg))        # Eq. 25
        out.loc[date, demeaned.index] = c_t * demeaned

    return out


#Risikomodel
def compute_risk_model(monthly_excess, window=RISK_WINDOW,
                       theta=CORR_PRESHRINK, verbose=True):
    idx = monthly_excess.index
    cols = monthly_excess.columns
    data = monthly_excess.values.astype(np.float64)

    corr_dict = {}
    vols_dict = {}

    n_total = len(idx)
    n_iters = n_total - window
    report = max(1, n_iters // 20)

    for i in range(window, n_total):
        if verbose and (i - window) % report == 0:
            pct = 100.0 * (i - window) / n_iters

        date = idx[i]
        window_data = data[i - window: i]

        # Behold KUN industrier med FULD historik — ingen NaN-imputation
        n_obs = np.sum(~np.isnan(window_data), axis=0)
        valid = n_obs == window  # kræv præcis fuld historik

        if valid.sum() < 2:
            # Fallback: brug industrier med mindst 90% af observationer
            valid = n_obs >= int(0.9 * window)
            if valid.sum() < 2:
                continue

        W = window_data[:, valid]
        c = cols[valid]

        # Fjern rækker med NaN (de få resterende efter 90% filter)
        row_complete = ~np.isnan(W).any(axis=1)
        W = W[row_complete]

        if W.shape[0] < 2:
            continue

        K = W.shape[0]

        # Artiklens kovariansestimator (footnote 17) — ingen imputation
        W_dm = W - W.mean(axis=0)
        Sigma_raw = (W_dm.T @ W_dm) / (K - 1)

        # Udtræk vol
        var_diag = np.diag(Sigma_raw)
        nonzero = var_diag > MIN_VOL ** 2
        if nonzero.sum() < 2:
            continue

        Sigma_raw = Sigma_raw[np.ix_(nonzero, nonzero)]
        c = c[nonzero]
        vol = np.sqrt(np.diag(Sigma_raw))

        # Korrelationsmatrix
        denom = np.outer(vol, vol)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_s = np.where(denom > 0, Sigma_raw / denom, 0.0)
        np.fill_diagonal(corr_s, 1.0)
        corr_s = np.clip(corr_s, -1.0, 1.0)

        # Pre-shrinkage
        n = len(c)
        corr_p = (1.0 - theta) * corr_s + theta * np.eye(n)

        corr_dict[date] = pd.DataFrame(corr_p, index=c, columns=c)
        vols_dict[date] = pd.Series(vol, index=c)

    if verbose:
        print("  Risk model: 100.0% — done.", flush=True)

    return corr_dict, vols_dict


#EPO vægte (lign. 19 og 20)

def epo_weights(signal:  pd.Series,
                corr:    pd.DataFrame,
                vols:    pd.Series,
                gamma:   float,
                w:       float,
                min_vol: float = MIN_VOL) -> pd.Series:

    common = (signal.dropna().index
              .intersection(corr.index)
              .intersection(vols.index))
    if len(common) < 2:
        return pd.Series(dtype=float)

    s_vec = signal.loc[common]
    C     = corr.loc[common, common]
    v     = vols.loc[common]

    # Drop near-zero-vol industries
    ok    = v >= min_vol
    s_vec = s_vec[ok]; v = v[ok]; C = C.loc[ok, ok]
    if len(s_vec) < 2:
        return pd.Series(dtype=float)

    v_a = v.values
    s_a = s_vec.values
    C_a = C.values

    # Step 1: Σ = D · Ω̃ · D
    Sigma   = C_a * np.outer(v_a, v_a)

    # Step 2: Σ_w = (1−w)·Σ + w·diag(Σ)   (Eq. 19)
    Sigma_d = np.diag(np.diag(Sigma))
    Sigma_w = (1.0 - w) * Sigma + w * Sigma_d

    # Step 3: x = (1/γ) · Σ_w^{-1} · s   (Eq. 20)
    try:
        Sigma_inv = np.linalg.inv(Sigma_w)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma_w)

    x_raw = (1.0 / gamma) * (Sigma_inv @ s_a)
    x_s   = pd.Series(x_raw, index=s_vec.index)

    # Step 4: Unit-leverage normalisation
    pos = x_s[x_s > 0].sum()
    neg = x_s[x_s < 0].sum()
    if pos == 0 or neg == 0:
        return pd.Series(dtype=float)

    return x_s / max(pos, abs(neg))


#Backtest

def backtest_strategy(monthly_excess: pd.DataFrame,
                      weight_fn,
                      name: str) -> pd.Series:

    idx = monthly_excess.index
    rets, dates = [], []

    for t in range(len(idx) - 1):
        date = idx[t]
        wts  = weight_fn(date)
        if len(wts) == 0:
            continue

        nxt   = idx[t + 1]
        r     = monthly_excess.loc[nxt, wts.index].dropna()
        if len(r) == 0:
            continue

        w_aln = wts.reindex(r.index).dropna()
        r_aln = r.reindex(w_aln.index)
        rets.append((w_aln.values * r_aln.values).sum())
        dates.append(nxt)

    return pd.Series(rets, index=dates, name=name)


def backtest_epo_fixed_w(monthly_excess, signals, corr_dict, vols_dict,
                         gamma, w) -> pd.Series:
    """EPO with a fixed shrinkage parameter w."""
    risk_dates = set(corr_dict)
    sig_dates  = set(signals.index)

    def weight_fn(date):
        if date not in risk_dates or date not in sig_dates:
            return pd.Series(dtype=float)
        return epo_weights(signals.loc[date], corr_dict[date],
                           vols_dict[date], gamma, w)

    return backtest_strategy(monthly_excess, weight_fn,
                             name=f"EPO_w_{w:.2f}")


def backtest_indmom(monthly_excess: pd.DataFrame,
                    signals: pd.DataFrame) -> pd.Series:
    """
    INDMOM benchmark: apply XSMOM signal weights directly, without any
    covariance-based optimisation.
    """
    sig_dates = set(signals.index)

    def weight_fn(date):
        if date not in sig_dates:
            return pd.Series(dtype=float)
        return signals.loc[date].dropna()   # already unit-leverage normalised

    return backtest_strategy(monthly_excess, weight_fn, name="INDMOM")


def backtest_equal_weight(monthly_excess: pd.DataFrame) -> pd.Series:
    """1/N equal-weight benchmark across all available industries each month."""
    idx = monthly_excess.index
    rets, dates = [], []

    for t in range(len(idx) - 1):
        nxt = idx[t + 1]
        r   = monthly_excess.loc[nxt].dropna()
        if len(r) == 0:
            continue
        rets.append(r.mean())
        dates.append(nxt)

    return pd.Series(rets, index=dates, name="EW_1N")


def backtest_mvo_no_shrink(monthly_excess, signals, corr_dict_raw,
                            vols_dict, gamma) -> pd.Series:

    risk_dates = set(corr_dict_raw)
    sig_dates  = set(signals.index)

    def weight_fn(date):
        if date not in risk_dates or date not in sig_dates:
            return pd.Series(dtype=float)
        return epo_weights(signals.loc[date], corr_dict_raw[date],
                           vols_dict[date], gamma, w=0.0)

    return backtest_strategy(monthly_excess, weight_fn, name="MVO_no_shrink")


# Dynamisk EPO OOS

def build_epo_panel(monthly_excess, signals, corr_dict, vols_dict,
                    gamma, candidate_ws) -> pd.DataFrame:
    """Run all fixed-w EPO strategies and collect into a single panel."""
    parts = []
    for w in candidate_ws:
        print(f"  w = {w:.2f} …", flush=True)
        parts.append(backtest_epo_fixed_w(
            monthly_excess, signals, corr_dict, vols_dict, gamma, w))
    return pd.concat(parts, axis=1).sort_index()


def build_dynamic_oos_epo(panel: pd.DataFrame,
                          oos_start: str,
                          min_history: int = MIN_HISTORY_OOS) -> pd.Series:

    panel = panel.sort_index().dropna(how="all")
    t0 = pd.to_datetime(oos_start)

    # Find første dato hvor vi har min_history måneder af historik
    # og datoen er >= oos_start
    rets, dates = [], []

    for i in range(1, len(panel)):
        date = panel.index[i]

        # Kun datoer fra og med oos_start
        if date < t0:
            continue

        past = panel.iloc[:i]  # al historik før dato t

        # Kræv mindst min_history måneder
        if len(past) < min_history:
            continue

        # Vælg w* = argmax Sharpe over al tilgængelig historik
        best_w, best_sr = None, -np.inf
        for col in panel.columns:
            r = past[col].dropna()
            if len(r) < min_history:
                continue
            sr = sharpe_ratio(r)
            if sr > best_sr:
                best_sr = sr
                best_w = col

        if best_w is None:
            continue

        rets.append(panel.loc[date, best_w])
        dates.append(date)

    s = pd.Series(rets, index=dates, name="EPO_OOS_dynamic")
    print(f"  OOS EPO første handel: {s.index.min().strftime('%Y-%m')}")
    print(f"  OOS EPO observationer: {len(s)}")
    return s


# Performance
def sharpe_ratio(r: pd.Series) -> float:
    """Annualised Sharpe ratio (monthly data, annualised by √12)."""
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    s = r.std(ddof=1)
    return float((r.mean() / s) * np.sqrt(12)) if s > 0 else np.nan


def performance_summary(r: pd.Series, name: str) -> dict:
    r    = r.dropna()
    nyrs = len(r) / 12.0
    cum  = (1 + r).prod() - 1
    return {
        "Strategy":    name,
        "Ann. Return": round((1 + cum) ** (1 / nyrs) - 1, 4) if nyrs > 0 else np.nan,
        "Ann. Vol":    round(r.std(ddof=1) * np.sqrt(12), 4),
        "Sharpe":      round(sharpe_ratio(r), 3),
        "Win Rate":    round((r > 0).mean(), 3),
        "Best Month":  round(r.max(), 4),
        "Worst Month": round(r.min(), 4),
        "N months":    len(r),
    }


def subset(df, start, end):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return df.loc[(df.index >= s) & (df.index <= e)]

def verify_xsmom(xsmom: pd.DataFrame, monthly_ret: pd.DataFrame):
    """
    Verificerer at XSMOM er korrekt implementeret ved at tjekke:
    1. Unit leverage: positive vægte summer til ~1, negative til ~-1
    2. Cross-sectional demean: vægte summer til ~0 hver måned
    3. Signal retning: høj momentum → positiv vægt
    """
    print("\n" + "="*55)
    print("XSMOM SIGNAL VERIFIKATION")
    print("="*55)

    # Tjek et specifikt eksempel: januar 2005
    test_date = pd.Timestamp("2014-06-01")
    # Find nærmeste dato
    available = xsmom.index[xsmom.index >= test_date]
    if len(available) == 0:
        print("Ingen dato fundet")
        return
    test_date = available[0]

    signal = xsmom.loc[test_date].dropna()
    print(f"\nTestdato: {test_date.strftime('%Y-%m')}")
    print(f"Antal industrier med signal: {len(signal)}")

    # Check 1: Summer til 0 (cross-sectional neutral)
    print(f"  Sum af alle vægte: {signal.sum():.6f}  (skal være ~0)")

    # Check 2: Unit leverage
    pos = signal[signal > 0].sum()
    neg = signal[signal < 0].sum()
    print(f"  Sum positive vægte: {pos:.4f}  (skal være ~1.0)")
    print(f"  Sum negative vægte: {neg:.4f}  (skal være ~-1.0)")

    # Check 3: Retning — høj 12m afkast → positiv vægt
    # Beregn 12m afkast for testdato (window slutter ved t-1)
    idx = monthly_ret.index.get_loc(test_date)
    if idx >= 12:
        cum_ret = monthly_ret.iloc[idx-12:idx].sum()
        top5_ret = cum_ret.nlargest(5)
        bot5_ret = cum_ret.nsmallest(5)
        print(f"  Signalretning for top 5 industrier → bedste afkast sidste 12 måneder:")
        for ind in top5_ret.index:
            if ind in signal.index:
                print(f"    {ind:20s}: 12m afkast = {top5_ret[ind]:+.1f}%,"
                      f" signal={signal[ind]:+.4f} "
                      if signal[ind] > 0 else
                      f"    {ind:20s}: 12m afkast = {top5_ret[ind]:+.1f}%,"
                      f" signal={signal[ind]:+.4f} ")
        print(f"  Signalretning for dårligste 5 industrier → dårligste afkast sidste 12 måneder:")
        for ind in bot5_ret.index:
            if ind in signal.index:
                print(f"    {ind:20s}: 12m afkast = {bot5_ret[ind]:+.1f}%,"
                      f" signal={signal[ind]:+.4f} "
                      if signal[ind] < 0 else
                      f"    {ind:20s}: 12m afkast = {bot5_ret[ind]:+.1f}%,"
                      f" signal={signal[ind]:+.4f} ")

    # Check 4: Statistik over hele OOS periode
    oos = xsmom.loc["1942-01-01":"2018-12-31"]
    pos_sum = oos.clip(lower=0).sum(axis=1)
    neg_sum = oos.clip(upper=0).sum(axis=1)
    print(f"\nCheck 4 — Gennemsnitlig leverage over OOS periode:")
    print(f"  Gns. sum positive vægte: {pos_sum.mean():.4f}  (skal være ~1.0)")
    print(f"  Gns. sum negative vægte: {neg_sum.mean():.4f}  (skal være ~-1.0)")
    print(f"  Gns. sum alle vægte:     "
          f"{(pos_sum + neg_sum).mean():.6f}  (skal være ~0)")

    # Check 5: Antal aktive industrier per måned
    n_active = xsmom.notna().sum(axis=1)
    n_long   = (xsmom > 0).sum(axis=1)
    n_short  = (xsmom < 0).sum(axis=1)
    print(f"\nCheck 5 — Aktive industrier (gns. over OOS):")
    print(f"  Gns. antal industrier i signal: {n_active.mean():.1f}")
    print(f"  Gns. antal long:  {n_long.mean():.1f}")
    print(f"  Gns. antal short: {n_short.mean():.1f}")

    print("\n" + "="*55)


#Main
def main():
    print("\n" + "=" * 65)
    print("EPO EQUITY 1  —  Pedersen, Babu & Levine (2021)")
    print(f"Data:     {DATA_START_DATE} →")
    print(f"Backtest: {BACKTEST_START_DATE} → {BACKTEST_END_DATE}")
    print("=" * 65 + "\n")

    #Data indsamling
    monthly_ret = get_monthly_return()
    rf = get_monthly_risk()

    #månedlig merafkast
    monthly_excess = calculate_monthly_excess_returns(monthly_ret, rf)

    #XSMOM signal  (uses raw % returns for the cumulative sum, then excess returns are only needed for the covariance estimation)
    xsmom = compute_xsmom(monthly_ret, LOOKBACK_MONTHS)
    verify_xsmom(xsmom, monthly_ret)

    #risikomodel
    print("\nBuilding 60-month rolling risk model (5% pre-shrinkage) …")
    corr_shrunk, vols = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=CORR_PRESHRINK, verbose=True)

    # Unshrunk risk model for naive MVO benchmark (θ=0, w=0)
    print("Building unshrunk risk model for MVO benchmark …")
    corr_raw, _ = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=0.0, verbose=False)

    #Benchmark
    print("\nBacktesting benchmarks …")
    ew_full     = backtest_equal_weight(monthly_excess)
    indmom_full = backtest_indmom(monthly_excess, xsmom)
    mvo_full    = backtest_mvo_no_shrink(
        monthly_excess, xsmom, corr_raw, vols, GAMMA)

    # EPO for mulige vægte
    print("\nBuilding EPO panel for all candidate w values …")
    epo_panel = build_epo_panel(
        monthly_excess, xsmom, corr_shrunk, vols, GAMMA, CANDIDATE_WS)
    print(f"  Panel shape: {epo_panel.shape}")

    #Dynamisk EPO
    print("\nBuilding dynamic OOS EPO …")
    epo_dyn = build_dynamic_oos_epo(
        epo_panel, oos_start=BACKTEST_START_DATE, min_history=MIN_HISTORY_OOS)
    print(f"  Dynamic OOS EPO observations: {len(epo_dyn)}")

    # Begrænsning til OOS periode
    s, e = BACKTEST_START_DATE, BACKTEST_END_DATE

    ew_oos     = subset(ew_full,     s, e)
    indmom_oos = subset(indmom_full, s, e)
    mvo_oos    = subset(mvo_full,    s, e)
    panel_oos  = subset(epo_panel,   s, e)
    dyn_oos    = subset(epo_dyn,     s, e)

    print(f"\n  Dynamic OOS EPO OOS observations: {len(dyn_oos)}")

    # Performance
    rows = [
        performance_summary(ew_oos,     "1/N"),
        performance_summary(indmom_oos, "INDMOM"),
        performance_summary(mvo_oos,    "MVO (no shrinkage)"),
        performance_summary(dyn_oos,    "EPO: out-of-sample"),
    ]

    w_labels = {
        0.00: "EPO w=0%  (MVO + 5% pre-shrink)",
        0.10: "EPO w=10%",
        0.25: "EPO w=25%",
        0.50: "EPO w=50%",
        0.75: "EPO w=75%",
        0.90: "EPO w=90%",
        0.99: "EPO w=99%",
        1.00: "EPO w=100%  (anchor = INDMOM)",
    }
    for w in CANDIDATE_WS:
        col = f"EPO_w_{w:.2f}"
        if col in panel_oos.columns:
            rows.append(performance_summary(panel_oos[col], w_labels[w]))

    perf = pd.DataFrame(rows).set_index("Strategy")

    print("\n" + "=" * 65)
    print(f"PERFORMANCE — OOS: {BACKTEST_START_DATE} → {BACKTEST_END_DATE}")
    print("=" * 65)
    print(perf.to_string())

    # Sharpe-ratio summary (Table 5 format)
    order = [
        "1/N",
        "INDMOM",
        "MVO (no shrinkage)",
        "EPO: out-of-sample",
        "EPO w=0%  (MVO + 5% pre-shrink)",
        "EPO w=10%", "EPO w=25%", "EPO w=50%",
        "EPO w=75%", "EPO w=90%", "EPO w=99%",
        "EPO w=100%  (anchor = INDMOM)",
    ]
    sharpe_tbl = (perf["Sharpe"]
                  .reindex([r for r in order if r in perf.index])
                  .rename("Equity 1")
                  .to_frame())

    print("\nSharpe ratio table (Table 5 format):")
    print(sharpe_tbl.to_string())

    return {
        "monthly_excess": monthly_excess,
        "xsmom":          xsmom,
        "corr_shrunk":    corr_shrunk,
        "vols":           vols,
        "epo_panel":      epo_panel,
        "epo_dyn_oos":    dyn_oos,
        "performance":    perf,
        "sharpe_table":   sharpe_tbl,
    }

if __name__ == "__main__":
    results = main()