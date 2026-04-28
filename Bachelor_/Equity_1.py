"""
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
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Konstanter
DATA_START_DATE     = "2010-01-01"
BACKTEST_START_DATE = "2020-01-01"
BACKTEST_END_DATE   = "2025-12-31"

LOOKBACK_MONTHS  = 12    # XSMOM signal lookback (Eq. 24)
RISK_WINDOW      = 24    # Rolling window for covariance estimation (months)
CORR_PRESHRINK   = 0.05  # θ: 5% pre-shrinkage toward I (Table 1, Equity 1)
GAMMA            = 3     # Risk aversion γ (cancels in Sharpe ratio)
MIN_VOL          = 1e-8  # Floor on vol to avoid division by zero

CANDIDATE_WS     = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]
MIN_HISTORY_OOS  = 6
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

def compute_tsmom_signal(monthly_ret: pd.DataFrame,
                         lookback: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """
    TSMOM signal skaleret til unit-leverage (samme interface som compute_xsmom).

    For hver dato t:
      - Long:  12m afkast > 0  → signal = +|r_i|
      - Short: 12m afkast < 0  → signal = -|r_i|
      - Nul:   12m afkast = 0  → signal =  0

    Normalisering: c_t = 1 / max(sum_long, sum_short)
    så max(brutto_long, brutto_short) = 1  (unit-leverage, Eq. 25-analog).
    """
    roll = (monthly_ret
            .rolling(window=lookback, min_periods=lookback)
            .sum())

    out = pd.DataFrame(np.nan, index=roll.index, columns=roll.columns)

    for date, row in roll.iterrows():
        avail = row.dropna()
        if len(avail) < 2:
            continue

        signal = pd.Series(0.0, index=avail.index)
        signal[avail > 0] =  avail[avail > 0]   # long:  positivt afkast
        signal[avail < 0] =  avail[avail < 0]   # short: negativt afkast

        pos = signal[signal > 0].sum()
        neg = signal[signal < 0].sum()

        # Kræv begge sider for at undgå rent long/short univers
        if pos == 0 or neg == 0:
            continue

        c_t = 1.0 / max(pos, abs(neg))          # unit-leverage normalisering
        out.loc[date, signal.index] = c_t * signal

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
            print(f"  Risk model: {pct:.1f}%", end="\r", flush=True)

        date = idx[i]
        window_data = data[i - window: i]

        # Kræv præcis fuld historik — ingen 90%-fallback, ingen row-fjernelse
        n_obs = np.sum(~np.isnan(window_data), axis=0)
        valid = n_obs == window

        if valid.sum() < 2:
            continue

        W = window_data[:, valid]
        c = cols[valid]

        # W er per konstruktion NaN-fri — ingen row_complete-fjernelse
        K = W.shape[0]  # altid == window

        # Artiklens kovariansestimator (footnote 17)
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

    # Tjek et specifikt eksempel
    test_date = pd.Timestamp("2025-06-01")
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


# Vægte pr. 2022-12-31:
def get_weights_at_date(
        monthly_excess: pd.DataFrame,
        signals:        pd.DataFrame,
        corr_dict:      dict,
        vols_dict:      dict,
        epo_panel:      pd.DataFrame,
        gamma:          float,
        target_date:    str,
        top_n:          int = 10
) -> pd.DataFrame:
    """
    Finder EPO-vægte pr. target_date med dynamisk w* (argmax Sharpe).
    Returnerer top_n største positive og top_n største negative vægte.
    """
    t = pd.to_datetime(target_date)

    # Find den seneste tilgængelige signal-dato <= target_date
    valid_dates = monthly_excess.index[monthly_excess.index <= t]
    if len(valid_dates) == 0:
        raise ValueError(f"Ingen data fundet før {target_date}")
    signal_date = valid_dates[-1]
    print(f"  Signal-dato:  {signal_date.strftime('%Y-%m')}")

    # Vælg w* = argmax Sharpe over AL tilgængelig panel-historik til og med signal_date
    past = epo_panel.loc[epo_panel.index <= signal_date]
    if len(past) < MIN_HISTORY_OOS:
        raise ValueError(f"For lidt historik til at vælge w* ({len(past)} mdr.)")

    best_w, best_sr = None, -np.inf
    for col in past.columns:
        r = past[col].dropna()
        if len(r) < MIN_HISTORY_OOS:
            continue
        sr = sharpe_ratio(r)
        if sr > best_sr:
            best_sr = sr
            best_w  = col

    # Udtræk den numeriske w-værdi fra kolonnenavnet, fx "EPO_w_0.50" → 0.50
    w_val = float(best_w.replace("EPO_w_", ""))
    print(f"  Valgt w*:     {w_val:.2f}  (Sharpe = {best_sr:.3f})")

    # Tjek at risiko og signal eksisterer for signal_date
    if signal_date not in corr_dict:
        raise ValueError(f"Ingen risikomodel for {signal_date.strftime('%Y-%m')}")
    if signal_date not in signals.index:
        raise ValueError(f"Intet XSMOM-signal for {signal_date.strftime('%Y-%m')}")

    # Beregn EPO-vægte
    wts = epo_weights(
        signals.loc[signal_date],
        corr_dict[signal_date],
        vols_dict[signal_date],
        gamma, w_val
    )
    if len(wts) == 0:
        raise ValueError("EPO returnerede ingen vægte — tjek signal og risikomodel")

    # Top N positive og top N negative (efter absolut størrelse)
    pos = wts[wts > 0].nlargest(top_n)
    neg = wts[wts < 0].nsmallest(top_n)   # nsmallest = mest negative

    pos_df = pos.rename("Weight").to_frame()
    pos_df["Side"] = "Long"
    pos_df["Rank"] = [f"Long {i+1}" for i in range(len(pos_df))]

    neg_df = neg.rename("Weight").to_frame()
    neg_df["Side"] = "Short"
    neg_df["Rank"] = [f"Short {i+1}" for i in range(len(neg_df))]

    result = pd.concat([pos_df, neg_df])
    result["Weight_%"] = (result["Weight"] * 100).round(2)
    return result[["Rank", "Side", "Weight_%"]]


def print_avg_epo_weights_year(monthly_excess, xsmom, corr_dict, vols_dict,
                                epo_panel, gamma,
                                start="2022-01-01", end="2022-12-31",
                                top_n=10) -> pd.DataFrame:
    """
    Beregner gennemsnitlige EPO-vægte (dynamisk w*) for hver måned i 2024
    og udskriver de top_n long og top_n short industrier rangeret efter
    gennemsnitlig vægt over perioden.
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    dates = [d for d in monthly_excess.index if s <= d <= e]

    weight_records = {}
    for date in dates:
        past = epo_panel.loc[epo_panel.index < date]
        if len(past) < MIN_HISTORY_OOS:
            continue

        best_w, best_sr = None, -np.inf
        for col in past.columns:
            r = past[col].dropna()
            if len(r) < MIN_HISTORY_OOS:
                continue
            sr = sharpe_ratio(r)
            if sr > best_sr:
                best_sr = sr
                best_w  = col

        if best_w is None:
            continue

        w_val = float(best_w.replace("EPO_w_", ""))

        if date not in corr_dict or date not in xsmom.index:
            continue

        wts = epo_weights(xsmom.loc[date], corr_dict[date],
                          vols_dict[date], gamma, w_val)
        if len(wts) > 0:
            weight_records[date] = wts

    if not weight_records:
        print("Ingen vægte beregnet for perioden.")
        return pd.DataFrame()

    weights_df = pd.DataFrame(weight_records).T.fillna(0)
    mean_wts   = weights_df.mean()

    top_long  = mean_wts.nlargest(top_n)
    top_short = mean_wts.nsmallest(top_n)

    print("\n" + "=" * 55)
    print(f"GENNEMSNITLIGE EPO-VÆGTE — {start[:4]}")
    print("=" * 55)
    print(f"\n  Top {top_n} long:")
    print(f"  {'Industri':<25} {'Gns. vægt':>10}")
    print("  " + "-" * 37)
    for ticker, val in top_long.items():
        print(f"  {ticker:<25} {val:>+10.4f}")

    print(f"\n  Top {top_n} short:")
    print(f"  {'Industri':<25} {'Gns. vægt':>10}")
    print("  " + "-" * 37)
    for ticker, val in top_short.items():
        print(f"  {ticker:<25} {val:>+10.4f}")
    print("=" * 55)

    return weights_df


def compute_elo_scores(rolling_weights: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Beregner ELO-inspireret score per industri baseret på:
      1) Antal gange industrien optræder i top-n long/short
      2) Hvilken rank de har fået (rank 1 = højest score)

    Scoren per observation = n - rank + 1
    (rank 1 → n point, rank 2 → n-1 point, ..., rank n → 1 point)

    Returnerer DataFrame med long_score, short_score, long_appearances,
    short_appearances sorteret efter long_score descending.
    """
    long_scores  = {}
    short_scores = {}
    long_app     = {}
    short_app    = {}

    for date, row in rolling_weights.iterrows():
        avail = row.dropna()
        if len(avail) < n * 2:
            continue

        top_n    = avail.nlargest(n)
        bottom_n = avail.nsmallest(n)

        for rank, ticker in enumerate(top_n.index, 1):
            score = n - rank + 1
            long_scores[ticker] = long_scores.get(ticker, 0) + score
            long_app[ticker]    = long_app.get(ticker, 0) + 1

        for rank, ticker in enumerate(bottom_n.index, 1):
            score = n - rank + 1
            short_scores[ticker] = short_scores.get(ticker, 0) + score
            short_app[ticker]    = short_app.get(ticker, 0) + 1

    all_tickers = set(long_scores) | set(short_scores)
    rows = []
    for ticker in all_tickers:
        rows.append({
            "Industri":          ticker,
            "Long score":        long_scores.get(ticker, 0),
            "Long optræden":     long_app.get(ticker, 0),
            "Short score":       short_scores.get(ticker, 0),
            "Short optræden":    short_app.get(ticker, 0),
        })

    df = (pd.DataFrame(rows)
            .set_index("Industri")
            .sort_values("Long score", ascending=False))

    return df

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
    xsmom = compute_xsmom(monthly_excess, LOOKBACK_MONTHS)
    verify_xsmom(xsmom, monthly_excess)


    #risikomodel
    print("\nBuilding 60-month rolling risk model (5% pre-shrinkage) …")
    corr_shrunk, vols = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=CORR_PRESHRINK, verbose=True)

    # Unshrunk risk model for naive MVO benchmark (θ=0, w=0)
    print("Building unshrunk risk model for MVO benchmark …")
    corr_raw, vols_raw = compute_risk_model(
        monthly_excess, window=RISK_WINDOW, theta=0.0, verbose=False)

    #Benchmark
    print("\nBacktesting benchmarks …")
    ew_full     = backtest_equal_weight(monthly_excess)
    indmom_full = backtest_indmom(monthly_excess, xsmom)
    mvo_full    = backtest_mvo_no_shrink(
        monthly_excess, xsmom, corr_raw, vols_raw, GAMMA)

    # EPO for mulige vægte
    print("\nBuilding EPO panel for all candidate w values …")
    epo_panel = build_epo_panel(
        monthly_excess, xsmom, corr_shrunk, vols, GAMMA, CANDIDATE_WS)
    print(f"  Panel shape: {epo_panel.shape}")


    # Her kan der findes vægte på bestemt dag ved funktionen get_weights_at_date
    result = get_weights_at_date(
        monthly_excess, xsmom, corr_shrunk, vols, epo_panel,
        gamma=GAMMA, target_date="2022-12-31", top_n=15
    )
    print(result)
    # Eller gennemsnitlige vægte ved funktionen print_avg_epo_weights_year()

    print_avg_epo_weights_year(monthly_excess, xsmom, corr_raw, vols_raw,
    epo_panel, GAMMA, start = "2022-01-01", end = "2022-12-31", top_n = 10)

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
        1.00: "EPO w=100%  (anchor)",
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
        "EPO w=100%  (anchor)",
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