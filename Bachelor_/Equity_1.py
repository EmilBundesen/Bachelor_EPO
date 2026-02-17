import pandas as pd
import numpy as np

from rf_monthly import (
    get_file_path,
    load_and_clean_industry_data,
    load_and_clean_rf_data,
    PERCENT_TO_DECIMAL
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
DATA_START_DATE     = "1927-01-01"
BACKTEST_START_DATE = "1942-01-01"
BACKTEST_END_DATE   = "2018-12-31"

LOOKBACK_PERIOD_MONTHS = 12   # 12-month return signal (XSMOM)
RISK_WINDOW_MONTHS     = 60   # 60-month equal-weighted risk model (Table 1, Equity 1)
GAMMA                  = 3.0  # Risk aversion (cancels in Sharpe)
MIN_VOLATILITY         = 0.001

# 5% pre-shrinkage toward identity for the correlation matrix (Equity 1: "5% shrunk")
CORR_PRESHRINK = 0.05

CANDIDATE_WS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0]

# Minimum history for OOS ω-selection (15 years = 180 months)
MIN_HISTORY_FOR_OOS = 180


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def convert_daily_to_monthly_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    daily_decimal = daily_returns / PERCENT_TO_DECIMAL

    # Beregn månedlige afkast
    monthly = (1 + daily_decimal).resample('ME').prod() - 1

    # Hvis ALLE daglige værdier i måneden er NaN → sæt månedens afkast til NaN
    all_nan = daily_decimal.resample('ME').apply(lambda x: x.isna().all())
    monthly[all_nan] = np.nan

    return monthly



def calculate_monthly_excess_returns(monthly_returns: pd.DataFrame,
                                     rf_monthly: pd.DataFrame) -> pd.DataFrame:
    # Align indices to month-end timestamps
    mr = monthly_returns.copy()
    mr.index = mr.index.to_period('M').to_timestamp('M')

    rf = rf_monthly.copy()
    rf.index = rf.index.to_period('M').to_timestamp('M')

    # Find common dates
    common = mr.index.intersection(rf.index)

    # Compute excess returns WITHOUT filling missing data with 0
    excess = mr.loc[common].subtract(rf.loc[common, "RF"], axis=0)

    # Mask missing monthly returns → NaN instead of 0 - rf
    excess = excess.where(mr.loc[common].notna())

    return excess



def subset(df, start, end):
    return df.loc[(df.index >= pd.to_datetime(start)) &
                  (df.index <= pd.to_datetime(end))]


# ---------------------------------------------------------------------------
# XSMOM signal  —  Equations 24–25 in the paper
# ---------------------------------------------------------------------------

def calculate_xsmom_signal(monthly_returns: pd.DataFrame,
                           lookback_months: int = LOOKBACK_PERIOD_MONTHS
                           ) -> pd.DataFrame:
    # Ensure missing data stays NaN
    monthly_returns = monthly_returns.where(monthly_returns.notna())

    # Rolling 12-month sum, shifted by 1 so we never use current-month return
    rolling = monthly_returns.rolling(window=lookback_months).sum().shift(1)

    # Demeaned across industries at each date
    raw = rolling.subtract(rolling.mean(axis=1), axis=0)

    out = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns)

    for date in raw.index:
        row = raw.loc[date].dropna()
        if len(row) == 0:
            continue
        pos_sum = row[row > 0].sum()
        neg_sum = row[row < 0].sum()
        if pos_sum == 0 or neg_sum == 0:
            continue
        c_t = 1.0 / max(pos_sum, abs(neg_sum))
        out.loc[date, row.index] = c_t * row

    return out


# ---------------------------------------------------------------------------
# 60-month equal-weighted risk model  —  Table 1, Equity 1
# ---------------------------------------------------------------------------

def calculate_monthly_risk_model(monthly_excess: pd.DataFrame,
                                 window_months: int = RISK_WINDOW_MONTHS,
                                 corr_preshrink: float = CORR_PRESHRINK,
                                 verbose: bool = True):
    """
    Returns:
        corr_dict[date] : 5%-pre-shrunk correlation matrix (toward identity, simple EPO)
        vols_dict[date] : per-industry volatility (monthly std, ddof=1)
    """
    idx  = monthly_excess.index
    cols = monthly_excess.columns
    data = monthly_excess.values.astype(np.float64)

    corr_dict = {}
    vols_dict = {}

    n_total     = len(idx)
    total_iters = n_total - window_months
    report_every = max(1, total_iters // 20)

    for i in range(window_months, n_total):
        if verbose and (i - window_months) % report_every == 0:
            pct = 100.0 * (i - window_months) / total_iters
            print(f"  Risk model: {pct:5.1f}%  "
                  f"(step {i - window_months}/{total_iters})", flush=True)

        date   = idx[i]
        window = data[i - window_months: i]

        valid_mask = ~np.all(np.isnan(window), axis=0)
        if not np.any(valid_mask):
            continue
        w = window[:, valid_mask]
        c = cols[valid_mask]

        nobs_ok = np.sum(~np.isnan(w), axis=0) >= 2
        if not np.any(nobs_ok):
            continue
        w = w[:, nobs_ok]
        c = c[nobs_ok]

        vol = np.nanstd(w, axis=0, ddof=1)
        nonzero = vol > 0
        if not np.any(nonzero):
            continue
        w   = w[:, nonzero]
        vol = vol[nonzero]
        c   = c[nonzero]

        # NaN-safe sample correlation matrix Ω_sample
        col_means = np.nanmean(w, axis=0)
        X_dm = w - col_means
        X0   = np.where(np.isnan(X_dm), 0.0, X_dm)
        cov_pw = X0.T @ X0
        ss     = (X0 ** 2).sum(axis=0)
        denom  = np.sqrt(np.outer(ss, ss))
        with np.errstate(invalid='ignore', divide='ignore'):
            corr_sample = np.where(denom > 0, cov_pw / denom, 0.0)
        np.fill_diagonal(corr_sample, 1.0)

        # ------------------------------------------------------------------
        # Simple EPO-style correlation shrinkage:
        #   Ω_shrunk = (1 - λ) Ω_sample + λ I
        # i.e. off-diagonals scaled by (1 - λ), diagonals stay at 1.
        # ------------------------------------------------------------------
        lam = corr_preshrink  # 0.05 for Equity 1
        n = corr_sample.shape[0]
        I = np.eye(n)
        corr = (1.0 - lam) * corr_sample + lam * I

        corr_dict[date] = pd.DataFrame(corr, index=c, columns=c)
        vols_dict[date] = pd.Series(vol, index=c)

    if verbose:
        print("  Risk model: 100.0% — done.", flush=True)

    return corr_dict, vols_dict




# ---------------------------------------------------------------------------
# EPO weights  —  Simple EPO, Equations 19–20
# ---------------------------------------------------------------------------

def calculate_epo_weights(signal: pd.Series,
                          corr_matrix: pd.DataFrame,
                          volatilities: pd.Series,
                          gamma: float,
                          w: float,
                          min_vol: float = MIN_VOLATILITY) -> pd.Series:
    # Align assets
    common = (signal.dropna().index
                     .intersection(corr_matrix.index)
                     .intersection(volatilities.index))
    if len(common) == 0:
        return pd.Series(dtype=float)

    s   = signal.loc[common]
    C   = corr_matrix.loc[common, common]
    vol = volatilities.loc[common]

    # Drop near-zero vol assets
    ok  = vol >= min_vol
    s   = s[ok];  vol = vol[ok];  C = C.loc[ok, ok]
    if len(s) == 0:
        return pd.Series(dtype=float)

    vol_v = vol.values
    s_v   = s.values

    # Σ = D Ω D
    Sigma = C.values * np.outer(vol_v, vol_v)

    # diag(Σ)
    Sigma_d = np.diag(np.diag(Sigma))

    # Equation (19): Σ_w = (1 - w) Σ + w diag(Σ)
    Sigma_w = (1.0 - w) * Sigma + w * Sigma_d

    try:
        Sigma_inv = np.linalg.inv(Sigma_w)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma_w)

    # Equation (20): x_raw = (1/γ) Σ_w^{-1} s
    x_raw = (1.0 / gamma) * (Sigma_inv @ s_v)
    x_raw_s = pd.Series(x_raw, index=s.index)

    # Unit-leverage normalisation (same convention as XSMOM / INDMOM)
    pos_sum = x_raw_s[x_raw_s > 0].sum()
    neg_sum = x_raw_s[x_raw_s < 0].sum()
    if pos_sum == 0 or neg_sum == 0:
        return pd.Series(dtype=float)

    c = 1.0 / max(pos_sum, abs(neg_sum))
    return c * x_raw_s


# ---------------------------------------------------------------------------
# EPO backtest for a single fixed w
# ---------------------------------------------------------------------------

def backtest_epo_fixed_w(monthly_excess: pd.DataFrame,
                         signals: pd.DataFrame,
                         corr_dict: dict,
                         vols_dict: dict,
                         gamma: float,
                         w: float) -> pd.Series:
    idx        = monthly_excess.index
    risk_dates = set(corr_dict.keys())
    sig_dates  = set(signals.index)
    rets, dates = [], []

    for t, date in enumerate(idx[:-1]):
        if date not in risk_dates or date not in sig_dates:
            continue

        weights = calculate_epo_weights(
            signals.loc[date], corr_dict[date], vols_dict[date], gamma, w
        )
        if len(weights) == 0:
            continue

        next_date = idx[t + 1]
        r = monthly_excess.loc[next_date, weights.index]
        rets.append((weights.values * r.values).sum())
        dates.append(next_date)

    return pd.Series(rets, index=dates, name=f"EPO_w_{w:.2f}")


# ---------------------------------------------------------------------------
# INDMOM benchmark  (Moskowitz & Grinblatt 1999)
# ---------------------------------------------------------------------------

def calculate_indmom_benchmark(monthly_excess: pd.DataFrame,
                               signals: pd.DataFrame) -> pd.Series:
    idx       = monthly_excess.index
    sig_dates = set(signals.index)
    rets, dates = [], []

    for t, date in enumerate(idx[:-1]):
        if date not in sig_dates:
            continue
        s = signals.loc[date].dropna()
        if len(s) == 0:
            continue
        next_date = idx[t + 1]
        common = s.index.intersection(monthly_excess.columns)
        r = monthly_excess.loc[next_date, common]
        rets.append((s.loc[common].values * r.values).sum())
        dates.append(next_date)

    return pd.Series(rets, index=dates, name="INDMOM")


# ---------------------------------------------------------------------------
# 1/N benchmark  (equal-weight, forward return)
# ---------------------------------------------------------------------------

def backtest_equal_weight(monthly_excess: pd.DataFrame) -> pd.Series:
    fwd = monthly_excess.shift(-1)
    ew_ret = fwd.mean(axis=1).iloc[:-1]
    ew_ret.name = "EW_1N"
    return ew_ret


# ---------------------------------------------------------------------------
# MVO benchmark (no correlation shrinkage at all)
# ---------------------------------------------------------------------------

def backtest_mvo_no_shrink(monthly_excess: pd.DataFrame,
                           signals: pd.DataFrame,
                           corr_dict_raw: dict,
                           vols_dict: dict,
                           gamma: float) -> pd.Series:
    idx        = monthly_excess.index
    risk_dates = set(corr_dict_raw.keys())
    sig_dates  = set(signals.index)
    rets, dates = [], []

    for t, date in enumerate(idx[:-1]):
        if date not in risk_dates or date not in sig_dates:
            continue

        corr = corr_dict_raw[date]
        vol  = vols_dict[date]
        s    = signals.loc[date].dropna()

        common = s.index.intersection(corr.index).intersection(vol.index)
        if len(common) == 0:
            continue

        C   = corr.loc[common, common].values
        v   = vol.loc[common].values
        s_v = s.loc[common].values

        Sigma = C * np.outer(v, v)

        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        w_raw = (1.0 / gamma) * (Sigma_inv @ s_v)
        w_raw_s = pd.Series(w_raw, index=common)

        pos_sum = w_raw_s[w_raw_s > 0].sum()
        neg_sum = w_raw_s[w_raw_s < 0].sum()
        if pos_sum == 0 or neg_sum == 0:
            continue
        c = 1.0 / max(pos_sum, abs(neg_sum))
        w_norm = c * w_raw_s

        next_date = idx[t + 1]
        r = monthly_excess.loc[next_date, common]
        rets.append((w_norm.values * r.values).sum())
        dates.append(next_date)

    return pd.Series(rets, index=dates, name="MVO_no_shrink")


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def gross_sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    s = returns.std()
    return 0.0 if s == 0 else (returns.mean() / s) * np.sqrt(12)


def calculate_performance_metrics(returns: pd.Series, name: str) -> dict:
    returns = returns.dropna()
    if len(returns) == 0:
        return {k: np.nan for k in
                ["Strategy", "Total Return", "Annualized Return",
                 "Annualized Vol", "Sharpe", "Win Rate",
                 "Best Month", "Worst Month"]}
    cum   = (1 + returns).prod() - 1
    n_yrs = len(returns) / 12
    ann_r = (1 + cum) ** (1 / n_yrs) - 1
    vol_a = returns.std() * np.sqrt(12)
    return {
        "Strategy":          name,
        "Total Return":      cum,
        "Annualized Return": ann_r,
        "Annualized Vol":    vol_a,
        "Sharpe":            gross_sharpe(returns),
        "Win Rate":          (returns > 0).mean(),
        "Best Month":        returns.max(),
        "Worst Month":       returns.min(),
    }


# ---------------------------------------------------------------------------
# Build EPO panel (all candidate w values, full sample)
# ---------------------------------------------------------------------------

def build_epo_panel(monthly_excess: pd.DataFrame,
                    signals: pd.DataFrame,
                    corr_dict: dict,
                    vols_dict: dict,
                    gamma: float,
                    candidate_ws: list) -> pd.DataFrame:
    series_list = []
    for w in candidate_ws:
        print(f"  Running EPO for w={w:.2f} …")
        s = backtest_epo_fixed_w(
            monthly_excess, signals, corr_dict, vols_dict, gamma, w
        )
        series_list.append(s)
    return pd.concat(series_list, axis=1).sort_index()


# ---------------------------------------------------------------------------
# Out-of-sample (dynamic) EPO  —  paper's primary OOS EPO strategy
# ---------------------------------------------------------------------------

def build_dynamic_epo(panel_epo: pd.DataFrame,
                      oos_start: str,
                      min_history: int = MIN_HISTORY_FOR_OOS) -> pd.Series:
    panel = panel_epo.dropna(how="all").sort_index()

    cols      = panel.columns
    dyn_rets  = []
    dyn_dates = []

    t0_oos = pd.to_datetime(oos_start)

    for i in range(1, len(panel)):
        date = panel.index[i]

        # Only record returns from the OOS start onwards
        if date < t0_oos:
            continue

        past = panel.iloc[:i]   # all history available before date t

        if len(past) < min_history:
            continue

        best_w  = None
        best_sr = -np.inf

        for c in cols:
            r = past[c].dropna()
            if len(r) < min_history:
                continue
            sr = gross_sharpe(r)
            if sr > best_sr:
                best_sr = sr
                best_w  = c

        if best_w is None:
            continue

        dyn_rets.append(panel.loc[date, best_w])
        dyn_dates.append(date)

    return pd.Series(dyn_rets, index=dyn_dates, name="EPO_dynamic_OOS")


# ---------------------------------------------------------------------------
# Raw (no pre-shrinkage) risk model — used only for the MVO benchmark
# ---------------------------------------------------------------------------

def calculate_monthly_risk_model_raw(monthly_excess: pd.DataFrame,
                                     window_months: int = RISK_WINDOW_MONTHS,
                                     verbose: bool = False):
    """Identical to calculate_monthly_risk_model but with corr_preshrink=0."""
    return calculate_monthly_risk_model(
        monthly_excess,
        window_months=window_months,
        corr_preshrink=0.0,
        verbose=verbose
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 80)
    print("EPO INDUSTRY MOMENTUM — EQUITY 1 (Pedersen, Babu & Levine 2021)")
    print("=" * 80 + "\n")

    # 1. Load data
    print("Loading data …")
    industry_file = get_file_path("Select daily industry returns CSV:")
    rf_file       = get_file_path("Select monthly risk-free rate CSV:")

    daily_returns = load_and_clean_industry_data(
        industry_file, DATA_START_DATE, BACKTEST_END_DATE
    )
    rf_monthly = load_and_clean_rf_data(
        rf_file, DATA_START_DATE, BACKTEST_END_DATE
    )
    monthly_returns = convert_daily_to_monthly_returns(daily_returns)
    monthly_excess  = calculate_monthly_excess_returns(monthly_returns, rf_monthly)

    # XSMOM signal from total monthly returns
    xsmom_signals = calculate_xsmom_signal(monthly_returns, LOOKBACK_PERIOD_MONTHS)

    # 2. Risk models
    print("\nBuilding 60-month risk model with 5% correlation pre-shrinkage …")
    corr_shrunk, vols = calculate_monthly_risk_model(
        monthly_excess,
        window_months=RISK_WINDOW_MONTHS,
        corr_preshrink=CORR_PRESHRINK,
        verbose=True
    )
    print(f"  → {len(corr_shrunk)} observations in risk-model cache.\n")

    print("Building unshrunk risk model for MVO benchmark …")
    corr_raw, vols_raw = calculate_monthly_risk_model_raw(
        monthly_excess,
        window_months=RISK_WINDOW_MONTHS,
        verbose=False
    )

    # 3. Period boundaries
    in_start, in_end   = DATA_START_DATE,     "1941-12-31"
    oos_start, oos_end = BACKTEST_START_DATE, BACKTEST_END_DATE
    t0_oos = pd.to_datetime(oos_start)
    t1_oos = pd.to_datetime(oos_end)

    # 4. Benchmarks (OOS window only)
    print("Backtesting benchmarks …")
    eqw_oos    = subset(backtest_equal_weight(monthly_excess), oos_start, oos_end)
    indmom_oos = subset(
        calculate_indmom_benchmark(monthly_excess, xsmom_signals),
        oos_start, oos_end
    )

    corr_raw_oos = {d: v for d, v in corr_raw.items() if t0_oos <= d <= t1_oos}
    vols_oos     = {d: v for d, v in vols.items()     if t0_oos <= d <= t1_oos}
    mvo_oos = subset(
        backtest_mvo_no_shrink(
            monthly_excess, xsmom_signals, corr_raw_oos, vols_oos, GAMMA
        ),
        oos_start, oos_end
    )

    # 5. EPO panel over the full sample (1927–2018)
    print("\nBuilding EPO panel for all candidate w values (full sample) …")
    epo_panel = build_epo_panel(
        monthly_excess, xsmom_signals, corr_shrunk, vols,
        GAMMA, CANDIDATE_WS
    )

    # 6. In-sample ω* selection (1927–1941)
    print("\nSelecting omega* from in-sample period (1927–1941) …")
    epo_panel_in = subset(epo_panel, in_start, in_end)
    in_sample_sharpes = {}
    for w in CANDIDATE_WS:
        col = f"EPO_w_{w:.2f}"
        if col not in epo_panel_in.columns:
            in_sample_sharpes[w] = np.nan
            continue
        sr = gross_sharpe(epo_panel_in[col])
        in_sample_sharpes[w] = sr
        print(f"  w={w:>4.2f}  in-sample Sharpe = {sr:.3f}")

    w_star = max(in_sample_sharpes, key=lambda x: in_sample_sharpes[x])
    print(f"\n  Selected omega* = {w_star:.2f}  (in-sample best)\n")

    # 7. OOS EPO strategies
    epo_panel_oos = subset(epo_panel, oos_start, oos_end)

    # (a) Fixed omega* chosen once in-sample
    epo_star_oos = epo_panel_oos[f"EPO_w_{w_star:.2f}"].rename(
        f"EPO_w*_{w_star:.2f}"
    )

    # (b) Dynamic OOS EPO: expanding-window ω selection
    print("Building dynamic OOS EPO (expanding-window w selection, ≥180 months) …")
    epo_dyn_oos = build_dynamic_epo(
        epo_panel, oos_start=oos_start, min_history=MIN_HISTORY_FOR_OOS
    )
    epo_dyn_oos = subset(epo_dyn_oos, oos_start, oos_end)

    # 9. Full performance table
    all_series = {
        "1/N":                  eqw_oos,
        "INDMOM":               indmom_oos,
        "MVO (no corr.)":       mvo_oos,
        f"EPO_w*_{w_star:.2f}": epo_star_oos,
        "EPO: out-of-sample":   epo_dyn_oos,
    }

    # Add selected w grid points matching the paper’s table
    for w in [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]:
        col = f"EPO_w_{w:.2f}"
        if col in epo_panel_oos.columns:
            all_series[col] = epo_panel_oos[col]

    rows = [calculate_performance_metrics(s, name)
            for name, s in all_series.items()]
    perf_df = pd.DataFrame(rows).set_index("Strategy")

    print("\nFull performance summary (1942–2018):")
    print(perf_df.round(4))

    # 10. Equity 1-style Sharpe table
    equity1 = perf_df["Sharpe"].rename("Equity 1").to_frame()
    # Reorder rows to match the paper’s table
    desired_order = [
        "1/N",
        "INDMOM",
        "MVO (no corr.)",
        "EPO: out-of-sample",
        "EPO_w_0.00",
        "EPO_w_0.10",
        "EPO_w_0.25",
        "EPO_w_0.50",
        "EPO_w_0.75",
        "EPO_w_0.90",
        "EPO_w_0.99",
        "EPO_w_1.00",
    ]
    equity1 = equity1.reindex([r for r in desired_order if r in equity1.index])

    print("\nEquity 1-style Sharpe table (paper format):")
    print(equity1.round(2))

    return {
        "epo_panel":         epo_panel,
        "epo_star_oos":      epo_star_oos,
        "epo_dynamic_oos":   epo_dyn_oos,
        "in_sample_sharpes": in_sample_sharpes,
        "w_star":            w_star,
        "performance":       perf_df,
        "equity1_table":     equity1,
    }


if __name__ == "__main__":
    results = main()
