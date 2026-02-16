import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rf_monthly import (
    get_file_path,
    load_and_clean_industry_data,
    load_and_clean_rf_data,
    PERCENT_TO_DECIMAL
)

#KONSTANTER
EPO_START_DATE = "1942-01-01"
EPO_END_DATE = "2018-12-31"

LOOKBACK_PERIOD_MONTHS = 12
RISK_WINDOW_MONTHS = 60
GAMMA = 3.0
SHRINKAGE_W = 1.0
THETA = 0.0
MIN_VOLATILITY = 0.01
SIGNAL_SCALING = 0.01

PLOT_WIDTH = 14
PLOT_HEIGHT = 7
LINE_WIDTH = 2


#Datarens

def convert_daily_to_monthly_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    daily_returns_decimal = daily_returns / PERCENT_TO_DECIMAL
    monthly_returns = (1 + daily_returns_decimal).resample('ME').prod() - 1
    return monthly_returns


def calculate_monthly_excess_returns(monthly_returns: pd.DataFrame,
                                     rf_monthly: pd.DataFrame) -> pd.DataFrame:
    monthly_returns_normalized = monthly_returns.copy()
    monthly_returns_normalized.index = monthly_returns_normalized.index.to_period('M').to_timestamp('M')

    rf_normalized = rf_monthly.copy()
    rf_normalized.index = rf_normalized.index.to_period('M').to_timestamp('M')

    common_dates = monthly_returns_normalized.index.intersection(rf_normalized.index)
    monthly_returns_aligned = monthly_returns_normalized.loc[common_dates]
    rf_aligned = rf_normalized.loc[common_dates]

    excess_returns = monthly_returns_aligned.subtract(rf_aligned["RF"], axis=0)
    return excess_returns


#XSMOM signal

def calculate_xsmom_signal(monthly_returns: pd.DataFrame,
                           lookback_months: int = 12) -> pd.DataFrame:

    rolling_returns = monthly_returns.rolling(window=lookback_months).sum().shift(1)
    cross_sectional_mean = rolling_returns.mean(axis=1)
    relative_performance = rolling_returns.subtract(cross_sectional_mean, axis=0)

    signals = pd.DataFrame(index=relative_performance.index,
                           columns=relative_performance.columns, dtype=float)

    for date in relative_performance.index:
        signal_row = relative_performance.loc[date].dropna()
        if len(signal_row) == 0:
            continue

        positive_sum = signal_row[signal_row > 0].sum()
        negative_sum = abs(signal_row[signal_row < 0].sum())

        c_t = 1 / max(positive_sum, negative_sum) if positive_sum > 0 and negative_sum > 0 else 0
        signals.loc[date, signal_row.index] = c_t * signal_row

    return signals


#Risikomodel

def calculate_rolling_correlation_and_volatility(monthly_excess_returns: pd.DataFrame,
                                                 window_months: int = 60,
                                                 theta: float = 0.0) -> tuple:
    correlation_matrices = {}
    volatilities = {}

    for i in range(window_months, len(monthly_excess_returns)): #rullende korrelation og votalitet
        date = monthly_excess_returns.index[i]
        window_data = monthly_excess_returns.iloc[i - window_months:i]
        window_data = window_data.dropna(axis=1, how='all')

        if len(window_data.columns) == 0:
            continue

        corr_matrix = window_data.corr()
        vol = window_data.std()

        if theta > 0:
            n = len(corr_matrix)
            identity = pd.DataFrame(np.eye(n), index=corr_matrix.index, columns=corr_matrix.columns)
            corr_matrix = (1 - theta) * corr_matrix + theta * identity

        correlation_matrices[date] = corr_matrix
        volatilities[date] = vol

    return correlation_matrices, volatilities


#EPO

def calculate_epo_weights(signal: pd.Series,
                          correlation_matrix: pd.DataFrame,
                          volatilities: pd.Series,
                          gamma: float,
                          w: float,
                          min_volatility: float = 0.01,
                          signal_scaling: float = 0.01) -> pd.Series:

    common_assets = signal.dropna().index.intersection(
        correlation_matrix.index).intersection(volatilities.index)

    if len(common_assets) == 0:
        return pd.Series(dtype=float)

    signal_aligned = signal.loc[common_assets]
    corr_aligned = correlation_matrix.loc[common_assets, common_assets]
    vol_aligned = volatilities.loc[common_assets]

    #Fjerner lav votalitet (ved missing data, lav votalitet -> ekstremt høje vægte)
    valid_vol_mask = vol_aligned >= min_volatility
    signal_aligned = signal_aligned[valid_vol_mask]
    vol_aligned = vol_aligned[valid_vol_mask]
    corr_aligned = corr_aligned.loc[valid_vol_mask, valid_vol_mask]

    if len(signal_aligned) == 0:
        return pd.Series(dtype=float)

    signal_scaled = signal_aligned * signal_scaling

    # For w = 1: EPO_s(w=1) = (1/γ) * s / σ²
    if w == 1.0:
        variance = vol_aligned ** 2
        weights = (1 / gamma) * (signal_scaled / variance)
        return weights

    # For w < 1: Matrix inversion
    vol_matrix = np.diag(vol_aligned.values)
    cov_matrix = vol_matrix @ corr_aligned.values @ vol_matrix
    cov_diag = np.diag(np.diag(cov_matrix))
    cov_shrunk = (1 - w) * cov_matrix + w * cov_diag

    try:
        cov_shrunk_inv = np.linalg.inv(cov_shrunk)
    except np.linalg.LinAlgError:
        cov_shrunk_inv = np.linalg.pinv(cov_shrunk)

    weights = (1 / gamma) * cov_shrunk_inv @ signal_scaled.values
    return pd.Series(weights, index=signal_scaled.index)


def backtest_epo_strategy(monthly_excess_returns: pd.DataFrame,
                          signals: pd.DataFrame,
                          correlation_matrices: dict,
                          volatilities_dict: dict,
                          gamma: float,
                          w: float) -> tuple:

    portfolio_returns = []
    portfolio_weights_history = []
    dates = []

    for rebalance_date in sorted(correlation_matrices.keys()):
        try:
            date_idx = monthly_excess_returns.index.get_loc(rebalance_date)
        except KeyError:
            continue

        if date_idx >= len(monthly_excess_returns) - 1:
            break

        next_date = monthly_excess_returns.index[date_idx + 1]

        if rebalance_date not in signals.index:
            continue

        signal = signals.loc[rebalance_date]
        corr_matrix = correlation_matrices[rebalance_date]
        vols = volatilities_dict[rebalance_date]

        weights = calculate_epo_weights(signal, corr_matrix, vols, gamma, w,
                                        MIN_VOLATILITY, SIGNAL_SCALING)

        if len(weights) == 0:
            continue

        next_returns = monthly_excess_returns.loc[next_date, weights.index]
        portfolio_return = (weights * next_returns).sum()

        portfolio_returns.append(portfolio_return)
        portfolio_weights_history.append(weights)
        dates.append(next_date)

    results = pd.DataFrame({'date': dates, 'return': portfolio_returns})
    results.set_index('date', inplace=True)

    return results, portfolio_weights_history


def calculate_indmom_benchmark(monthly_excess_returns: pd.DataFrame,
                               signals: pd.DataFrame) -> pd.DataFrame:
    portfolio_returns = []
    dates = []

    for i in range(len(signals)):
        date = signals.index[i]
        try:
            date_idx = monthly_excess_returns.index.get_loc(date)
        except KeyError:
            continue

        if date_idx >= len(monthly_excess_returns) - 1:
            break

        next_date = monthly_excess_returns.index[date_idx + 1]
        signal = signals.loc[date].dropna()

        if len(signal) == 0:
            continue

        common_assets = signal.index.intersection(monthly_excess_returns.columns)
        signal_aligned = signal.loc[common_assets]
        next_returns = monthly_excess_returns.loc[next_date, common_assets]

        portfolio_return = (signal_aligned * next_returns).sum()
        portfolio_returns.append(portfolio_return)
        dates.append(next_date)

    results = pd.DataFrame({'date': dates, 'return': portfolio_returns})
    results.set_index('date', inplace=True)

    return results


#Printer performance
def calculate_performance_metrics(returns: pd.Series, strategy_name: str = "") -> dict:
    cumulative_return = (1 + returns).prod() - 1
    n_years = len(returns) / 12
    annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1
    monthly_vol = returns.std()
    annualized_vol = monthly_vol * np.sqrt(12)
    sharpe = (returns.mean() / monthly_vol) * np.sqrt(12) if monthly_vol > 0 else 0
    cum_returns = (1 + returns).cumprod()

    return {
        'Strategy': strategy_name,
        'Total Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe,
        'Win Rate': (returns > 0).sum() / len(returns),
        'Best Month': returns.max(),
        'Worst Month': returns.min()
    }


def calculate_leverage(weights_history: list) -> pd.Series:
    return pd.Series([abs(w).sum() for w in weights_history])


#Visualisering
def plot_cumulative_returns(epo_results: pd.DataFrame,
                            indmom_results: pd.DataFrame) -> None:
    epo_cum = (1 + epo_results['return']).cumprod()
    indmom_cum = (1 + indmom_results['return']).cumprod()

    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    plt.plot(epo_cum.index, epo_cum.values, label='EPO Equity 1', linewidth=LINE_WIDTH)
    plt.plot(indmom_cum.index, indmom_cum.values, label='INDMOM Benchmark',
             linewidth=LINE_WIDTH, linestyle='--', alpha=0.7)

    plt.title('Kumulativt Afkast: EPO vs INDMOM (1947-2018)', fontsize=14, pad=20)
    plt.xlabel('Dato', fontsize=12)
    plt.ylabel('Kumulativt Afkast', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe(epo_results: pd.DataFrame,
                        indmom_results: pd.DataFrame,
                        window: int = 36) -> None:
    epo_rolling = epo_results['return'].rolling(window).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(12) if x.std() > 0 else 0
    )
    indmom_rolling = indmom_results['return'].rolling(window).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(12) if x.std() > 0 else 0
    )

    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    plt.plot(epo_rolling.index, epo_rolling.values, label='EPO Equity 1', linewidth=LINE_WIDTH)
    plt.plot(indmom_rolling.index, indmom_rolling.values, label='INDMOM Benchmark',
             linewidth=LINE_WIDTH, linestyle='--', alpha=0.7)

    plt.title(f'{window}-Måneders Rullende Sharpe Ratio', fontsize=14, pad=20)
    plt.xlabel('Dato', fontsize=12)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_performance_summary(epo_metrics: dict,
                              indmom_metrics: dict,
                              epo_leverage: pd.Series) -> None:
    print("\n" + "=" * 80)
    print("PERFORMANCE SAMMENLIGNING: EPO EQUITY 1 vs INDMOM BENCHMARK")
    print("=" * 80)

    print("\nEPO EQUITY 1")
    print("-" * 80)
    print(f"  Total Return:          {epo_metrics['Total Return']:>12.2%}")
    print(f"  Annualized Return:     {epo_metrics['Annualized Return']:>12.2%}")
    print(f"  Annualized Volatility: {epo_metrics['Annualized Volatility']:>12.2%}")
    print(f"  Sharpe Ratio:          {epo_metrics['Sharpe Ratio']:>12.4f}")
    print(f"  Win Rate:              {epo_metrics['Win Rate']:>12.2%}")
    print(f"  Average Leverage:      {epo_leverage.mean():>12.2f}x")

    print("\nINDMOM BENCHMARK")
    print("-" * 80)
    print(f"  Total Return:          {indmom_metrics['Total Return']:>12.2%}")
    print(f"  Annualized Return:     {indmom_metrics['Annualized Return']:>12.2%}")
    print(f"  Annualized Volatility: {indmom_metrics['Annualized Volatility']:>12.2%}")
    print(f"  Sharpe Ratio:          {indmom_metrics['Sharpe Ratio']:>12.4f}")
    print(f"  Win Rate:              {indmom_metrics['Win Rate']:>12.2%}")

    print("\nFORBEDRING (EPO - INDMOM)")
    print("-" * 80)
    print(f"  Sharpe Ratio:          {epo_metrics['Sharpe Ratio'] - indmom_metrics['Sharpe Ratio']:>+12.4f}")
    print(f"  Annualized Return:     {epo_metrics['Annualized Return'] - indmom_metrics['Annualized Return']:>+12.2%}")
    print(
        f"  Annualized Volatility: {epo_metrics['Annualized Volatility'] - indmom_metrics['Annualized Volatility']:>+12.2%}")
    print("=" * 80 + "\n")


#Main
def main():
    print("\n" + "=" * 80)
    print("EPO EQUITY 1 BACKTEST (1942-2018)")
    print("=" * 80 + "\n")

    # 1. Data indlæsning
    print("Indlæser data...")
    industry_file = get_file_path("Vælg daglige industri-afkast CSV:")
    rf_file = get_file_path("Vælg månedlig risikofri rente CSV:")

    daily_returns = load_and_clean_industry_data(industry_file, EPO_START_DATE, EPO_END_DATE)
    rf_monthly = load_and_clean_rf_data(rf_file, EPO_START_DATE, EPO_END_DATE)
    print(f"{len(daily_returns)} daglige obs, {len(daily_returns.columns)} industrier")

    # 2. Data transformering
    print("\nTransformerer data...")
    monthly_returns = convert_daily_to_monthly_returns(daily_returns)
    monthly_excess_returns = calculate_monthly_excess_returns(monthly_returns, rf_monthly)
    print(f"{len(monthly_excess_returns)} månedlige merafkast")

    # 3. Signal beregning
    print("\nBeregner XSMOM signaler...")
    xsmom_signals = calculate_xsmom_signal(monthly_returns, LOOKBACK_PERIOD_MONTHS)
    print(f"{len(xsmom_signals.dropna(how='all'))} signaler")

    # 4. Risikomodel
    print("\nBeregner risikomodel...")
    correlation_matrices, volatilities = calculate_rolling_correlation_and_volatility(
        monthly_excess_returns, RISK_WINDOW_MONTHS, THETA
    )
    print(f" {len(correlation_matrices)} rebalanceringstidspunkter")

    # 5. Backtest
    print("\nBacktester strategier...")
    epo_results, weights_history = backtest_epo_strategy(
        monthly_excess_returns, xsmom_signals, correlation_matrices,
        volatilities, GAMMA, SHRINKAGE_W
    )
    indmom_results = calculate_indmom_benchmark(monthly_excess_returns, xsmom_signals)
    print(f"EPO: {len(epo_results)} måneder")
    print(f"INDMOM: {len(indmom_results)} måneder")

    # 6. Performance beregning
    print("\nBeregner performance...")
    epo_metrics = calculate_performance_metrics(epo_results['return'], 'EPO Equity 1')
    indmom_metrics = calculate_performance_metrics(indmom_results['return'], 'INDMOM')
    epo_leverage = calculate_leverage(weights_history)

    # 7. Resultater
    print_performance_summary(epo_metrics, indmom_metrics, epo_leverage)

    # 8. Visualisering
    print("Genererer plots...")
    plot_cumulative_returns(epo_results, indmom_results)
    plot_rolling_sharpe(epo_results, indmom_results, window=36)

    return {
        'epo_results': epo_results,
        'indmom_results': indmom_results,
        'epo_metrics': epo_metrics,
        'indmom_metrics': indmom_metrics,
        'leverage': epo_leverage
    }


if __name__ == "__main__":
    results = main()