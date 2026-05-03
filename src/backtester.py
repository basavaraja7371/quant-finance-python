import numpy as np
import pandas as pd
import numpy.typing as npt


def generate_signals(
    prices: npt.ArrayLike, short_window: int = 20, long_window: int = 50
) -> np.ndarray:
    """
    It generates a buy-sell signals, with 1 (buy) and 0 (sell).
    Returns an array, of (0 if short MA < long MA and  1 if short MA >= long MA)

    Parameters
    ----------
    prices : npt.ArrayLike
        Close prices of the stock or asset
    short_window : int, optional
        Window length in days to calculate short moving average, by default 20
    long_window : int, optional
        Window length in days to calculate long moving average, by default 50

    Returns
    -------
    trade_signals : np.ndarray
        Array of 0 and 1 s indicating sell and buy
    """

    prices = pd.Series(prices)
    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()

    trade_signals = (short_ma >= long_ma).astype(int)

    return trade_signals


def compute_returns(
    prices: npt.ArrayLike, trade_signals: npt.ArrayLike
) -> tuple[pd.Series, pd.Series]:
    """
    Compute the returns for the Moving Average Crossover strategy
    and Benchmark buy and hold strategy.

    Parameters
    ----------
    prices : npt.ArrayLike
        Asset prices
    trade_signals : npt.ArrayLike
        Trade signal array with buy ans sell signals based on  short MA
        crossing the long MA.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Returns of the MA crossover strategy, Returns of Buy and hold strategy
    """
    prices = pd.Series(prices)
    trade_signals = pd.Series(trade_signals)

    log_returns = np.log(prices).diff() * 100

    strategy_returns = (log_returns * trade_signals.shift(1)).dropna()
    benchmark_returns = log_returns.dropna()

    # align both to same index
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns[common_index]
    benchmark_returns = benchmark_returns[common_index]

    return strategy_returns, benchmark_returns


def compute_metrics(
    returns: npt.ArrayLike, benchmark_returns: npt.ArrayLike, rate: float
) -> dict:
    """
    Computes the basic 6 metrics of the backtesting.
    1. CAGR
    2. Win rate
    3. Sharpe ratio
    4. Maximum drawdown
    5. Calmar ratio
    6. Alpha

    Parameters
    ----------
    returns : npt.ArrayLike
        Return of the MA crossover strategy
    benchmark_returns: npt.ArrayLike
        Benchmark returns for the buy and hold strategy
    rate : float
        risk free rate annualized

    Returns
    -------
    dict with keys
        'CAGR' : CAGR
        'WinRate' : Win rate
        'SharpeRatio' : Sharpe Ratio
        'MDD' : Maximum drawdown
        'CalmarRatio': Calmar ratio
        'Alpha' : Alpha of the MA crossover strategy
    """
    returns = np.asarray(returns) / 100
    benchmark_returns = np.asarray(benchmark_returns) / 100

    p = 252  # Assuming 252 trading days in a year
    n_years = len(returns) / p

    # 1. CAGR
    strategy_cagr = np.exp(np.sum(returns)) ** (1 / n_years) - 1
    benchmark_cagr = np.exp(np.sum(benchmark_returns)) ** (1 / n_years) - 1

    # 2. Win rate
    active_returns = returns[
        returns != 0
    ]  # Filter to only days when the strategy was active
    WinRate = np.mean(active_returns > 0)

    # 3. Sharpe ratio
    rate_p = rate / p
    SharpeRatio = np.sqrt(p) * np.mean(returns - rate_p) / np.std(returns)

    # 4.  Maximum Drawdown
    wealth = np.exp(np.cumsum(returns))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peaks) / peaks
    MDD = abs(np.min(drawdowns))  # since negative min is used for max drawdown

    # 5. Calmar ratio
    CalmarRatio = strategy_cagr / MDD

    # 6. Alpha
    Alpha = strategy_cagr - benchmark_cagr

    metrics = {
        "CAGR": strategy_cagr,
        "WinRate": WinRate,
        "SharpeRatio": SharpeRatio,
        "MDD": MDD,
        "CalmarRatio": CalmarRatio,
        "Alpha": Alpha,
    }

    return metrics


def run_backtest(
    prices: npt.ArrayLike, rate: float, short_window: int = 20, long_window: int = 50
) -> dict:
    """
    Runs the backtest for MA crossover strategy and returns
    1. Strategy returns
    2. Benchmark returns
    3. Trade signals
    4. Strategy metrics
    5. Benchmark metrics

    Parameters
    ----------
    prices : npt.ArrayLike
        Prices array of the underlying asset/stock.
    rate : float
        risk free rate
    short_window : int, optional
        Window length in days to calculate short moving average,, by default 20
    long_window : int, optional
        Window length in days to calculate long moving average,, by default 50

    Returns
    -------
    dict with keys
        'strategy_returns'  : pd.Series — daily returns of the strategy
        'benchmark_returns' : pd.Series — daily returns of buy and hold
        'signals'           : pd.Series — trade signals over time
        'strategy_metrics'  : dict — Sharpe, MDD, CAGR, etc for strategy
        'benchmark_metrics' : dict — same metrics for buy and hold
    """

    trade_signals = generate_signals(
        prices, short_window=short_window, long_window=long_window
    )

    strategy_returns, benchmark_returns = compute_returns(prices, trade_signals)

    strategy_metrics = compute_metrics(strategy_returns, benchmark_returns, rate)
    benchmark_metrics = compute_metrics(benchmark_returns, benchmark_returns, rate)

    results = {
        "strategy_returns": pd.Series(strategy_returns),
        "benchmark_returns": pd.Series(benchmark_returns),
        "signals": pd.Series(trade_signals),
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
    }
    return results
