import numpy as np
import pandas as pd
import yfinance as yf
import numpy.typing as npt
from typing import Optional
from arch import arch_model
from arch.univariate.base import ARCHModelResult


def fetch_returns(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "3y",
) -> pd.Series:
    """
    Using yfinance, it fetches the close prices of the ticker for the date
    range specified by start and stop or for the period asked, and calculates
    the log-returns in percentages and returns it as a pandas series.

    Parameters
    ----------
    ticker : str
        Ticker of the stock
    start : Optional[str], optional
        start date for the data: eg: '2024-01-01', by default None
    end : Optional[str], optional
        end date for the data: eg: '2025-01-01', by default None
    period : str, optional
        Time period for which the returns are calculated, by default '3y'

    Returns
    -------
    pd.Series
        log-returns extracted for the given ticker
    """
    if start and end:
        data = yf.download(ticker, start=start, end=end)
    else:
        data = yf.download(ticker, period=period)

    close_prices = data["Close"].squeeze()  # squeeze is added in order return a series
    log_returns = np.log(close_prices).diff().dropna() * 100

    return log_returns


def fit_garch(
    returns: npt.ArrayLike, p: int = 1, q: int = 1, dist: str = "normal"
) -> dict:
    """
    Fit the GARCH(p, q) model to the returns data.

    Parameters
    ----------
    returns : npt.ArrayLike
        stock returns data
    p : int, optional
        lag order of squared variance, by default 1
    q : int, optional
        lag order of returns, by default 1
    dist : str, optional
        the distribution of the errors, by default 'normal'


    Returns
    -------
    dict with keys:
        'result'  : ARCHModelResult — fitted model object
        'omega'   : float — constant variance term
        'alpha'   : float — reaction coefficient
        'beta'    : float — persistence coefficient
        'aic'     : float — Akaike Information Criterion
    """

    model = arch_model(returns, vol="Garch", p=p, q=q, dist=dist)
    result = model.fit(disp="off")
    omega = result.params["omega"]
    alpha = result.params["alpha[1]"]
    beta = result.params["beta[1]"]
    aic = result.aic

    fitted_results = {
        "result": result,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "aic": aic,
    }

    return fitted_results


def forecast_volatility(
    garch_fitted_object: ARCHModelResult, horizon: int = 5
) -> np.ndarray:
    """
    Using fitted GARCH(p, q) model, predicts the future forecasts for a given horizon.

    Parameters
    ----------
    garch_fitted_object : ARCHModelResult
        GARCH(p, q) fitted model
    horizon : int, optional
        horizon for prediction, by default 5

    Returns
    -------
    np.ndarray
        forecasted volatility array for the given horizon
    """
    forecast_obj = garch_fitted_object.forecast(horizon=horizon, reindex=False)
    variance_forecast = forecast_obj.variance.values[-1]
    volatility_forecast = np.sqrt(variance_forecast) / 100
    return volatility_forecast
