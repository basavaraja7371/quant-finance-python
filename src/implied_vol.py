import numpy as np
import numpy.typing as npt
import pandas as pd
import yfinance
from scipy.optimize import brentq
from src.black_scholes import bs_option_price


def fetch_option_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """
    Fetches the option chain data using yfinance api, for the given ticker and
    expiry.

    Parameters
    ----------
    ticker : str
        The ticker symbol of the stock or the underlying.
    expiry : str
        Expiry date of the option. Must be in the ISO format 'YYYY-MM-DD`. eg. '2022-02-23'

    Returns
    -------
    option_chain : pd.DataFrame
        The dataframe of the option chain containing columns spot, strike, expiry,
        mid_price, option_type, and time_to_expiry
    """
    tk = yfinance.Ticker(ticker)
    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()
    calls["option_type"] = "call"
    puts = chain.puts.copy()
    puts["option_type"] = "put"
    df = pd.concat([calls, puts], ignore_index=True)

    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0

    # liquidity filters: keep only liquid options
    df = df[df["bid"] > 0]
    df = df[df["ask"] > 0]
    df = df[df["volume"] > 0]
    df = df[(df["ask"] - df["bid"]) / df["mid_price"] < 0.5]  # spread < 50% of mid

    df["time_to_expiry"] = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365.25
    df["expiry"] = expiry
    df = df.dropna()
    return df[["strike", "expiry", "mid_price", "option_type", "time_to_expiry"]]


def compute_implied_vol(
    spot: float,
    strike: float,
    rate: float,
    expiry: float,
    market_price: float,
    option_type: str = "call",
) -> float:
    """
    Given spot, strike, rate, expiry, matket_price and option_type, calculate the
    implied volatility using Black-Scholes model. Root finding method is used to
    to solve for sigma for the equation:
         BS_price(sigma)-C_market = 0
    To find the roots brentq is used. Returns np.nan if the root cannot be found
    within the bounds.

    Parameters
    ----------
    spot : float
        Current price of the underlying asset.
    strike : float
        Strike price for the option.
    rate : float
        Continuously compounded risk-free rate. (annualised)
    expiry : float
        Time to expiration in years. (must be > 0)
    market_price : float
        Option price observed in the market having the same parameters.
    option_type : str, optional
        Type of the option, 'call' or 'put', by default "call"

    Returns
    -------
    imp_vol : float
        Implied volatility calculated from Black-Scholes model.
    """

    def objective(sigma):
        return (
            bs_option_price(spot, strike, rate, sigma, expiry, option_type=option_type)
            - market_price
        )

    try:
        imp_vol = brentq(objective, 1e-6, 5.0)
    except ValueError:
        imp_vol = np.nan

    return imp_vol


def build_iv_surface(
    option_dataframe: pd.DataFrame, spot: float, rate: float
) -> pd.DataFrame:
    """
    Calculates the implied volatility surface for the given option data.
    Returns a implied volatility as numpy array.
    Parameters
    ----------
    option_dataframe : pd.DataFrame
        Cleaned option chain data, as a pandas dataframe.
        The dataframe of the option chain must contain columns
        strike, expiry, mid_price, option_type, and time_to_expiry

    spot : float
        Current price of the underlying asset. (must be > 0)

    rate : float
        Continuously compounded risk-free rate. (annualised)
    Returns
    -------
    vol_surface : pd.DataFrame
        Volatility surface dataframe

    """
    df = option_dataframe.copy()

    df["implied_vol"] = df.apply(
        lambda row: compute_implied_vol(
            spot=spot,
            strike=row["strike"],
            rate=rate,
            expiry=row["time_to_expiry"],
            market_price=row["mid_price"],
            option_type=row["option_type"],
        ),
        axis=1,
    )

    return df.dropna(subset=["implied_vol"])
