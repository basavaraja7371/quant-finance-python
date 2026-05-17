import numpy as np
import numpy.typing as npt
import pandas as pd
import yfinance
from scipy.optimize import brentq
from src.black_scholes import bs_option_price


def fetch_option_chain(
    ticker: str,
    expiry: str,
    min_bid: float = 0.0,
    min_volume: int = 0,
    min_open_interest: int = 0,
    max_spread_pct: float = 0.8,
    min_moneyness: float = 0.7,
    max_moneyness: float = 1.3,
    max_spread_abs: float = 1.0,
    use_open_interest: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Fetches the option chain data using yfinance api, for the given ticker and
    expiry. It applys various filters with user defined values.

    Parameters
    ----------
    ticker : str
        The ticker symbol of the stock or the underlying.
    expiry : str
        Expiry date of the option. Must be in the ISO format 'YYYY-MM-DD`. eg. '2022-02-23'
    min_bid : float, optional
        Minimum bid price. Options with zero bid have no active buyer, by default 0.0
    min_volume : int, optional
        Minimum volume of the option trade, by default 0
    min_open_interest : int, optional
        Minimum open interest of the option, by default 0
    max_spread_pct : float, optional
        Maximim bid-ask spread, by default 0.8
    min_moneyness : float, optional
        Minimum moneyness: min(strike/spot), by default 0.7
    max_moneyness : float, optional
        Maximum moneyness max(strike/spot), by default 1.3
    max_spread_abs : float, optional
        Absolute maximum bid-ask spread, not percentage, by default 1.0.
    use_open_interest : bool, optional
        Whether to use Open interest filter rather volume filter, by default True
    verbose : bool, optional
        Print information of each filter, by default False

    Returns
    -------
    pd.DataFrame
        The dataframe of the option chain containing columns spot, strike, moneyness, expiry,
        mid_price, option_type, and time_to_expiry
    """
    tk = yfinance.Ticker(ticker)
    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()
    calls["option_type"] = "call"
    puts = chain.puts.copy()
    puts["option_type"] = "put"
    df = pd.concat([calls, puts], ignore_index=True)

    if verbose:
        print("Number of options")
        print("-----------------")
        print(f"Raw optioins: {len(df)}")

    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0

    spot = tk.fast_info["last_price"]
    df["spot"] = spot

    df["moneyness"] = df["strike"] / spot

    # liquidity filters: keep only liquid options
    df = df[df["bid"] > min_bid]
    df = df[df["ask"] > 0]
    if verbose:
        print(f"After bid filter: {len(df)}")

    if use_open_interest:
        df = df[df["openInterest"] >= min_open_interest]
    else:
        df = df[df["volume"] > min_volume]

    if verbose:
        print(f"After volume or OI filter: {len(df)}")

    df = df[
        (df["ask"] - df["bid"]) / df["mid_price"] < max_spread_pct
    ]  # spread < 50% of mid
    if verbose:
        print(f"After spread filter: {len(df)}")

    df = df[(df["moneyness"] >= min_moneyness) & (df["moneyness"] <= max_moneyness)]
    if verbose:
        print(f"After moneyness filter: {len(df)}")

    df["spread"] = df["ask"] - df["bid"]
    # df = df[df["spread"] < max_spread_abs]  # maximum $1 wide bid-ask
    df = df[
        df["spread"] < np.maximum(max_spread_abs, df["mid_price"] * max_spread_pct)
    ]  # maximum $1 wide bid-ask

    if verbose:
        print(f"After absolute spread filter: {len(df)}")

    df["time_to_expiry"] = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365.25
    df["expiry"] = expiry
    df = df.dropna()

    return df[
        [
            "spot",
            "strike",
            "moneyness",
            "expiry",
            "mid_price",
            "option_type",
            "time_to_expiry",
        ]
    ]


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
