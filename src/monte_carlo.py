import numpy as np
import numpy.typing as npt
from typing import Optional


def simulate_gbm_terminal(
    spot: float,
    rate: float,
    sigma: float,
    expiry: float,
    n_simulations: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulates the terminal stock price assuming that the stock price follows
    a geometric Browninan motion, for n_iteration and returns the
    terminal price S of each step in an array.

    Parameters
    ----------
    spot : float
        Initial spot price at time zero (must be > 0)
    rate : float
        Risk free rate
    sigma : float
        Volatility of the stock (must be > 0)
    expiry : float
        The time period for the stock to evolve
    n_simulations : int, optional
        Number of simulation to be carried out, by default 100
    seed : Optional[int], optional
        Random seed

    Returns
    -------
    terminal_prices : np.ndarray
        Terminal stock price of each simulation as an array
    """

    if spot <= 0:
        raise ValueError("spot must be positive.")

    if sigma <= 0:
        raise ValueError("volatility must be positive.")

    if not isinstance(n_simulations, int):
        raise ValueError("n_simulations must be an integer.")

    if expiry <= 0:
        raise ValueError("expiry must be positive.")

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_simulations)

    terminal_prices = spot * np.exp(
        (rate - 0.5 * sigma**2) * expiry + sigma * np.sqrt(expiry) * Z
    )

    return terminal_prices


def compute_payoff(
    spot: npt.ArrayLike, strike: float, option_type: str = "call"
) -> np.ndarray:
    """
    Compute the payoff of an European option.
        call : max(S - K, 0)
        put : max(K - S, 0)

    Parameters
    ----------
    spot : npt.ArrayLike
        spot prices of the underlying
    strike : float
        strike for the option
    option_type : str, optional
        option type call or put, by default 'call'

    Returns
    -------
    payoff : np.ndarray
        payoff of the option
    """
    spot = np.asarray(spot, dtype=float)
    option_type = option_type.lower()

    if not np.all(spot > 0):
        raise ValueError("spot price must be positive.")

    if strike <= 0:
        raise ValueError("strike price must be positive")

    if option_type == "call":
        payoff = np.maximum(spot - strike, 0)
    elif option_type == "put":
        payoff = np.maximum(strike - spot, 0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return payoff


def mc_option_price(
    payoff: npt.ArrayLike, rate: float, expiry: float
) -> tuple[float, float]:
    """
    Calculate the average of the payoff and discount it to t=0.

    Parameters
    ----------
    payoff : npt.ArrayLike
        option payoff from the mc_simulations
    rate : float
        risk free rate
    expiry : float
        Time to expiry of the option

    Returns
    -------
    price : float
        Discounted average of the payoffs
    standard_error : float
        Standard error of the simulation. The 95% confidence interval
        is approximately price ± 1.96 * standard_error.
    """
    payoff = np.asarray(payoff)

    discount = np.exp(-rate * expiry)
    price = payoff.mean() * discount
    standard_error = payoff.std() / np.sqrt(len(payoff)) * discount

    return price, standard_error
