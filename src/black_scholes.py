import numpy as np
import numpy.typing as npt
from scipy.stats import norm


def calc_d1_d2(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    rate: npt.ArrayLike,
    sigma: npt.ArrayLike,
    expiry: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the d1 and d2 terms of the Black-Scholes formula.

    Assumes constant volatility, constant risk-free rate,
    no dividends, and log-normally distributed returns (GBM).

    Parameters
    ----------
    spot : array-like
        Current price of the underlying asset.
    strike : array-like
        Strike price of the option.
    rate : array-like
        Continuously compounded risk-free rate (annualised).
    sigma : array-like
        Volatility of the underlying (annualised). Must be > 0.
    expiry : array-like
        Time to expiration in years. Must be > 0.

    Returns
    -------
    d1 : np.ndarray
    d2 : np.ndarray

    Example
    -------
    >>> calc_d1_d2(100, 100, 0.05, 0.2, 1.0)
    (array([0.35]), array([0.15]))
    """
    # convert to array
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    expiry = np.asarray(expiry, dtype=float)

    # Check for valid data
    if not np.all(spot > 0):
        raise ValueError("spot price must be positive.")

    if not np.all(strike > 0):
        raise ValueError("strike price must be positive.")

    if not np.all(sigma > 0):
        raise ValueError("volatility must be positive.")

    if not np.all(expiry > 0):
        raise ValueError("time to expiry must be positive.")

    # computing d1 and d2
    term1 = np.log(spot / strike)
    term2 = (rate + 0.5 * sigma**2) * expiry
    sqrt_expiry = np.sqrt(expiry)
    d1 = (term1 + term2) / (sigma * sqrt_expiry)
    d2 = d1 - sigma * sqrt_expiry

    return d1, d2


def bs_option_price(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    rate: npt.ArrayLike,
    sigma: npt.ArrayLike,
    expiry: npt.ArrayLike,
    option_type: str = "call",
) -> np.ndarray:
    """
    Compute the call or put option price using the Black-Scholes formula.

    Assumes constant volatility, constant risk-free rate,
    no dividends, and log-normally distributed returns (GBM).

    Parameters
    ----------
    spot : array-like
        Current price of the underlying asset.
    strike : array-like
        Strike price of the option.
    rate : array-like
        Continuously compounded risk-free rate (annualised).
    sigma : array-like
        Volatility of the underlying (annualised). Must be > 0.
    expiry : array-like
        Time to expiration in years. Must be > 0.
    option_type : str
        Specify the type of option, 'call' or 'put'. Default is 'call'

    Returns
    -------
    price : np.ndarray

    Example
    -------
    >>> bs_option_price(100, 100, 0.05, 0.2, 1.0, option_type='call')
    array([10.451])
    """

    option_type = option_type.lower()

    d1, d2 = calc_d1_d2(spot, strike, rate, sigma, expiry)

    discount = np.exp(-rate * expiry)

    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    elif option_type == "put":
        price = strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return price


def bs_delta(d1: npt.ArrayLike, option_type: str = "call") -> np.ndarray:
    """
    Computes the Greek delta of an option. It measures the sensitivity of the
    option price with respect to the spot price.

    Delta is defined as:
        call: N(d1)
        put:  N(d1) - 1

    where N(.) is the cumulative standard Normal distribution.

    Parameters
    ----------
    d1 : array-like
        d1 of the Black-Scholes formula
    option_type : str
        Specify the type of option, 'call' or 'put'. Default is 'call'

    Returns
    -------
    delta : np.ndarray

    Example
    -------
    >>> bs_delta(0.5, option_type='call')
    array([0.6915])
    """
    option_type = option_type.lower()
    d1 = np.asarray(d1, dtype=float)

    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return delta


def bs_gamma(
    spot: npt.ArrayLike,
    sigma: npt.ArrayLike,
    expiry: npt.ArrayLike,
    d1: npt.ArrayLike,
) -> np.ndarray:
    """
    Computes the Greek gamma of an option. It measures the sensitivity of
    the option price with respect to delta.

    Gamma is defined as:
        N'(d1) / (spot * sigma * sqrt(T))

    where N'(.) is the standard Normal probability density function.

    Parameters
    ----------
    spot : array-like
        Current price of the underlying asset.
    sigma : array-like
         Volatility of the underlying (annualised). Must be > 0.
    expiry : array-like
        Time to expiration in years. Must be > 0.
    d1 : array-like
        d1 of the Black-Scholes formula

    Returns
    -------
    gamma : np.ndarray

    Example
    -------
    >>> bs_gamma(100, 0.2, 1.0, 0.35)
    array([0.0187])
    """

    # Convert to array
    spot = np.asarray(spot, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    expiry = np.asarray(expiry, dtype=float)
    d1 = np.asarray(d1, dtype=float)

    # Check for valid data
    if not np.all(spot > 0):
        raise ValueError("spot price must be positive.")

    if not np.all(sigma > 0):
        raise ValueError("volatility must be positive.")

    if not np.all(expiry > 0):
        raise ValueError("time to expiry must be positive")

    gamma = norm.pdf(d1) / (spot * sigma * np.sqrt(expiry))

    return gamma


def bs_vega(
    spot: npt.ArrayLike,
    expiry: npt.ArrayLike,
    d1: npt.ArrayLike,
) -> np.ndarray:
    """
    Computes the Greek vega of an option. Which measures the sensitivity
    of the option price with respect to volatility.

    Vega is defined as:
        spot * N'(d1) * sqrt(T))

    where N'(.) is the standard Normal probability density function.

    Parameters
    ----------
    spot : array-like
        Current price of the underlying asset.
    expiry : array-like
        Time to expiration in years. Must be > 0.
    d1 : array-like
        d1 of the Black-Scholes formula

    Returns
    -------
    vega : np.ndarray

    Example
    -------
    >>> bs_vega(100, 1.0, 0.35)
    array([37.52])
    """

    # Convert to array
    spot = np.asarray(spot, dtype=float)
    expiry = np.asarray(expiry, dtype=float)
    d1 = np.asarray(d1, dtype=float)

    # Check for valid data
    if not np.all(spot > 0):
        raise ValueError("spot price must be positive.")

    if not np.all(expiry > 0):
        raise ValueError("time to expiry must be positive")

    vega = spot * norm.pdf(d1) * np.sqrt(expiry)

    return vega


def bs_theta(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    rate: npt.ArrayLike,
    sigma: npt.ArrayLike,
    expiry: npt.ArrayLike,
    d1: npt.ArrayLike,
    d2: npt.ArrayLike,
    option_type: str = "call",
    annualised: bool = True,
) -> np.ndarray:
    """
    Computes the Greek theta of an option. Which measures the sensitivity
    of the option price with respect to expiry.

    Theta is defined as:
        call: (-spot * N'(d1) * sigma / (2 * sqrt(T)))) - r * K * exp(-r*T) * N(d2)
        put:  (-spot * N'(d1) * sigma / (2 * sqrt(T)))) + r * K * exp(-r*T) * N(-d2)

    where N'(.) is the standard Normal probability density function and N(.) is the standard
    normal cumulative distribution. Theta is the price decay with respect to expiry, which is
    a negative value.

    Parameters
    ----------
    spot : array-like
        Current price of the underlying asset.
    strike : array-like
        Strike price of the option.
    rate : array-like
        Continuously compounded risk-free rate (annualised).
    sigma : array-like
        Volatility of the underlying (annualised). Must be > 0.
    expiry : array-like
        Time to expiration in years. Must be > 0.
    d1: array-like
        d1 of the Black-Scholes formula
    d2: array-like
        d2 of the Black-Scholes formula
    option_type : str
        Specify the type of option, 'call' or 'put'. Default is 'call'
    annualised : bool
        If annualised is True, theta is calculated as price decay per year.
        If False, theta is calculated as price decay per day dividing per year value
        by 365.00

    Returns
    -------
    theta : np.ndarray

    Example
    -------
    >>> bs_theta(100, 100, 0.05, 0.2, 1.0, 0.35, 0.15, option_type='call', annualized=True)
    (array([-6.414]))
    """
    # convert to array
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    rate = np.asarray(rate, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    expiry = np.asarray(expiry, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)

    # Check for valid data
    if not np.all(spot > 0):
        raise ValueError("spot price must be positive.")

    if not np.all(strike > 0):
        raise ValueError("strike price must be positive.")

    if not np.all(sigma > 0):
        raise ValueError("volatility must be positive.")

    if not np.all(expiry > 0):
        raise ValueError("time to expiry must be positive.")

    # calculating theta as price decay per year.
    term1 = -(spot * norm.pdf(d1) * sigma) / (2 * np.sqrt(expiry))
    term2 = rate * strike * np.exp(-rate * expiry)
    if option_type == "call":
        theta = term1 - term2 * norm.cdf(d2)
    elif option_type == "put":
        theta = term1 + term2 * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if not annualised:
        theta = theta / 365.00  # convert to price decay per day

    return theta


class EuropeanOption:
    """
    A European option priced under the Black-Scholes model.

    Parameters
    ----------
    spot : float
    strike : float
    rate : float
    sigma : float
    expiry : float

    Notes
    -----
    Inputs are treated as immutable. Modifying attributes
    after construction will produce incorrect results.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        rate: float,
        sigma: float,
        expiry: float,
    ) -> None:
        # store inputs
        self.spot = spot
        self.strike = strike
        self.rate = rate
        self.sigma = sigma
        self.expiry = expiry

        # call calc_d1_d2 once and store d1, d2
        self.d1, self.d2 = calc_d1_d2(
            self.spot, self.strike, self.rate, self.sigma, self.expiry
        )

    def price(self, option_type: str = "call") -> np.ndarray:
        return bs_option_price(
            self.spot,
            self.strike,
            self.rate,
            self.sigma,
            self.expiry,
            option_type=option_type,
        )

    def delta(self, option_type: str = "call") -> np.ndarray:
        return bs_delta(self.d1, option_type=option_type)

    def gamma(self) -> np.ndarray:
        return bs_gamma(self.spot, self.sigma, self.expiry, self.d1)

    def vega(self) -> np.ndarray:
        return bs_vega(self.spot, self.expiry, self.d1)

    def theta(self, option_type: str = "call", annualised: bool = True) -> np.ndarray:
        return bs_theta(
            self.spot,
            self.strike,
            self.rate,
            self.sigma,
            self.expiry,
            self.d1,
            self.d2,
            option_type=option_type,
            annualised=annualised,
        )
