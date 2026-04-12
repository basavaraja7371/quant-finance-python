import pytest
import numpy as np
from src.black_scholes import calc_d1_d2, bs_option_price
from src.black_scholes import bs_delta, bs_gamma, bs_vega, bs_theta
from src.black_scholes import EuropeanOption


def test_calc_d1_d2_atm():
    """At-the-money, known analytical result"""
    d1, d2 = calc_d1_d2(spot=100, strike=100, rate=0.05, sigma=0.2, expiry=1.0)
    np.testing.assert_allclose(d1, 0.35, atol=1e-6)
    np.testing.assert_allclose(d2, 0.15, atol=1e-6)


def test_calc_d1_d2_vectorized():
    """Accepts array inputs and returns arrays of correct shape"""
    spots = np.array([90.0, 100.0, 110.0])
    d1, d2 = calc_d1_d2(spot=spots, strike=100, rate=0.05, sigma=0.2, expiry=1.0)
    assert d1.shape == (3,)
    assert d2.shape == (3,)


def test_calc_d1_d2_invalid_sigma():
    """Zero volatility must raise ValueError."""
    with pytest.raises(ValueError, match="volatility"):
        calc_d1_d2(100, 100, 0.05, sigma=0.0, expiry=1.0)


def test_bs_option_price_invalid_option_type():
    """Invalid option_type such as 'banana' should raise ValueError"""
    with pytest.raises(ValueError, match="option_type"):
        bs_option_price(100, 100, 0.05, 0.2, 1.0, option_type="banana")


def test_bs_call_price_atm():
    """At-the-money, known analytical result for call"""
    price = bs_option_price(
        spot=100, strike=100, rate=0.05, sigma=0.2, expiry=1.0, option_type="call"
    )
    np.testing.assert_allclose(price, 10.4506, atol=1e-3)


def test_bs_put_price_atm():
    """At-the-money, known analytical result for put"""
    price = bs_option_price(
        spot=100, strike=100, rate=0.05, sigma=0.2, expiry=1.0, option_type="put"
    )
    np.testing.assert_allclose(price, 5.5735, atol=1e-3)


def test_bs_delta_atm():
    """At-the-money, known delta value"""
    delta = bs_delta(d1=0.35, option_type="call")
    np.testing.assert_allclose(delta, 0.63683, atol=1e-3)


def test_bs_gamma_atm():
    """At-the-money, known gamma value"""
    gamma = bs_gamma(spot=100, sigma=0.2, expiry=1.0, d1=0.35)
    np.testing.assert_allclose(gamma, 0.01876, atol=1e-3)


def test_bs_vega_atm():
    """At-the-money, known vega value"""
    vega = bs_vega(spot=100, expiry=1.0, d1=0.35)
    np.testing.assert_allclose(vega, 37.52403, atol=1e-3)


def test_bs_theta_atm():
    """At-the-money, known theta value"""
    theta = bs_theta(
        spot=100,
        strike=100,
        rate=0.05,
        sigma=0.2,
        expiry=1.0,
        d1=0.35,
        d2=0.15,
        option_type="call",
        annualised=True,
    )
    np.testing.assert_allclose(theta, -6.4140, atol=1e-3)


def test_bs_theta_call_is_negative():
    """The theta for a long call is always negative"""
    theta = bs_theta(
        spot=100,
        strike=100,
        rate=0.05,
        sigma=0.2,
        expiry=1.0,
        d1=0.35,
        d2=0.15,
        option_type="call",
        annualised=True,
    )
    np.testing.assert_array_less(
        theta, 0, err_msg="Theta for a long call must be positive"
    )


def test_european_option_consistency():
    """Class and functional interface must return identical results."""
    option = EuropeanOption(spot=100, strike=100, rate=0.05, sigma=0.2, expiry=1.0)
    d1, d2 = calc_d1_d2(100, 100, 0.05, 0.2, 1.0)

    # price
    np.testing.assert_allclose(
        option.price("call"),
        bs_option_price(
            option.spot,
            option.strike,
            option.rate,
            option.sigma,
            option.expiry,
            option_type="call",
        ),
    )
    np.testing.assert_allclose(
        option.price("put"),
        bs_option_price(
            option.spot,
            option.strike,
            option.rate,
            option.sigma,
            option.expiry,
            option_type="put",
        ),
    )

    # delta
    np.testing.assert_allclose(option.delta("call"), bs_delta(d1, option_type="call"))

    # gamma
    np.testing.assert_allclose(
        option.gamma(), bs_gamma(option.spot, option.sigma, option.expiry, d1)
    )

    # vega
    np.testing.assert_allclose(option.vega(), bs_vega(option.spot, option.expiry, d1))

    # theta
    np.testing.assert_allclose(
        option.theta("call"),
        bs_theta(
            option.spot,
            option.strike,
            option.rate,
            option.sigma,
            option.expiry,
            d1,
            d2,
            option_type="call",
            annualised=True,
        ),
    )
