import pytest
import numpy as np
from src.monte_carlo import simulate_gbm_terminal, compute_payoff, mc_option_price
from src.black_scholes import bs_option_price


def test_simulate_gbm_terminal_shape():
    terminal_prices = simulate_gbm_terminal(
        100, 0.05, 0.20, 1, n_simulations=100, seed=42
    )
    assert terminal_prices.shape == (100,)


def test_simulate_gbm_terminal_reproducible():
    terminal_prices_1 = simulate_gbm_terminal(
        100, 0.05, 0.20, 1, n_simulations=100, seed=42
    )
    terminal_prices_2 = simulate_gbm_terminal(
        100, 0.05, 0.20, 1, n_simulations=100, seed=42
    )

    np.testing.assert_allclose(terminal_prices_1, terminal_prices_2, atol=1e-6)


def test_compute_payoff_call():
    spot = [90, 100, 110]
    strike = 100
    payoff = compute_payoff(spot, strike, option_type="call")

    np.testing.assert_allclose(payoff, np.array([0, 0, 10]), atol=1e-6)


def test_mc_option_price_convergence():
    bs_price = bs_option_price(100, 100, 0.05, 0.2, 1, option_type="call")
    terminal_prices = simulate_gbm_terminal(
        100, 0.05, 0.2, 1, n_simulations=100000, seed=42
    )
    payoff = compute_payoff(terminal_prices, 100, option_type="call")
    mc_price, stderr = mc_option_price(payoff, 0.05, 1)

    np.testing.assert_allclose(bs_price, mc_price, atol=0.1)
