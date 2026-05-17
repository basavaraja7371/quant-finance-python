# Quantitative Finance in Python

A quantitative finance library built from first principles. Each module is independently usable, and documented.

## Modules

| Module | File | Description |
|---|---|---|
| Black-Scholes Pricer | `src/black_scholes.py` | European option pricing, Greeks, `EuropeanOption` class |
| Monte Carlo Pricer | `src/monte_carlo.py` | GBM simulation, vectorised MC pricing with confidence intervals |
| Implied Volatility Surface | `src/implied_vol.py` | IV surface from live market data via yfinance |
| GARCH Model | `src/time_series.py` | Volatility modelling and forecasting with GARCH(1,1) |
| Backtesting Engine | `src/backtester.py` | MA crossover strategy with Sharpe, MDD, Calmar metrics |
| Factor Model | `src/factor_model.py` | Fama-French three-factor model and return attribution |

## Project Structure

```
├── src/            ← importable modules
├── tests/          ← pytest test suite
├── notebooks/      ← exploration and visualisation
├── data/           ← raw and processed data
└── results/        ← saved plots and outputs
```

## Setup

```bash
# clone the repository
git clone https://github.com/basavaraja7371//quant-finance-python.git
cd quant-finance-python

# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install numpy scipy pandas matplotlib yfinance statsmodels arch pandas_datareader pytest
```

## Usage

```python
from src.black_scholes import EuropeanOption

option = EuropeanOption(spot=100, strike=100, rate=0.05, sigma=0.2, expiry=1.0)
print(option.price('call'))   # 10.4506
print(option.delta('call'))   # 0.6368
print(option.gamma())         # 0.0188
```

```python
from src.monte_carlo import simulate_gbm_terminal, compute_payoff, mc_option_price

terminal = simulate_gbm_terminal(100, 0.05, 0.2, 1.0, n_simulations=100_000, seed=42)
payoff   = compute_payoff(terminal, strike=100, option_type='call')
price, stderr = mc_option_price(payoff, rate=0.05, expiry=1.0)
print(f"MC price: {price:.4f} ± {1.96*stderr:.4f}")
```

```python
from src.time_series import fetch_returns, fit_garch, forecast_volatility

returns = fetch_returns("SPY", period="3y")
results = fit_garch(returns, p=1, q=1, dist='t')
print(f"alpha: {results['alpha']:.4f}, beta: {results['beta']:.4f}")

vol_forecast = forecast_volatility(results['result'], horizon=22)
print(f"22-day vol forecast: {vol_forecast[-1]*252**0.5:.2%} annualised")
```
## Key Results

**Black-Scholes** (ATM, S=100, K=100, r=0.05, σ=0.2, T=1)

| Output | Value |
|---|---|
| Call price | 10.4506 |
| Put price | 5.5735 |
| Delta | 0.6368 |
| Gamma | 0.0188 |
| Vega | 37.52 |
| Theta | -6.41 per year |

**SPY GARCH(1,1) with Student-t** (3-year daily data)

| Parameter | Value |
|---|---|
| alpha | 0.1007 |
| beta | 0.8219 |
| Long-run vol | 14.0% annualised |
| Half-life | 8.6 days |

**SPY MA Crossover Backtest** (20/50-day, 2023–2026)

| Metric | Strategy | Buy and Hold |
|---|---|---|
| CAGR | 10.9% | 22.3% |
| Sharpe | 0.55 | 1.00 |
| Max Drawdown | 9.6% | 19.2% |
| Alpha | -11.4% | — |

**Fama-French Factor Model** (SPY vs AAPL, 2023–2026)

| Asset | β_mkt | β_smb | β_hml | R² |
|---|---|---|---|---|
| SPY | 0.993 | -0.090 | -0.008 | 0.994 |
| AAPL | 1.120 | -0.250 | -0.144 | 0.459 |

## Dependencies

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `yfinance` — market data
- `statsmodels` — OLS regression
- `arch` — GARCH modelling
- `pandas_datareader` — Fama-French factor data
- `pytest` — testing

## Notes

All models are implemented under standard assumptions (constant volatility, no dividends, continuous trading) unless otherwise stated in the function docstrings. Results depend on the market data available at the time of execution.
Only few test modules are written. It will be completed soon.
