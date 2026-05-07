import numpy.typing as npt
import pandas as pd
import pandas_datareader as pdr
import statsmodels.api as sm


def fetch_factor_data(start: str, end: str) -> pd.DataFrame:
    """
    Get factor data from Kenneth French from his website using pandas_datareader.


    Parameters
    ----------
    start : str
        start date for the data, in the format 'YYYY-MM-DD'
    end : str
        end date for the data, in the format 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
        Returns DataFrame with columns Mkt-RF, SMB, HML, RF. The data is in percentages.
    """

    factors = pdr.get_data_famafrench(
        "F-F_Research_Data_Factors_daily", start=start, end=end
    )
    df = factors[0]  # first element is the daily data
    return df


def fit_factor_model(
    returns: npt.ArrayLike,
    factor_returns: npt.ArrayLike,
) -> dict:
    """
    Fit the ordinary least squares model to the data and calculates the
    coefficients beta_i and alpha. Also returns relavant statistics.

    Parameters
    ----------
    returns : npt.ArrayLike
        Assets returns, in percentage.
    factor_returns : npt.ArrayLike
        Factor returns, in percentage

    Returns
    -------
    dict with keys:
        "betas": coefficents of the model, excluding alpha
        "t_statistic": t-statistic for the betas
        "p_values": p-values for the betas
        "r_squared": R-squared statistic
        "alpha": alpha of the model
        "alpha_p_value": alpha p-value
    """

    common_index = factor_returns.index.intersection(returns.index)
    returns = returns.loc[common_index]
    factor_returns = factor_returns.loc[common_index].drop(columns=["RF"])

    X = sm.add_constant(factor_returns)
    results = sm.OLS(returns, X).fit()

    result_dict = {
        "betas": results.params.iloc[1:],
        "t_statistic": results.tvalues.iloc[1:],
        "p_values": results.pvalues.iloc[1:],
        "r_squared": results.rsquared,
        "alpha": results.params.iloc[0],
        "alpha_p_value": results.pvalues.iloc[0],
    }
    return result_dict


def compute_factor_attribution(
    returns: npt.ArrayLike,
    factor_returns: pd.DataFrame,
    betas: npt.ArrayLike,
    alpha: float,
) -> dict:
    """
    Calculate total return and decompose them into factor contributions and alpha.

    Parameters
    ----------
    returns : npt.ArrayLike
        Asset return array
    factor_returns : pd.DataFrame
        Factor returns
    betas : npt.ArrayLike
        betas of the Fama-French model
    alpha : float
        alpha of the Fama-French model

    Returns
    -------
    dict with keys:
        r_total: total returns of the asset
        r_alpha: alpha contribution to returns
        r_mkt: market contribution to returns
        r_SMB: SMB contribution to returns
        r_HML: HML contribution to returns
        epsilon: residual error
    """

    r_total = returns.mean()
    r_mkt = betas["Mkt-RF"] * factor_returns["Mkt-RF"].mean()
    r_SMB = betas["SMB"] * factor_returns["SMB"].mean()
    r_HML = betas["HML"] * factor_returns["HML"].mean()
    epsilon = r_total - alpha - r_mkt - r_SMB - r_HML

    results = {
        "r_total": r_total,
        "r_alpha": alpha,
        "r_mkt": r_mkt,
        "r_SMB": r_SMB,
        "r_HML": r_HML,
        "epsilon": epsilon,
    }
    return results
