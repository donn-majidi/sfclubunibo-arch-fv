import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

def impute_missing(series: pd.Series, max_iter: int = 50) -> pd.Series:
    """
    Impute missing values in a univariate time series using a Kalman smoother.

    Fits a local level model (random walk + noise) via the EM algorithm to
    estimate the observation variance (sigma2_irregular) and level variance
    (sigma2_level) from the observed data. The Kalman smoother then fills
    NaN positions with the smoothed state mean at each time step.

    Model
    -----
        y_t   = mu_t + eps_t,    eps_t ~ N(0, sigma2_irregular)   [observation]
        mu_t  = mu_{t-1} + xi_t, xi_t ~ N(0, sigma2_level)        [state / level]

    Parameters
    ----------
    series : pd.Series
        Univariate time series, may contain NaN values anywhere (interior,
        leading, or trailing). Must have at least 2 non-missing observations.
    max_iter : int, default 50
        Maximum number of iterations for the Powell optimizer used to
        estimate model variances via MLE.

    Returns
    -------
    pd.Series
        Series of the same length and index as the input, with NaN positions
        replaced by smoothed state means. Non-missing values are unchanged.

    Raises
    ------
    ValueError
        If the series has fewer than 2 non-missing observations (model
        cannot be identified).

    """
    if series.isna().all():
        raise ValueError("Series is entirely NaN — cannot fit a Kalman smoother.")

    n_obs = series.notna().sum()
    if n_obs < 2:
        raise ValueError(
            f"Need at least 2 non-missing observations to fit the model, got {n_obs}."
        )

    # statsmodels requires a float array; NaNs are natively handled as missing
    y = series.astype(float).values

    # --- Fit local level model via numerical MLE ---
    model = UnobservedComponents(y, level="local level")
    result = model.fit(
        method="powell",
        maxiter=max_iter,
        disp=False,
    )

    # --- Kalman smoother: recover smoothed state means ---
    smoothed_means = result.smoother_results.smoothed_state[0]  # shape (T,)

    # Replace only the NaN positions; keep observed values as-is
    imputed_values = np.where(np.isnan(y), smoothed_means, y)

    return pd.Series(imputed_values, index=series.index, name=series.name)

def log_transform(series: pd.Series) -> pd.Series:
    """
    Transform price series into log returns.
    
    Formula
    -------
        r_t = log(p_t) - log(p_{t-1})

    Parameters
    ----------
    series : pd.Series
        Univariate price series, if the series contains missing values must
        first be imputed with the impute_missing() function.

    Returns
    -------
    seires : pd.Series
        Series of log returns with length equal to len(input)-1 and index as
        the input series.

    Raises
    ------
    ValueError
        If the series has missing values.

    """
    
    if series.isna().any():
        raise ValueError("Series contains missing values. Consider imputing first.")
        
    log_returns = np.log(series).diff().dropna()
    return log_returns