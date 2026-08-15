import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from statsmodels.tsa.stattools import kpss, adfuller
from scipy.stats import chi2 as _chi2, norm as _norm
    
# ---------------------------------------------------------------------------
# Asymptotic critical values for the CUSUM-of-squares test.
# Source: Brown, Durbin & Evans (1975), Table 1.
# The standardised statistic is  max |S_k - k/T|  and the critical value
# is the c(alpha) constant (boundary = c(alpha) / sqrt(T) + 1/T offset
# absorbed into the standardised form below).
# ---------------------------------------------------------------------------
_CUSUM_SQ_CV = {0.10: 0.850, 0.05: 1.143, 0.01: 1.628}

@dataclass
class CUSUMResult:
    """Results container for a single CUSUM-of-squares moment stability test."""
    moment: int           # 2 or 4
    statistic: float      # max |S_k - k/T|  (standardised)
    critical_value: float # asymptotic c.v. at requested alpha
    alpha: float          # significance level used
    reject: bool          # True  →  reject stability
    pvalue_bound: str     # conservative bound string, e.g. "< 0.05"
    breakpoint_index: int # positional index where max deviation occurs
    breakpoint_label: object  # corresponding Series index label
    cusum_path: pd.Series # full S_k path (length T), same index as input
    
def cusum_moment_stability(
    returns: pd.Series,
    moments: list[int] | None = None,
    alpha: float = 0.05,
) -> dict[int, CUSUMResult]:
    """
    Test the stability of the 2nd and/or 4th order moments of a return series
    using the CUSUM-of-squares procedure (Brown, Durbin & Evans, 1975).
 
    For each requested moment ``m``, the series ``x_t = r_t ** m`` is formed
    and its cumulative partial sums are standardised to produce a path S_k in
    [0, 1].  Instability is flagged when the maximum deviation of S_k from the
    expected linear trend k/T exceeds the asymptotic critical value.
 
    Test statistic
    --------------
        S_k = (sum_{t=1}^{k} x_t) / (sum_{t=1}^{T} x_t),   k = 1, ..., T
        stat = sqrt(T) * max_{k} |S_k - k/T|
 
    The critical value c(alpha) is taken from Brown et al. (1975), Table 1.
    Rejection occurs when stat > c(alpha).
 
    Parameters
    ----------
    returns : pd.Series
        Return series (demeaned or raw). Must be free of NaN values.
    moments : list of int, optional
        Which moments to test. Accepts [2], [4], or [2, 4] (default).
    alpha : float, default 0.05
        Significance level. Supported values: 0.10, 0.05, 0.01.
 
    Returns
    -------
    dict[int, CUSUMResult]
        Keyed by moment order. Each value is a :class:`CUSUMResult` dataclass
        containing the test statistic, critical value, rejection decision,
        conservative p-value bound, break-point location, and the full CUSUM
        path as a ``pd.Series`` aligned to the input index.
 
    Raises
    ------
    ValueError
        If ``returns`` contains NaN values, ``alpha`` is not supported, or
        an unsupported moment order is requested.
 
    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = pd.Series(rng.standard_normal(500), name="returns")
    >>> results = cusum_moment_stability(r)
    >>> results[2].reject
    False
    >>> results[4].reject
    False
    """
    # --- Input validation ---
    if moments is None:
        moments = [2, 4]
 
    unsupported = set(moments) - {2, 4}
    if unsupported:
        raise ValueError(f"Unsupported moment order(s): {unsupported}. Use 2 or 4.")
 
    if alpha not in _CUSUM_SQ_CV:
        raise ValueError(
            f"alpha={alpha} not supported. Choose from {sorted(_CUSUM_SQ_CV)}."
        )
 
    if returns.isna().any():
        raise ValueError(
            "returns contains NaN values. Run impute_missing() before testing."
        )
 
    cv = _CUSUM_SQ_CV[alpha]
    r = returns.to_numpy(dtype=float)
    T = len(r)
    results: dict[int, CUSUMResult] = {}
 
    for m in moments:
        # --- Build the powered series and its cumulative sum ---
        x = r ** m                          # shape (T,)
        cumsum = np.cumsum(x)               # S_k numerator
        total = cumsum[-1]                  # sum over full sample
 
        if total == 0:
            raise ValueError(
                f"Sum of r^{m} is zero — cannot standardise the CUSUM path."
            )
 
        # Standardised path: S_k in [0, 1]
        S = cumsum / total                  # shape (T,)
 
        # Expected linear trend under stability
        k = np.arange(1, T + 1)
        trend = k / T                       # shape (T,)
 
        # Deviations and scaled test statistic
        deviations = np.abs(S - trend)      # shape (T,)
        bp_idx = int(np.argmax(deviations))
        stat = float(np.sqrt(T) * deviations[bp_idx])
 
        reject = stat > cv
 
        # Conservative p-value bound from the three tabulated levels
        if stat > _CUSUM_SQ_CV[0.01]:
            pvalue_bound = "< 0.01"
        elif stat > _CUSUM_SQ_CV[0.05]:
            pvalue_bound = "< 0.05"
        elif stat > _CUSUM_SQ_CV[0.10]:
            pvalue_bound = "< 0.10"
        else:
            pvalue_bound = ">= 0.10"
 
        results[m] = CUSUMResult(
            moment=m,
            statistic=round(stat, 6),
            critical_value=cv,
            alpha=alpha,
            reject=reject,
            pvalue_bound=pvalue_bound,
            breakpoint_index=bp_idx,
            breakpoint_label=returns.index[bp_idx],
            cusum_path=pd.Series(S, index=returns.index, name=f"CUSUM_r^{m}"),
        )
 
    return results

# ---------------------------------------------------------------------------
# Unit root tests
# ---------------------------------------------------------------------------

@dataclass
class UnitRootResult:
    """Results container for unit root tests."""
    unitroot_test: str  # unit root test being implemented: adfuller or kpss
    statistic: float    # test statistic
    critical_value: float   # asymptotic c.v. at requested alpha
    alpha: float        # significance level used
    reject: bool        # Whether the null is rejected at the specified significance level
    pvalue_bound: str    # conservative bound string, e.g. "< 0.01"
    lags: int           # the number of lags used

def unitroot_test(series: pd.Series,
                  alpha: float = 0.05,
                  regression_components: str = 'c') -> dict[str, UnitRootResult]:
    """
    Perform unit root analysis using ADF and KPSS tests.

    This function wraps the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) 
    tests to assess the stationarity of a time series. By combining both, one can distinguish 
    between a series that is stationary, has a unit root, or is trend-stationary.

    Parameters
    ----------
    series : pd.Series
        The univariate time series to be tested. Must be free of missing values.
    alpha : float, default 0.05
        Significance level for the tests. Supported values are 0.01, 0.05, and 0.10.
    regression_components : str, default 'c'
        The regression component to include in the tests. 
        - 'c': Constant (intercept) only.
        - 'ct': Constant and linear time trend.

    Returns
    -------
    dict[str, UnitRootResult]
        A dictionary containing the test results, keyed by the test name ('adfuller', 'kpss').
        Each value is a UnitRootResult dataclass containing the statistic, critical value, 
        rejection decision, p-value bound, and lag order used.

    Raises
    ------
    ValueError
        If `regression_components` is not 'c' or 'ct'.
        If `alpha` is not one of the supported values (0.01, 0.05, 0.10).
        If the input `series` contains any NaN values.

    Notes
    -----
    The tests have opposite null hypotheses:
    - ADF: H0 = Unit Root (Non-stationary). Rejection implies stationarity.
    - KPSS: H0 = Trend Stationary. Rejection implies a unit root.
    
    A robust conclusion of stationarity is reached when ADF rejects H0 and KPSS fails to reject H0.
    """
    
    # --- Input Validation ---
    unsupported = set(regression_components) - {'c', 'ct'}
    
    if unsupported:
        raise ValueError(
            f"Unsupported regression components {regression_components}. Must be either 'c' or 'ct'."    
        )
    
    if alpha not in (0.01,0.05,0.10):
        raise ValueError(
            f"alpha={alpha} not supported. Choose from {sorted([0.01,0.05,0.10])}."
        )
    
    if series.isna().any():
        raise ValueError(
            "series contains NaN values. Run impute_missing() before testing."
        )
        
    alpha_p = f"{int(alpha*100)}%"
    results: dict[str,UnitRootResult] = {}

    # adfuller test wrapper:
    adf_results = adfuller(series, regression=regression_components, autolag='AIC')
    adf_stat = adf_results[0]
    adf_cv = adf_results[4][alpha_p]
    # Conservative p-value bound from the three tabulated levels
    if adf_stat < adf_results[4]['1%']:
        adf_pvalue_bound = "< 0.01"
    elif adf_stat < adf_results[4]['5%']:
        adf_pvalue_bound = "< 0.05"
    elif adf_stat < adf_results[4]['10%']:
        adf_pvalue_bound = "< 0.10"
    else:
        adf_pvalue_bound = ">= 0.10"
    
    adf_reject = adf_stat < adf_cv
    
    results['adfuller'] = UnitRootResult(
        unitroot_test = 'adfuller',
        statistic = round(adf_stat,6),
        critical_value = adf_cv,
        alpha=alpha,
        reject = adf_reject,
        pvalue_bound = adf_pvalue_bound,
        lags = adf_results[2]
        )
    
    # kpss test wrapper
    kpss_results = kpss(series, regression=regression_components, nlags='auto')
    kpss_stat = kpss_results[0]
    kpss_cv = kpss_results[3][alpha_p]
    # Conservative p-value bound from the three tabulated levels
    if kpss_stat > kpss_results[3]['1%']:
        kpss_pvalue_bound = "< 0.01"
    elif kpss_stat > kpss_results[3]['5%']:
        kpss_pvalue_bound = "< 0.05"
    elif kpss_stat > kpss_results[3]['10%']:
        kpss_pvalue_bound = "< 0.10"
    else:
        kpss_pvalue_bound = ">= 0.10"
    
    kpss_reject = kpss_stat > kpss_cv
    
    results['kpss'] = UnitRootResult(
        unitroot_test = 'kpss',
        statistic = round(kpss_stat,6),
        critical_value = kpss_cv,
        alpha=alpha,
        reject = kpss_reject,
        pvalue_bound = kpss_pvalue_bound,
        lags = kpss_results[2]
        )
    
    return results

# ---------------------------------------------------------------------------
# Sample mean significance test (Newey-West HAC)
# ---------------------------------------------------------------------------
 
@dataclass
class MeanTestResult:
    """Results container for the HAC mean significance test."""
 
    mean: float           # sample mean
    se_hac: float         # Newey-West HAC standard error
    t_stat: float         # t-statistic: mean / se_hac
    pvalue: float         # two-sided p-value from N(0,1) asymptotic distribution
    ci_lower: float       # lower bound of (1 - alpha) CI on the mean
    ci_upper: float       # upper bound of (1 - alpha) CI on the mean
    lags: int             # number of Newey-West lags used
    reject: bool          # True → reject H0: mu = 0 at given alpha
    alpha: float          # significance level used
 
 
def test_mean_significance(
    returns: pd.Series,
    lags: int | None = None,
    alpha: float = 0.05,
) -> MeanTestResult:
    """
    Test whether the mean of a return series is significantly different from
    zero using a Newey-West HAC t-test, robust to serial correlation and
    conditional heteroskedasticity.
 
    The HAC variance of the sample mean is estimated as:
 
        V_HAC = (1/n) * [ gamma_0 + 2 * sum_{j=1}^{L} w_j * gamma_j ]
 
    where gamma_j is the j-th sample autocovariance of the demeaned returns,
    w_j = 1 - j/(L+1) are the Bartlett weights, and L is the lag truncation.
    The test statistic is:
 
        t = x_bar / sqrt(V_HAC)  ->  N(0, 1)  as  n -> inf
 
    The automatic lag selection follows Newey & West (1994):
 
        L = floor( 4 * (n/100)^{2/9} )
 
    Parameters
    ----------
    returns : pd.Series
        Return series, free of NaN values.
    lags : int or None, default None
        Number of lags for the Bartlett kernel. If None, the Newey-West
        (1994) automatic selector is used.
    alpha : float, default 0.05
        Significance level for the two-sided test and confidence interval.
 
    Returns
    -------
    MeanTestResult
        Dataclass with the sample mean, HAC standard error, t-statistic,
        two-sided p-value, confidence interval, lag count, and rejection
        decision.
 
    Raises
    ------
    ValueError
        If returns contains NaN values or fewer than 2 observations.
 
    References
    ----------
    Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
        heteroskedasticity and autocorrelation consistent covariance matrix.
        Econometrica, 55(3), 703-708.
    Newey, W. K., & West, K. D. (1994). Automatic lag selection in covariance
        matrix estimation. Review of Economic Studies, 61(4), 631-653.
 
    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = pd.Series(rng.standard_normal(1000), name="returns")
    >>> res = test_mean_significance(r)
    >>> res.reject   # H0: mu = 0 unlikely to be rejected for pure noise
    False
    """
 
    if returns.isna().any():
        raise ValueError(
            "returns contains NaN values. Run impute_missing() before testing."
        )
    n = len(returns)
    if n < 2:
        raise ValueError("Need at least 2 observations.")
 
    r    = returns.to_numpy(dtype=float)
    xbar = float(np.mean(r))
    u    = r - xbar          # demeaned residuals
 
    # --- Lag selection ---
    if lags is None:
        L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    else:
        if lags < 0:
            raise ValueError(f"lags must be non-negative, got {lags}.")
        L = lags
 
    # --- Newey-West HAC variance of the mean ---
    # gamma_0: variance term
    gamma0 = float(np.dot(u, u) / n)
    hac_var = gamma0
 
    # Bartlett-weighted autocovariances
    for j in range(1, L + 1):
        gamma_j = float(np.dot(u[j:], u[:-j]) / n)
        w_j     = 1.0 - j / (L + 1)        # Bartlett weight
        hac_var += 2.0 * w_j * gamma_j
 
    # Variance of the sample mean
    se_hac = float(np.sqrt(hac_var / n))
 
    # --- Test statistic and p-value ---
    t_stat = float(xbar / se_hac)
    pvalue = float(2 * (1 - _norm.cdf(abs(t_stat))))
    reject = pvalue < alpha
 
    # --- Confidence interval ---
    z        = _norm.ppf(1 - alpha / 2)
    ci_lower = xbar - z * se_hac
    ci_upper = xbar + z * se_hac
 
    return MeanTestResult(
        mean=round(xbar, 8),
        se_hac=round(se_hac, 8),
        t_stat=round(t_stat, 6),
        pvalue=round(pvalue, 6),
        ci_lower=round(ci_lower, 8),
        ci_upper=round(ci_upper, 8),
        lags=L,
        reject=reject,
        alpha=alpha,
    )
 
# ---------------------------------------------------------------------------
# Hill estimator & asymptotic Wald test for the tail index
# ---------------------------------------------------------------------------
 
@dataclass
class HillResult:
    """Results container for the Hill tail index estimator and Wald test."""
 
    tail: str              # "right" or "left"
    k: int                 # number of order statistics used
    gamma: float           # Hill estimator of tail exponent (1 / alpha)
    tail_index: float      # alpha = 1 / gamma  (the tail index itself)
    se: float              # asymptotic std error of gamma:  gamma / sqrt(k)
    ci_lower: float        # lower bound of (1 - test_alpha) CI on tail_index
    ci_upper: float        # upper bound of (1 - test_alpha) CI on tail_index
 
    # Wald test: H0: alpha == alpha0
    alpha0: float          # null hypothesis value
    wald_stat: float       # (gamma - gamma0)^2 / Var(gamma) ~ chi2(1)
    pvalue: float          # p-value from chi2(1) distribution
    reject: bool           # True → reject H0 at test_alpha level
    test_alpha: float      # significance level of the test / CI
 
 
def hill_estimator(
    returns: pd.Series,
    k: int | None = None,
    alpha0: float = 2.0,
    test_alpha: float = 0.05,
) -> dict[str, HillResult]:
    """
    Estimate the tail index of a return series using the Hill (1975) estimator,
    separately for the right and left tails, with an asymptotic Wald test.
 
    The Hill estimator of the tail exponent gamma = 1/alpha is:
 
        gamma_k = (1/k) * sum_{i=1}^{k} [ log X_{(n-i+1)} - log X_{(n-k)} ]
 
    where X_{(1)} <= ... <= X_{(n)} are the order statistics of the tail
    observations, and k is the number of upper order statistics used.
    The tail index is then alpha = 1 / gamma_k.
 
    Asymptotic distribution (Hall, 1982):
 
        sqrt(k) * (gamma_k - gamma)  ->  N(0, gamma^2)
 
    so  SE(gamma_k) = gamma_k / sqrt(k),  and the Wald statistic for
    H0: alpha = alpha0  (equivalently gamma = 1/alpha0) is:
 
        W = (gamma_k - gamma0)^2 / (gamma_k^2 / k)  ~  chi2(1)
 
    Parameters
    ----------
    returns : pd.Series
        Return series, free of NaN values. The sign of each observation
        determines which tail it contributes to.
    k : int or None, default None
        Number of upper order statistics to use for each tail. If None,
        defaults to floor(sqrt(n)) where n is the number of observations
        in that tail. Must satisfy 2 <= k < n_tail.
    alpha0 : float, default 2.0
        Null hypothesis value for the tail index in the Wald test.
        alpha0=2 tests for finite variance; alpha0=4 tests for finite kurtosis.
    test_alpha : float, default 0.05
        Significance level for both the Wald test and the confidence interval.
 
    Returns
    -------
    dict[str, HillResult]
        Keys are "right" and "left". Each value is a :class:`HillResult`
        dataclass with the estimator, standard error, confidence interval,
        and Wald test outcome.
 
    Raises
    ------
    ValueError
        If returns contains NaN values, k is out of range, or a tail has
        fewer than 2 observations.
 
    Notes
    -----
    The left tail is analysed by reflecting the negative returns:
    x = -r for all r < 0, so the estimator always operates on positive values.
    The two tails are estimated independently; no symmetry is assumed.
 
    References
    ----------
    Hill, B. M. (1975). A simple general approach to inference about the
        tail of a distribution. Annals of Statistics, 3(5), 1163-1174.
    Hall, P. (1982). On some simple estimates of an exponent of regular
        variation. Journal of the Royal Statistical Society B, 44(1), 37-42.
 
    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(42)
    >>> r = pd.Series(rng.standard_t(df=3, size=1000), name="returns")
    >>> res = hill_estimator(r)
    >>> round(res["right"].tail_index, 2)
    3.12
    >>> res["right"].reject   # H0: alpha=2 rejected?
    True
    """
    
    if returns.isna().any():
        raise ValueError(
            "returns contains NaN values. Run impute_missing() before estimating."
        )
 
    r = returns.to_numpy(dtype=float)
    z_crit = _chi2.ppf(1 - test_alpha, df=1)   # chi2(1) critical value
 
    results: dict[str, HillResult] = {}
 
    tail_specs = {
        "right": r[r > 0],
        "left":  -r[r < 0],   # reflect so we always work with positives
    }
 
    for tail_name, x in tail_specs.items():
        n = len(x)
        if n < 2:
            raise ValueError(
                f"Fewer than 2 observations in the {tail_name} tail (got {n})."
            )
 
        # --- k selection ---
        if k is None:
            k_use = max(2, int(np.floor(np.sqrt(n))))
        else:
            if not (2 <= k < n):
                raise ValueError(
                    f"k={k} is out of range for the {tail_name} tail "
                    f"(must satisfy 2 <= k < {n})."
                )
            k_use = k
 
        # --- Order statistics: sort ascending, take top k+1 ---
        x_sorted  = np.sort(x)                  # X_(1) <= ... <= X_(n)
        threshold = x_sorted[-(k_use + 1)]      # X_(n-k)
        top_k     = x_sorted[-k_use:]           # X_(n-k+1), ..., X_(n)
 
        # --- Hill estimator ---
        gamma      = float(np.mean(np.log(top_k) - np.log(threshold)))
        tail_index = 1.0 / gamma
 
        # --- Asymptotic SE and CI on tail_index via delta method ---
        # SE(gamma)     = gamma / sqrt(k)
        # SE(1/gamma)   = (1/gamma^2) * SE(gamma) = tail_index / sqrt(k)
        se_gamma     = gamma / np.sqrt(k_use)
        se_tailindex = tail_index / np.sqrt(k_use)
 
        z        = _norm.ppf(1 - test_alpha / 2)
        ci_lower = tail_index - z * se_tailindex
        ci_upper = tail_index + z * se_tailindex
 
        # --- Wald test: H0: alpha = alpha0  <=>  gamma = gamma0 = 1/alpha0 ---
        gamma0    = 1.0 / alpha0
        wald_stat = float((gamma - gamma0) ** 2 / (gamma ** 2 / k_use))
        pvalue    = float(1 - _chi2.cdf(wald_stat, df=1))
        reject    = wald_stat > z_crit
 
        results[tail_name] = HillResult(
            tail=tail_name,
            k=k_use,
            gamma=round(gamma, 6),
            tail_index=round(tail_index, 6),
            se=round(se_gamma, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            alpha0=alpha0,
            wald_stat=round(wald_stat, 6),
            pvalue=round(pvalue, 6),
            reject=reject,
            test_alpha=test_alpha,
        )
 
    return results

# ---------------------------------------------------------------------------
# Kernel density estimator with Normal comparison plot
# ---------------------------------------------------------------------------
 
@dataclass
class KDEResult:
    """Results container for the Gaussian KDE of a return series."""
 
    bandwidth: float         # Silverman bandwidth h
    grid: np.ndarray         # evaluation grid, shape (n_grid,)
    density: np.ndarray      # KDE density values on grid, shape (n_grid,)
    normal_density: np.ndarray  # N(mu, sigma^2) density on same grid
    figure: object           # matplotlib Figure
 
 
def kde_returns(
    returns: pd.Series,
    n_grid: int = 512,
    n_std: float = 4.0,
) -> KDEResult:
    """
    Estimate the PDF of a return series via a Gaussian kernel density estimator
    and produce a figure comparing it against the fitted normal distribution.
 
    The bandwidth is selected using Silverman's rule of thumb:
 
        h = 0.9 * min(sigma_hat, IQR/1.34) * n^{-1/5}
 
    where sigma_hat is the sample standard deviation and IQR is the
    interquartile range of the returns. The KDE is then:
 
        f_hat(x) = (1/nh) * sum_{i=1}^{n} K((x - r_i) / h)
 
    with K(u) = (1/sqrt(2*pi)) * exp(-u^2 / 2)  (Gaussian kernel).
 
    The reference normal distribution is N(mu, sigma^2) with mu and sigma
    estimated from the sample, plotted on the same grid for direct comparison.
 
    Parameters
    ----------
    returns : pd.Series
        Return series, free of NaN values.
    n_grid : int, default 512
        Number of equally spaced points on the evaluation grid. Powers of 2
        are preferred for potential FFT-based extensions.
    n_std : float, default 4.0
        Half-width of the evaluation grid in units of sample standard
        deviations, centred on the sample mean.
 
    Returns
    -------
    KDEResult
        Dataclass containing the bandwidth, evaluation grid, KDE density
        values, fitted normal density values, and the matplotlib Figure.
 
    Raises
    ------
    ValueError
        If returns contains NaN values or fewer than 2 observations.
 
    References
    ----------
    Silverman, B. W. (1986). Density Estimation for Statistics and Data
        Analysis. Chapman & Hall, London.  (Rule of thumb: p. 45.)
    """
    
    if returns.isna().any():
        raise ValueError(
            "returns contains NaN values. Run impute_missing() before calling kde_returns()."
        )
    if len(returns) < 2:
        raise ValueError("Need at least 2 observations to estimate a density.")
 
    r = returns.to_numpy(dtype=float)
    n = len(r)
    mu    = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    iqr   = float(np.percentile(r, 75) - np.percentile(r, 25))
 
    # --- Silverman bandwidth ---
    scale = min(sigma, iqr / 1.34)
    h     = 0.9 * scale * n ** (-1 / 5)
 
    # --- Evaluation grid ---
    x_lo = mu - n_std * sigma
    x_hi = mu + n_std * sigma
    grid = np.linspace(x_lo, x_hi, n_grid)
 
    # --- Gaussian KDE  (vectorised: shape (n_grid, n)) ---
    u       = (grid[:, None] - r[None, :]) / h      # (n_grid, n)
    density = np.mean((1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2), axis=1) / h
 
    # --- Reference normal density ---
    normal_density = _norm.pdf(grid, loc=mu, scale=sigma)
 
    # --- Figure ---
    fig, ax = plt.subplots(figsize=(9, 5))
 
    ax.plot(grid, density,        color="#2563EB", lw=2,   label="KDE (Gaussian kernel)")
    ax.plot(grid, normal_density, color="#DC2626", lw=1.5, linestyle="--",
            label=rf"Normal ($\mu$={mu:.3f}, $\sigma$={sigma:.3f})")
    ax.fill_between(grid, density, alpha=0.10, color="#2563EB")
 
    ax.set_xlabel("Return", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Kernel Density Estimate vs Normal — {returns.name or 'returns'}\n"
        f"Silverman bandwidth $h$ = {h:.5f},  $n$ = {n:,}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
 
    return KDEResult(
        bandwidth=round(h, 8),
        grid=grid,
        density=density,
        normal_density=normal_density,
        figure=fig,
    )
 