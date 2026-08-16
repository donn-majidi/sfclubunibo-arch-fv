import numpy as np
import pandas as pd

from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          bool_like,
                                          float_like,
                                          int_like,
                                          )

def bootstrap_block_size(series: pd.DataFrame | pd.Series) -> pd.DataFrame:
    x_series = array_like(series, 'series', ndim=2)
    opt_bs = [_bootstrap_block_size_univariate(col) for col in x_series.T]
    
    if isinstance(series, pd.DataFrame):
        indx = list(series.columns)
    elif isinstance(series, pd.Series):
        indx = [series.name]
    else:
        indx = [i for i in range(x_series.shape[1])]
    
    return pd.DataFrame(opt_bs, index=indx, columns=['Stationary Bootstrap',
                                                     'Circular Bootstrap',
                                                     'Moving-Blocks Bootstrap'])
    
    
def _bootstrap_block_size_univariate(series: np.array) -> tuple[float, float, float]:
    x_series = array_like(series, 'series', ndim=1)
    nobs = x_series.shape[0]
    
    ## First demean the series
    eps = x_series - x_series.mean(axis=0)
    
    ## Define the upper bounds
    b_max = np.ceil(min(3 * np.sqrt(nobs), nobs/3))
    kn = max(5, int(np.log10(nobs)))
    m_max = int(np.ceil(np.sqrt(nobs))) + kn
    
    ## Find the appropriate value for the tuning parameter such that
    ## the first kn autocorrelations are all within the confidence band cv:
    cv = 2 * np.sqrt( np.log10(nobs) / nobs )
    
    ## Declare the acv and abs_acorr variables
    acv = np.zeros(shape = m_max + 1)
    abs_acorr = np.zeros(shape = m_max + 1)
    opt_m = None
    
    for i in range(m_max + 1):
        v1 = eps[i+1:] @ eps[i+1:]
        v2 = eps[:-(i+1)] @ eps[:-(i+1)]
        
        cross_prod = eps[i:] @ eps[:nobs-i]
        acv[i] = cross_prod / nobs
        abs_acorr[i] = np.abs(cross_prod) / np.sqrt(v1 * v2)
        
        ## When there are enough autocorrelations estimated, that is when i >= kn
        ## check whether the remaining autocorrelations are within the confidence band
        ## this way we can deduce how many autocorrealtions are significant. 
        ## Once the conditions are met and the optimal value for the tuning parameter
        ## is found, we break out of the loop.
        if i >= kn:
            if np.all( abs_acorr[i-kn:i] < cv) and opt_m is None:
                opt_m = i - kn
                
    m = 2 * max(opt_m, 1) if opt_m is not None else m_max
    m = min(m, m_max)
    ## m is multiplied by 2 to double the cutoff as suggested by Patton-Politis-White
    ## to widen the window of autocorrelations considered
    
    ## Now the autocovariance-generating function
    g = 0.0
    lr_acv = acv[0]
    
    for k in range(1, m+1):
        h = min(1, 2*(1 - np.abs(k/m)))
        g += 2 * h * k * acv[k]
        lr_acv += 2 *h * acv[k]
    
    b_sb = ( g**2 / lr_acv**2 * nobs ) ** (1/3)
    b_cb = ( (2 * g**2) / (4/3 * lr_acv**2) * nobs ) ** (1/3)
    
    b_sb = min(b_sb, b_max)
    b_cb = b_mb = min(b_cb, b_max)
    
    return b_sb, b_cb, b_mb
        
    
        
        
    