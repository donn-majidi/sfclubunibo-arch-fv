import numpy as np
import pandas as pd

from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          bool_like,
                                          float_like,
                                          int_like,
                                          )

class LossContainer:
    '''
    Container computing pointwise forecast-evaluation loss series (MSE, MAE,
    QLIKE) between one or more volatility forecasts and their realized
    observations.

    Parameters
    ----------
    forecasts : pd.Series | pd.DataFrame
        Forecasted (strictly positive) variances, one column per model.
    observations : pd.Series
        Realized (strictly positive) variances, aligned with ``forecasts``.
    forecast_horizon : int | None, optional
        Forecast horizon associated with this set of forecasts. Stored for
        reference only; not used in the loss computations. The default is
        None.

    Attributes
    ----------
    mse_losses : pd.DataFrame | np.ndarray
        Squared-error loss series, (forecast - observation)^2.
    mae_losses : pd.DataFrame | np.ndarray
        Absolute-error loss series, |forecast - observation|.
    qlike_losses : pd.DataFrame | np.ndarray
        Quasi-likelihood loss series, log(forecast) + observation/forecast.
        Returned as a ``pd.DataFrame`` (indexed/columned like ``forecasts``)
        when ``forecasts`` is a pandas object, otherwise as a raw ndarray.

    Raises
    ------
    ValueError
        If ``forecasts`` and ``observations`` have a different number of
        observations, or either contains non-positive values.
    '''

    def __init__(self, forecasts: pd.Series | pd.DataFrame,
                 observations: pd.Series, forecast_horizon: int | None = None):
        
        self._forecasts = array_like(forecasts, 'forecasts', ndim=2)
        self._nobs = self._forecasts.shape[0]
        self._observations = array_like(observations, 'observations', ndim=2)
        self._forecast_horizon = int_like(forecast_horizon, 'forecast_horizon', optional=True)
        
        ## Ensure that the number of observations in forecasts and observations match
        if self._nobs != self._observations.shape[0]:
            raise ValueError('The number of observations in the forecast series '
                             'and the observations do not match.')
            
        ## Ensure that all forecasted and observed variances are strictly positive
        if np.any(self._forecasts <= 0) or np.any(self._observations <= 0):
            raise ValueError('Forecasted series or observations include non-positive values.')
            
        self._index = None
        self._columns = []
        if isinstance(forecasts, pd.DataFrame):
            self._index = forecasts.index
            self._columns = forecasts.columns
        if isinstance(forecasts, pd.Series):
            self._index = forecasts.index
            self._columns = [forecasts.name if forecasts.name is not None else 'forecast']
            
        ## Declare class parameters and attributes
        self.mse_losses = None
        self.mae_losses = None
        self.qlike_losses = None
        
        ## Compute the loss series
        self._compute_loss_series()
        
        ## Prepare the output
        if self._index is not None:
            self._to_pandas()
            
    def __str__(self):
        columns = self._columns if len(self._columns) else [
            f'model_{i}' for i in range(self._forecasts.shape[1])
        ]

        summary = pd.DataFrame(
            {
                'MSE': np.asarray(self.mse_losses.mean(axis=0)),
                'MAE': np.asarray(self.mae_losses.mean(axis=0)),
                'QLIKE': np.asarray(self.qlike_losses.mean(axis=0)),
            },
            index=columns,
        ).round(6)

        header = f"LossContainer: {self._nobs} observations, {len(columns)} model(s)"
        return f"{header}\n{summary.to_string()}"

    def __repr__(self):
        columns = list(self._columns) if len(self._columns) else None
        return (
            f"{self.__class__.__name__}(nobs={self._nobs}, "
            f"columns={columns}, forecast_horizon={self._forecast_horizon})"
        )
    
    def _compute_loss_series(self):
        forecasts = self._forecasts
        observations = self._observations
        
        self.mse_losses = (forecasts - observations)**2
        self.qlike_losses = ( np.log(forecasts) + observations/forecasts )
        self.mae_losses = np.abs(forecasts - observations)
        
    def _to_pandas(self):
        
        index = self._index
        columns = self._columns
        
        ## MSE
        df =  pd.DataFrame(self.mse_losses, index=index, columns=columns)
        self.mse_losses = df
        
        ## MAE
        df = pd.DataFrame(self.mae_losses, index=index, columns=columns)
        self.mae_losses = df
        
        ## QLIKE
        df = pd.DataFrame(self.qlike_losses, index=index, columns=columns)
        self.qlike_losses = df
            
        
        