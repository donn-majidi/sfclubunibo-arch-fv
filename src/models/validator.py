import scipy
import numpy as np
import pandas as pd
from arch.univariate.base import ARCHModel as _ARCHModel
from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          bool_like,
                                          float_like,
                                          int_like,
                                          )

class Validator:
    
    def __init__(self, endog: np.ndarray, models: np.ndarray, window_size: int,
                 forecast_horizon: int):
        
        self._endog = array_like(endog, 'endog', ndim=2)
        self._models = array_like(models, 'models', ndim=1, dtype=object)
        self._window_size = int_like(window_size, 'window_size')
        self._forecast_horizon = int_like(forecast_horizon, 'forecast_horizon')
        self._nobs = endog.shape[0]
        self._nmodels = models.shape[0]
        
        if isinstance(endog, pd.Series) or isinstance(endog, pd.DataFrame):
            self._index = endog.index
        else:
            self._index = np.arange(len(endog))
        
        ## Input validation
        if np.any(np.isnan(endog)):
            raise ValueError('Input data contains NaN values.')
            
        if not np.all([isinstance(md, _ARCHModel) for md in models]):
            raise Exception('Model inputs should be instances of the ARCHModel class '
                            'initiated by the arch_model() model constructor.')
            
        if window_size >= self._nobs:
            raise ValueError('The length of the window must be strictly less than the '
                             'number of observations.')
            
        self.standardized_residuals = None
        self.forecasts = None
        self.model_params = None
        
        def _validate(self):
            endog = self._endog
            models = self._models
            window_size = self._window_size
            nobs = self._nobs
            nmodels = self._nmodels
            
            forecasts = np.zeros((nobs - window_size, nmodels))
            standardized_residuals = np.zeros((nobs - window_size, nmodels))
            
            for md in models:
                pass
            