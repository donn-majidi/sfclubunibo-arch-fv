import warnings
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
                 forecast_horizon: int, alpha: float, align: str | None = 'origin'):

        self._endog = array_like(endog, 'endog', ndim=2)
        self._models = array_like(models, 'models', ndim=1, dtype=object)
        self._window_size = int_like(window_size, 'window_size')
        self._forecast_horizon = int_like(forecast_horizon, 'forecast_horizon')
        self._alpha = float_like(alpha, 'alpha')
        self._align = string_like(align, 'align')
        self._nobs = endog.shape[0]
        self._nmodels = len(models)
        
        if isinstance(endog, pd.Series) or isinstance(endog, pd.DataFrame):
            self._index = endog.index
        else:
            self._index = np.arange(len(endog))
        
        ## Note: The indices of the output point to the information set used
        ## for producing the output, which is I_{t-1}, while the output itself
        ## is the projection of the model at time t.
        
        ## Input validation
        if np.any(np.isnan(endog)):
            raise ValueError('Input data contains NaN values.')
            
        if not np.all([isinstance(md, _ARCHModel) for md in models]):
            raise Exception('Model inputs should be instances of the ARCHModel class '
                            'initiated by the arch_model() model constructor.')
            
        if window_size >= self._nobs:
            raise ValueError('The length of the window must be strictly less than the '
                             'number of observations.')

        if window_size + forecast_horizon >= self._nobs:
            raise ValueError('window_size + forecast_horizon must be strictly less than '
                             'the number of observations.')
            
        if alpha > 1 or alpha < 0:
            raise ValueError('The value of alpha should be strictly between 0 and 1.')
            
        if align not in ('target', 'origin'):
            raise ValueError(f'Invalid index align input: "{align}". Choose either "origin" '
                             'for "origin" index alignment or "target" for target index alignment. '
                             'For more information refer to the docstring or the repo README.md file.')
        
        ## Initiate class params
        self.forecasts = None
        self.std_residuals = None
        self.model_fits = None
        self.MSE = None
        self.QLIKE = None
        self.MAE = None
        self.VaR = None
        self.ES = None
        
        ## Run the validation process
        self._validate()
        self._compute_loss()
        
        ## Prepare the output
        self._to_pandas()
        
    def __str__(self):
        columns = [f'model_{i} ({md.volatility.name})' for i, md in enumerate(self._models)]

        summary = pd.DataFrame(
            {
                'MSE': np.asarray(self.MSE.mean(axis=0)),
                'MAE': np.asarray(self.MAE.mean(axis=0)),
                'QLIKE': np.asarray(self.QLIKE.mean(axis=0)),
            },
            index=columns,
        ).round(6)

        header = (
            f"Validator: {len(self.forecasts)} evaluated window(s), "
            f"{self._nmodels} model(s), forecast_horizon={self._forecast_horizon}"
        )
        return f"{header}\n{summary.to_string()}"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(nobs={self._nobs}, nmodels={self._nmodels}, "
            f"window_size={self._window_size}, forecast_horizon={self._forecast_horizon})"
        )

    def _validate(self):

        models = self._models
        window_size = self._window_size
        nobs = self._nobs
        nmodels = self._nmodels
        forecast_horizon = self._forecast_horizon
        alpha = self._alpha
        
        ## Effective range: T - W - h + 1
        eff_range = nobs - window_size - forecast_horizon + 1
        
        forecasts = np.zeros((eff_range, nmodels))
        std_residuals = np.zeros((eff_range, nmodels ))
        
        VaR = np.zeros((eff_range, nmodels))
        ES = np.zeros((eff_range, nmodels))
        
        model_fits = []
        
        for j, md in enumerate(models):
            _method = 'analytic'
            
            for i in range(eff_range):
                (nobs - window_size)
                _fit = md.fit(first_obs=i, last_obs=window_size + i, 
                              options={'maxiter': 500}, disp=False)
                
                ## Try/Except handle for when analytical forecasts for horizon>1
                ## do not exist. I prefer the bootstrap method, maybe later change
                ## so that user can choose.
                try:
                    _fc = _fit.forecast(horizon=forecast_horizon, method=_method)
                except ValueError:
                    warnings.warn(
                        'Failed to produce analytic forecasts. Falling back to '
                        f'boot strap forecasts for model {md.volatility.name} at '
                        f'horizon={forecast_horizon}.',
                        stacklevel=2
                        )
                    _method = 'bootstrap'
                    _fc = _fit.forecast(horizon=forecast_horizon, method=_method)
                
                _fh = _fc.variance.iloc[0,-1]
                _std = _fit.std_resid.loc[_fit.std_resid.last_valid_index()]
                
                forecasts[i,j] = _fh
                std_residuals[i,j] = _std
                
                ## Next retrieve distribution parameters for VaR and ES estimation
                if md.distribution.num_params == 0:
                    dist_params = None
                else:
                    dist_nparams = md.distribution.num_params
                    dist_params = _fit.params[-dist_nparams:]
                    
                _z_alpha = md.distribution.ppf(alpha, parameters=dist_params)
                _VaR = _z_alpha * np.sqrt(_fh)
                _ES = md.distribution.partial_moment(1, _z_alpha, parameters=dist_params) / alpha * np.sqrt(_fh)
                
                VaR[i,j] = _VaR
                ES[i,j] = _ES
                
            ## Save the fitted model at the last iteration
            model_fits.append(_fit)
            
        self.forecasts = forecasts
        self.std_residuals = std_residuals
        self.VaR = VaR
        self.ES = ES
        self.model_fits = model_fits
        
    def _compute_loss(self):

        window_size = self._window_size
        forecast_horizon = self._forecast_horizon

        ## Align observations and forecasts for correct loss evaluation
        ## The first index that corresponds to the first target forecast is 
        ## Window_Size + Forecast_horizon - 1
        first_indx = window_size + forecast_horizon - 1
        endog = self._endog[first_indx:]
        forecasts = self.forecasts

        self.MSE = (forecasts - endog**2)**2
        self.QLIKE = ( np.log(forecasts) + endog**2/forecasts )
        self.MAE = np.abs(forecasts - endog**2)
        
    def _to_pandas(self):

        index = self._index
        window_size = self._window_size
        forecast_horizon = self._forecast_horizon
        align = self._align
        
        ### Index alignment
        if align == 'origin':
            fc_index = index[ (window_size - 1) : -forecast_horizon]
        else:
            fc_index = index[ (window_size - 1 + forecast_horizon) :]
        
        std_index = index[window_size - 1: -forecast_horizon]
        
        ## Forecasts
        df = pd.DataFrame(self.forecasts, index=fc_index)
        self.forecasts = df
        
        ## Standardized residuals
        df = pd.DataFrame(self.std_residuals, index=std_index)
        self.std_residuals = df
        
        ## VaR
        df = pd.DataFrame(self.VaR, index=fc_index)
        self.VaR = df
        
        ## Expected Shortfall
        df = pd.DataFrame(self.ES, index=fc_index)
        self.ES = df
        
        ## MSE
        df = pd.DataFrame(self.MSE, index=fc_index)
        self.MSE = df
        
        ## QLIKE
        df = pd.DataFrame(self.QLIKE, index=fc_index)
        self.QLIKE = df
        
        ## MAE
        df = pd.DataFrame(self.MAE, index=fc_index)
        self.MAE = df
        
        
        
            