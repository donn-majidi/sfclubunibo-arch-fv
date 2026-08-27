import sys
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

    def __init__(self, endog: np.ndarray, models: np.ndarray):

        self._endog = array_like(endog, 'endog', ndim=2)
        self._models = array_like(models, 'models', ndim=1, dtype=object)
        self._nobs = endog.shape[0]
        self._nmodels = len(models)
        
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
        
        ## Initiate class params
        self.forecasts = None
        self.std_residuals = None
        self.model_fits = None
        self.mse_loss = None
        self.mae_loss = None
        self.qlike_loss = None
        self.value_at_risk = None
        self.expected_shortfall = None
        
    def __str__(self):
        '''Returns a compact summary of average MSE, MAE, and QLIKE loss per model.'''
        columns = [f'model_{i} ({md.volatility.name})' for i, md in enumerate(self._models)]

        summary = pd.DataFrame(
            {
                'MSE': np.asarray(self.mse_loss.mean(axis=0)),
                'MAE': np.asarray(self.mae_loss.mean(axis=0)),
                'QLIKE': np.asarray(self.qlike_loss.mean(axis=0)),
            },
            index=columns,
        ).round(6)

        header = (
            f"Validator: {len(self.forecasts)} evaluated window(s), "
            f"{self._nmodels} model(s)."
        )
        return f"{header}\n{summary.to_string()}"

    def __repr__(self):
        '''Returns an unambiguous string representation of the instance's shape parameters.'''
        return (
            f"{self.__class__.__name__}(nobs={self._nobs}, nmodels={self._nmodels})"
        )

    def validate(self, window_size: int, horizon: int, alpha: float | None = None,
                 align: str | None = 'origin'):

        models = self._models
        nobs = self._nobs
        nmodels = self._nmodels

        window_size = int_like(window_size, 'window_size')
        horizon = int_like(horizon, 'horizon')
        alpha = float_like(alpha, 'alpha', optional=True)
        align = string_like(align, 'align')
        
        if window_size >= nobs:
            raise ValueError('The length of the window must be strictly less than the '
                             'number of observations.')
            
        if window_size + horizon >= nobs:
            raise ValueError('window_size + forecast_horizon must be strictly less than '
                             'the number of observations.')
            
        if alpha is not None and (alpha >= 1 or alpha <= 0):
            raise ValueError('The value of alpha should be strictly between 0 and 1.')
            
        if align not in ('target', 'origin'):
            raise ValueError(f'Invalid index align input: "{align}". Choose either "origin" '
                             'for "origin" index alignment or "target" for target index alignment. '
                             'For more information refer to the docstring or the repo README.md file.')

        ## Effective range: T - W - h + 1
        eff_range = nobs - window_size - horizon + 1
        
        forecasts = np.zeros((eff_range, nmodels))
        std_residuals = np.zeros((eff_range, nmodels ))
        
        value_at_risk = np.zeros((eff_range, nmodels))
        expected_shortfall = np.zeros((eff_range, nmodels))
        
        model_fits = []
        
        for j, md in enumerate(models):
            sys.stdout.write(f'\nNow fitting model number {j}\n')
            sys.stdout.flush()
            _method = 'analytic'
            _fit = None
            _warned_convergence = False

            for i in range(eff_range):
                sys.stdout.write(".")
                sys.stdout.flush()
                _start_params = _fit.params if _fit is not None else None

                _fit = md.fit(first_obs=i, last_obs=window_size + i,
                              starting_values=_start_params, disp=False)

                ## A warm start from the previous window's params can put the optimizer
                ## in a bad spot right after a sharp volatility swing, so it may fail to
                ## converge. Refit from the model's default starting values with a larger
                ## iteration budget instead of carrying a bad estimate forward as the next
                ## window's warm start.
                if _fit.convergence_flag != 0:
                    if not _warned_convergence:
                        warnings.warn(
                            f'Warm-started fit did not converge for model {j} '
                            f'({md.volatility.name}) at window ending index '
                            f'{window_size + i - 1}; refitting from default starting values.',
                            stacklevel=2,
                        )
                        _warned_convergence = True
                    _fit = md.fit(first_obs=i, last_obs=window_size + i,
                                  starting_values=None, options={'maxiter': 500}, disp=False)

                ## Try/Except handle for when analytical forecasts for horizon>1
                ## do not exist. I prefer the bootstrap method, maybe later change
                ## so that user can choose.
                try:
                    _fc = _fit.forecast(horizon=horizon, method=_method)
                except ValueError:
                    warnings.warn(
                        'Failed to produce analytic forecasts. Falling back to '
                        f'boot strap forecasts for model {md.volatility.name} at '
                        f'horizon={horizon}.',
                        stacklevel=2
                        )
                    _method = 'bootstrap'
                    _fc = _fit.forecast(horizon=horizon, method=_method)
                
                _fh = _fc.variance.iloc[0,-1]
                _std = _fit.std_resid.loc[_fit.std_resid.last_valid_index()]
                
                forecasts[i,j] = _fh
                std_residuals[i,j] = _std
                
                ## VaR and Expected Shortfall estimation
                if alpha is not None:
                    ## Frist retrieve distribution parameters for VaR and ES estimation
                    if md.distribution.num_params == 0:
                        dist_params = None
                    else:
                        dist_nparams = md.distribution.num_params
                        dist_params = _fit.params[-dist_nparams:]
                    
                    _z_alpha = md.distribution.ppf(alpha, parameters=dist_params)
                    _var = _z_alpha * np.sqrt(_fh)
                    _es = md.distribution.partial_moment(1, _z_alpha, parameters=dist_params) / alpha * np.sqrt(_fh)
                
                    value_at_risk[i,j] = _var
                    expected_shortfall[i,j] = _es
                
            ## Save the fitted model at the last iteration
            model_fits.append(_fit)
            
        self.forecasts = forecasts
        self.std_residuals = std_residuals
        self.value_at_risk = value_at_risk
        self.expected_shortfall = expected_shortfall
        self.model_fits = np.asarray(model_fits, dtype=object)
        self.mse_loss = self.compute_loss(forecasts, window_size, horizon, 'mse')
        self.mae_loss = self.compute_loss(forecasts, window_size, horizon, 'mae')
        self.qlike_loss = self.compute_loss(forecasts, window_size, horizon, 'qlike')
        
        ## Prepare the output
        self._to_pandas(window_size, horizon, align)
        
    def compute_loss(self, forecasts: np.ndarray, window_size: int, 
                     horizon: int, loss_function: str):

        forecasts = array_like(forecasts, 'forecasts', ndim=2)
        window_size = int_like(window_size, 'window_size')
        horizon = int_like(horizon, 'horizon')
        loss_function = string_like(loss_function, 'loss_function')

        valid_loss_functions = ('mse', 'MSE', 'mae', 'MAE', 'qlike', 'QLIKE')

        if loss_function not in valid_loss_functions:
            raise ValueError('Unsupported loss function. Must be one of: '
                             f'{valid_loss_functions}')

        ## Align observations and forecasts for correct loss evaluation
        ## The first index that corresponds to the first target forecast is 
        ## Window_Size + Forecast_horizon - 1
        first_indx = window_size + horizon - 1
        endog = self._endog[first_indx:]

        if loss_function in ('mse', 'MSE'):
            loss = (forecasts - endog**2)**2
        elif loss_function in ('mae', 'MAE'):
            loss = np.abs(forecasts - endog**2)
        elif loss_function in ('qlike', 'QLIKE'):
            loss = ( np.log(forecasts) + endog**2/forecasts )
        else:
            raise NotImplementedError('Loss function not implemented.')
        
        return loss

    def _to_pandas(self, window_size: int, horizon: int, align: str):
        '''
        Converts the raw ndarray results from ``validate``/``compute_loss``
        into index-aligned ``pd.DataFrame``s, per the ``align`` setting.
        '''
        index = self._index
        window_size = int_like(window_size, 'window_size')
        horizon = int_like(horizon, 'horizon')
        align = string_like(align, 'align')
        
        ### Index alignment
        if align == 'origin':
            fc_index = index[ (window_size - 1) : -horizon]
        else:
            fc_index = index[ (window_size - 1 + horizon) :]
        
        std_index = index[window_size - 1: -horizon]
        
        ## Forecasts
        df = pd.DataFrame(self.forecasts, index=fc_index)
        self.forecasts = df
        
        ## Standardized residuals
        df = pd.DataFrame(self.std_residuals, index=std_index)
        self.std_residuals = df
        
        ## VaR
        df = pd.DataFrame(self.value_at_risk, index=fc_index)
        self.value_at_risk = df
        
        ## Expected Shortfall
        df = pd.DataFrame(self.expected_shortfall, index=fc_index)
        self.expected_shortfall = df
        
        ## MSE
        df = pd.DataFrame(self.mse_loss, index=fc_index)
        self.mse_loss = df
        
        ## MAE
        df = pd.DataFrame(self.mae_loss, index=fc_index)
        self.mae_loss = df
    
        ## QLIKE
        df = pd.DataFrame(self.qlike_loss, index=fc_index)
        self.qlike_loss = df
