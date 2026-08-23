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
    '''
    Rolling-window forecast-validation harness for one or more fitted
    conditional-volatility (ARCH/GARCH-family) models.

    For each model, refits the model's parameters at every position of a
    fixed-length rolling window sliding one observation at a time across
    ``endog``, produces the ``forecast_horizon``-step-ahead conditional
    variance forecast from each window, and records the standardized
    residual at the newest in-window observation. Forecasts are then
    paired with their realized (squared-return) counterpart position by
    position, and MSE, QLIKE, and MAE loss series are computed per model.

    Parameters
    ----------
    endog : np.ndarray | pd.Series | pd.DataFrame
        Return series (e.g. demeaned log returns) the models were built
        on. Must not contain NaNs.
    models : np.ndarray
        1-D array of already-constructed, unfit ``arch_model`` instances
        (subclasses of ``arch.univariate.base.ARCHModel``), one per model
        to validate. Each instance is refit in place at every rolling
        window.
    window_size : int
        Number of observations in each rolling estimation window. Must be
        strictly less than ``len(endog)``.
    forecast_horizon : int
        Forecast horizon h to evaluate; only the h-step-ahead forecast
        from each window is kept (intermediate 1..h-1 step forecasts are
        discarded). ``window_size + forecast_horizon`` must be strictly
        less than ``len(endog)``, since the last ``forecast_horizon``
        windows have no realized value to evaluate against and are
        dropped from the output.

    Attributes
    ----------
    forecasts : pd.DataFrame
        h-step-ahead conditional variance forecast per model (columns)
        and rolling window origin (rows, indexed like ``endog``).
    std_residuals : pd.DataFrame
        Standardized residual at the newest in-window observation, per
        model and window origin.
    model_results : list of ARCHModelResult
        The final (last window) fit result object for each model, in the
        same order as ``models``.
    MSE, QLIKE, MAE : pd.DataFrame
        Pointwise loss series per model, comparing ``forecasts`` against
        the realized squared return ``forecast_horizon`` steps ahead of
        each window origin.

    Raises
    ------
    ValueError
        If ``endog`` contains NaNs, if ``window_size >= len(endog)``, or
        if ``window_size + forecast_horizon >= len(endog)``.
    Exception
        If any element of ``models`` is not an ``ARCHModel`` instance.
    '''

    def __init__(self, endog: np.ndarray, models: np.ndarray, window_size: int,
                 forecast_horizon: int):
        """Validate inputs and run the rolling-window estimation/forecast/loss pipeline.

        See the class docstring for parameter and attribute descriptions.
        """

        self._endog = array_like(endog, 'endog', ndim=2)
        self._models = array_like(models, 'models', ndim=1, dtype=object)
        self._window_size = int_like(window_size, 'window_size')
        self._forecast_horizon = int_like(forecast_horizon, 'forecast_horizon')
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
            
            
        
        ## Initiate class params
        self._model_ids = None
        self.forecasts = None
        self.std_residuals = None
        self.model_results = None
        self.MSE = None
        self.QLIKE = None
        self.MAE = None
        
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
        """Run the rolling-window re-estimation and forecasting loop.

        For each model, iterates over every window position, refits the model
        (warm-started from the previous window's fitted parameters), and
        forecasts ``forecast_horizon`` steps ahead. Falls back from analytic to
        bootstrap forecasts (with a one-time warning per model) where an
        analytic multi-step forecast isn't available (e.g. EGARCH, APARCH at
        horizon > 1). Populates ``self.forecasts``, ``self.std_residuals``
        (raw numpy arrays), and ``self.model_results``.
        """
        models = self._models
        window_size = self._window_size
        nobs = self._nobs
        nmodels = self._nmodels
        forecast_horizon = self._forecast_horizon
            
        forecasts = np.zeros((nobs - window_size, nmodels))
        std_residuals = np.zeros((nobs - window_size, nmodels))
        model_results = []
            
        for j, md in enumerate(models):
            _res = None
            _method = 'analytic'
            for i in range(nobs - window_size ):

                if _res is not None:
                    _start_params = _res.params
                else:
                    _start_params = None

                _res = md.fit(first_obs = i, last_obs = window_size + i + 1,
                              starting_values=_start_params, disp=False)

                ## Try/Except to handle forecasts for horizon>1 when analytical solution 
                ## does not exist. I prefer bootstrap over Monte Carlo. Maybe later change
                ## so that the user can choose.
                try:
                    _fc = _res.forecast(horizon=forecast_horizon, method=_method)
                except ValueError:
                    warnings.warn(
                        f'Analytic forecasts are not available for model {j} '
                        f'({md.volatility.name}) at horizon={forecast_horizon}; '
                        'falling back to bootstrap forecasts.',
                        stacklevel=2,
                    )
                    _method = 'bootstrap'
                    _fc = _res.forecast(horizon=forecast_horizon, method=_method)
                _fh = _fc.variance.iloc[0,-1]
                _std = _res.std_resid.loc[_res.std_resid.last_valid_index()]
                    
                forecasts[i,j] = _fh
                std_residuals[i,j] = _std
                    
            model_results.append(_res)
        
        self.forecasts = forecasts
        self.std_residuals = std_residuals
        self.model_results = model_results
            
    def _compute_loss(self):
        """Align each forecast with its realized target and compute loss series.

        ``arch``'s forecast output is indexed by the estimation window's last
        observation (the origin), not by the target it predicts, so the
        realized squared return is shifted forward by ``forecast_horizon``
        positions before comparison. The trailing ``forecast_horizon`` rows
        (whose targets fall beyond the sample) are dropped from
        ``self.forecasts``, ``self.std_residuals``, and the loss series.
        Populates ``self.MSE``, ``self.QLIKE``, and ``self.MAE``.
        """
        nobs = self._nobs
        window_size = self._window_size
        forecast_horizon = self._forecast_horizon
        n_valid = nobs - window_size - forecast_horizon

        ## A forecast made from the window ending at position `window_size + i` targets
        ## `window_size + i + forecast_horizon`, not `window_size + i` itself (arch's
        ## .forecast() output is indexed by origin, not by target). Shift endog forward
        ## by forecast_horizon to compare each forecast against its actual target, and
        ## drop the trailing forecast_horizon rows, which target observations beyond the
        ## sample (no realized value exists yet to evaluate them against).
        endog = self._endog[-(nobs - window_size):][forecast_horizon:]
        forecasts = self.forecasts[:n_valid]
        std_residuals = self.std_residuals[:n_valid]

        self.forecasts = forecasts
        self.std_residuals = std_residuals

        self.MSE = (forecasts - endog**2)**2
        self.QLIKE = ( np.log(forecasts) + endog**2/forecasts )
        self.MAE = np.abs(forecasts - endog**2)
        
    def _to_pandas(self):
        """Wrap the raw numpy outputs in DataFrames indexed by window origin.

        Applies the same truncation as ``_compute_loss`` to the time index, so
        ``forecasts``, ``std_residuals``, ``MSE``, ``QLIKE``, and ``MAE`` all
        share a common, correctly-aligned index.
        """
        index = self._index
        window_size = self._window_size
        nobs = self._nobs
        forecast_horizon = self._forecast_horizon
        n_valid = nobs - window_size - forecast_horizon

        time_index = index[ - (nobs - window_size): ][:n_valid]
        
        ## Forecasts
        df = pd.DataFrame(self.forecasts, index=time_index)
        self.forecasts = df
        
        ## Standardized residuals
        df = pd.DataFrame(self.std_residuals, index=time_index)
        self.std_residuals = df
        
        ## MSE
        df = pd.DataFrame(self.MSE, index=time_index)
        self.MSE = df
        
        ## QLIKE
        df = pd.DataFrame(self.QLIKE, index=time_index)
        self.QLIKE = df
        
        ## MAE
        df = pd.DataFrame(self.MAE, index=time_index)
        self.MAE = df
        
        
        
            