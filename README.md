# sfclubunibo-arch-fv
Forecast-Validation for optimal ARCH/GARCH model selection in Python

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Workflow](#workflow)
- [Usage Example](#usage-example)
- [Models](#models)
- [Extra Modules](#extra-modules)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

## Overview

## Installation
```bash
git clone https://github.com/donn-majidi/sfclubunibo-arch-fv.git
cd sfclubunibo-arch-fv
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

See [Requirements](#requirements) for the list of required packages.

## Workflow

## Usage Example
```python
import numpy as np
import pandas as pd
from src.models.validator import Validator

## The following is an illustrative example using data fetched from yahoo finance.
# pip install yfinance #run this line if yfinance is not already installed.
import yfinance as yf
sp500 = yf.Ticker('^GSPC').history(start='2022-01-01')
returns = np.log(sp500['Close']).diff() * 100 ## Always a good idea to multiply returns by 100
returns = returns.dropna()

## Import arch model constructor
from arch import arch_model
mod1 = arch_model(returns, vol='GARCH', p=1, o=0, q=1, dist='gaussian') ## GARCH(1,1) with Gaussian likelihood function.
mod2 = arch_model(returns, vol='GARCH', p[1=, o=1, q=1, dist='studentst') ## GJR_GARCH(1,1,1) with Studend-t likelihood function.

## Create array of model instances
models = np.asarray([mod1, mod2], dtype=object) ## The dtype of the array has to be explicitly set to object.

## Set the window size and the forecast horizon to feed to the Validator class
ws = 252
fh = 1

model_validator = Validator(endog=returns, models=models, window_size=ws, forecast_horizone=fh)

## Out-of-sample forecasts along with forecast losses can be accessed from the class properties.
forecasts = model_validator.forecasts
forecasts_mse_loss = model_validator.MSE  ## DataFrame containing squared forecast errors
forecasts_qlike_loss = model_validator.QLIKE  ## DataFrame containing quasi-likelihood scores

## Index-aligned standardized residuals obtained from each iteration can be accessed via
zs = model_validator.std_residuals

## Finally, the model fit results from the last iteration can be accessed from the model_results property
mod1_fit = model_validator.model_results[0]
mod2_fit = model_validator.model_results[1]
```
## Models

### `class Validator`
```python
class Validator(endog: np.ndarray,
                models: np.ndarray,
                window_size: int,
                forecast_horizon: int)
```
Model validator class for rolling-window forecast loss evaluations.
#### Parameters
- endog: Return series (e.g. demeaned log returns) the models were built on.
- models: 1-D array of model instances.
- window_size: Number of observations in each rolling estimation window.
- forecast_horizon: Forecast horizon h to evaluate.

#### Properties
- forecasts: h-step-ahead conditional variance forecast per model (columns) and rolling window origin (rows, indexed like ``endog``).
- std_residuals: Standardized residual at the newest in-window observation, per model and window origin.
- model_results: list of ARCHModelResult containing model fit results from the last iteration.
- MSE, QLIKE, MAE: Index-aligned forecast losses obtained from model forecasts.

## Extra Modules

### `Class LossContainer`
Generic class for loss function calculations. It takes as input the array of model forecasts and the index-aligned observations and it calculates the MSE, MAE, and QLIKE loss of each forecast. The estimated loss series are stored as properties.
- LossContainer.MSE contains series of squared forecast errors.
- LossContainer.MAE contains series of absolute forecast errors.
- LossContainer.QLIKE contains series of quasi-likelihood forecast scores.
The class can be optionally instantiated with `forecast_horizon` which will include the forecast horizon in the summary results.

> [!NOTE]
> The Validator class automatically computes the loss series and handles index-alignment internally. Only use the LossContainer class if using an alternative forecasting scheme.
```python
from src.modules.loss_container import LossContainer

## assuming that the array of out-of-sample forecasts is stored in forecasts
## and the univariate array of observations (variance proxies in the case of volatility forecasts) is stored in observations
## then the class should be loaded as:
lc = LossContainer(returns, forecasts, forecast_horizon=1)
f_mse = lc.MSE
print(MSE)

## average forecast losses
print(lc)
```

### `bootstrap_block_size`
This function implements the automatic block-length selection procedure of Politis & White (2004) / Patton, Politis & White (2009).
- It takes as input a dataframe or a series containing the estimated model losses and calculates the optimal block-size for each column.
```python
from src.modules.bootstrap_params import bootstrap_block_size
optimal_bsz = bootstrap_block_size(f_mse)
print(optimal_bsz)
```
### `cusum_supf_test`
This function implements Bruce Hansen's CUSUM of Squares SupF test.


Inputs:
- x: Univariate array of standardized residuals.
- moment: Moment order being tested. Has to be either 2 or 4.
- alpha: The size of the test. Default is 0.05.
- trim: Trimming fraction for trimming the CUSUM path.
- bandwidth: Kernel bandwidth for variance estimation.
- ax: plt.Axes canvas on which to plot the graph of the CUSUM path along with the confidence bands.

```python
from src.modules.standard_diagnostics import cusum_supf_test
zs_0 = zs[0] ## Univariate array of standardized residuals

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

cusum_results = cusum_supf_test(zs_0, moment=2, ax=ax)
print(cusum_results)
plt.show()
```
### `class GenParetoMLE`
This is a child class of statsmodels GenericLikelihoodModel to estimate the shape and scale parameters of a Generalized Pareto Distribution.
```python
class GenParetoMLE(endog: np.ndarray)
```
#### Parameters
- endog: Tail observations for GPD parameters estimation. Must already be centered.
### Properties
- params: Estimated parameters.
  - params[0] contains the estimated shape parameter.
  - params[1] contains the estimated scale parameter.
```python
>>>Example
from src.modules.standard_diagnostics import GenParetoMLE
thresh = np.percentile(zs_0, 95)
ex_0 = zs_0[zs_0 > thresh] - thresh
gp_model = GenParetoMLE(ex_0)
gp_fit = gp_model.fit()
gp_fit.summary()

xi_hat = gp.params[0]
sigma_hat = gp.params[1]
```

### `hill_test`
This function is a generalization of the test for the heavy-tailedness via the Hill estimator. The asymptotic distribution of the shape parameter estimator $\hat{xi}$ minus its hypothesized value $\xi = 1/r$, where $r$ is the moment order being tested is normal with variance $(1+\xi)^2$:
```math
                        \sqrt(m)  (\hat{\xi} - \xi) \sim \mathcal{N}(0,(1+\xi)^2),
```        
where $m$ is the number of exceedances, ovvero the size of the series passed to the function. This motivates the test statistic:
```math
                 w = \sqrt(m)  (\hat{\xi} - \xi) / (1+\xi) \sim \mathcal{N}(0,1)
```
Inputs:
- z: Univariate array of centered exceedances above threshold. Must be the same array fed to GenParetoMLE.
- moment_order: Moment order being tested.
- xi_hat: MLE estimate of the shape parameter.
- sigma_hat: MLE estimate of the scale parameter.
- bandwidth: Optional. Number of bins to plot the histogram.
- trim_quantile: Optional | Default = 0.99. The cut-off quantile on the plot. By trimming the extreme values makes the plot tidier.
- ax: Optional. plt.Axes canvas on which to plot the empirical pdf of the input data along with the theoretical pdf of the Generalized Pareto Distribution with given shape and scale parameters.

Returns:
- Dictionary containing:
  - test statistic
  - critical value
  - p-value of the test
```python
from src.modules.standard_diagnostics import hill_test

## Hill's test
r = 4 ## Moment order being tested
fig, ax = plt.subplots()
hill_results = hill_test(ex_0, moment_order = r, xi_hat = xi_hat, sigma_hat = sigma_hat, ax=ax)
print(hill_results)
plt.show()
```
### `jb_test`
This function implements the Jarque-Bera test for asymptotic normality of the standardized residuals. It takes as input the sample data and returns the Jarque-Bera test statistic along with its associated critical value and the p-value of the test.

The Jarque-Bera test statistic is defined as
```math
                JB = T/6 (\hat{S}^2 + \frac{(\hat{K} - 3)^2}{4} ),
```
where $\hat{S}$ and $\hat{K}$ are sample estimates of Skewness and Kurtosis of the sample data. Under the null hypothesis that the sample data is normally distributed, the JB test statistic has an asymptotic chi-squared distribution with two degrees of freedom:
```math
                JB \sim \mathcal{\chi}^2(2).
```
```python
from src.modules.standard_diagnostics import jb_test
jb_results = jb_test(zs_0)
print(jb_results)
```
## Requirements
- [`numpy>=2.3.0`](https://numpy.org/)
- [`pandas>=2.3.0`](https://pandas.pydata.org/)
- [`scipy>=1.16.0`](https://scipy.org/)
- [`matplotlib>=3.10.0`](https://matplotlib.org/)
- [`seaborn>=0.13.0`](https://seaborn.pydata.org/)
- [`scikit-learn>=1.8.0`](https://scikit-learn.org/)
- [`statsmodels>=0.14.0`](https://www.statsmodels.org/)
- [`arch>=8.0.0`](https://bashtage.github.io/arch/)
## References

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
