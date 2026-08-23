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

## Out-of-sample forecasts along with forecast losses can be accessed from the class attributes
forecasts = model_validator.forecasts
forecasts_mse_loss = model_validator.MSE  ## DataFrame containing squared forecast errors
forecasts_qlike_loss = model_validator.QLIKE  ## DataFrame containing quasi-likelihood scores

## Index-aligned standardized residuals obtained from each iteration can be accessed via
std_residuals = model_validator.std_residuals

## Finally, the model fit results from the last iteration can be accessed from the model_results attribute
mod1_fit = model_validator.model_results[0]
mod2_fit = model_validator.model_results[1]
```
## Models

### `class Validator`

## Extra Modules

### `Class LossContainer`
Generic class for loss function calculations. It takes as input the array of model forecasts and the index-aligned observations and it calculates the MSE, MAE, and QLIKE loss of each forecast. The estimated loss series are stored as attributes.
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
lc = LossContainer(squared_returns, forecasts, forecast_horizon=1)
MSE = lc.MSE
print(MSE)

## average forecast losses
print(lc)
```

### `bootstrap_block_size`
This function implements the automatic block-length selection procedure of Politis & White (2004) / Patton, Politis & White (2009).
- It takes as input a dataframe or a series containing the estimated model losses and calculates the optimal block-size for each column.
```python
from src.modules.bootstrap_params import bootstrap_block_size
print(bootstrap_block_size(lc.MSE))
```
### `cusum_supf_test`
This function is an implementation of Bruce Hansen's CUSUM of Squares SupF test.

Inputs:
- x: Univariate array of standardized residuals.
- moment: Moment order being tested. Has to be either 2 or 4.
- alpha: The size of the test. Default is 0.05.
- trim: Trimming fraction for trimming the CUSUM path.
- bandwidth: Kernel bandwidth for variance estimation.
- ax: plt.Axes canvas on which to plot the graph of the CUSUM path along with the confidence bands.

```python
from src.modules.standard_diagnostics import cusum_supf_test

## 
```
### `hill_test`

### `jb_test`

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
