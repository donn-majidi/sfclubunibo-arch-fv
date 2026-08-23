# sfclubunibo-arch-fv
Forecast-Validation for optimal ARCH/GARCH model selection in Python

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Modules](#modules)
- [Workflow](#workflow)
- [Usage Example](#usage-example)
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

See [Requirements](#requirements) for the list of packages this installs.

## Modules

### `Class LossContainer`
Generic class for loss function calculations. It takes as input an array of model forecasts and the observations and it calculates the MSE, MAE, and QLIKE loss of each forecast. The estimated loss series are stored as attributes.
- LossContainer.MSE contains series of squared forecast errors.
- LossContainer.MAE contains series of absolute forecast errors.
- LossContainer.QLIKE contains series of quasi-likelihood forecast scores.

The class can be optionally instantiated with `forecast_horizon` which will include the forecast horizon in the summary results.
```python
from src.modules.loss_container import LossContainer
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

### `hill_test`

### `jb_test`

## Workflow

## Usage Example

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
