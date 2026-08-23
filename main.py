import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(-1, './src')
from arch import arch_model
from src.modules.loss_container import Loss_container

plt.rc('figure', figsize=(16,6))
sns.set_style('darkgrid')

## Load the data
returns = None
import arch.data.sp500
data = arch.data.sp500.load()
market = data['Adj Close']
returns = np.log(market).diff().dropna() * 100

## Models
__MODELS__ = []
__DIST__ = ["Normal", "t"]
__FCAST_HORIZONS__ = [1, 4]

## Construct all the possible models
am = arch_model(returns, vol="Garch", p=1, o=0, q=1, dist="Normal")

## Rolling Window Forecasts
indx = returns.index
start_indx = 0
#end_indx = np.where(indx >= '2026-01-01')[0].min()
end_indx = np.where(indx >= '2018-01-01')[0].min()
final_indx = len(returns)
forecast_range = final_indx - end_indx
forecasts = {}
for i in range(1, forecast_range):
    sys.stdout.write(".")
    sys.stdout.flush()
    res = am.fit(first_obs=i, last_obs=i + end_indx, disp="off")
    temp = res.forecast(horizon=1).variance
    fcast = temp.iloc[0]
    forecasts[fcast.name] = fcast
print()
print(pd.DataFrame(forecasts).T)

## Loss Series Computations
forecasts = pd.DataFrame(forecasts).T
observations = returns.iloc[end_indx+1:].pow(2)

forecasts_h1 = forecasts.iloc[:,0]
loss_series = Loss_container(forecasts_h1, observations)

## Bootstrap block size selection

## StepM and MCS

## Diagnostic Tests on Standardized Residuals
## CUSUM

## Moment Conditions

## VaR and ES