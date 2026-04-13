import yfinance as yf
#import pandas as pd
import matplotlib.pyplot as plt
#import src.data_loader
#import src.diagnostics
from src.data_loader import impute_missing
from src.data_loader import log_transform
from src.diagnostics import cusum_moment_stability
from src.diagnostics import unitroot_test
from src.diagnostics import hill_estimator
from src.diagnostics import kde_returns
from src.diagnostics import test_mean_significance
#from statsmodels.tsa.stattools import kpss, adfuller


spx_daily = yf.download(tickers='^GSPC', start='1970-01-01', interval='1d')
spx_daily.loc['1970-01-05'] = None
spx_daily.loc['1980-01-02'] = None
spx_daily.loc['1990-01-02'] = None
spx_daily.loc['1995-03-02'] = None
spx_daily.loc['1992-01-02'] = None
spx_daily.loc['2001-01-02'] = None
spx_daily.loc['2003-01-02'] = None

spx_daily = spx_daily["Close"]

spx_daily_smoothed = impute_missing(spx_daily['^GSPC'])
spx_returns = log_transform(spx_daily_smoothed)

mean_return = spx_returns.mean()
demeaned_spx_returns = spx_returns - spx_returns.mean()

plt.figure(figsize=(18,14))
plt.plot(spx_returns, linewidth = 1.5, color = 'black')
plt.title('S&P 500 Index - Daily Data', weight = 'bold')
#plt.axvspan(spx_returns.loc['1970'].index[0], spx_returns.loc['1970-11'].index[-1], facecolor = 'darkslategrey', alpha = 0.3)
#plt.axvspan(spx_returns.loc['1973-11'].index[0], spx_returns.loc['1975-03'].index[-1], facecolor = 'darkslategrey', alpha = 0.3)
#plt.axvspan(spx_returns.loc['1980-01'].index[0], spx_returns.loc['1980-07'].index[-1], facecolor = 'darkslategrey', alpha = 0.3)
#plt.axvspan(spx_returns.loc['1981-07'].index[0], spx_returns.loc['1982-11'].index[-1], facecolor = 'darkslategrey', alpha = 0.3)
#plt.axvspan(spx_returns.loc['1990-07'].index[0], spx_returns.loc['1991-03'].index[-1], facecolor = 'darkslategrey', alpha = 0.3)
#plt.axvspan(spx_returns.loc['2001-03'].index[0], spx_returns.loc['2001-11'].index[-1], facecolor = 'darkslategrey', alpha = 0.3)
#plt.axvspan(spx_returns.loc['2007-12'].index[0], spx_returns.loc['2009-06'].index[-1], facecolor = 'darkslategrey', alpha = 0.3)
plt.axvspan(demeaned_spx_returns.loc['2020-02'].index[0], demeaned_spx_returns.loc['2020-04'].index[-1], facecolor = 'darkslategrey', alpha = 0.3)
plt.grid(True)

results = cusum_moment_stability(demeaned_spx_returns, moments=[2,4])
unitroot_test_results = unitroot_test(demeaned_spx_returns)

hill_results = hill_estimator(demeaned_spx_returns, alpha0=4)
kde_results = kde_returns(demeaned_spx_returns)
means = test_mean_significance(demeaned_spx_returns)