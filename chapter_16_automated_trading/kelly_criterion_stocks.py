import pandas as pd
import numpy as np
from pylab import plt

import math

symbol = 'SOL'
data = pd.read_parquet("E:\\coding\\analytical_tools\\data\\SOL-USD_2024-07-21_to_2023-07-21.parquet").dropna()
"""
                  start     low    high    open   close         volume
0   2024-07-21 00:00:00  172.53  175.57  173.68  173.21   94182.424910
1   2024-07-20 18:00:00  172.25  175.00  174.35  173.68  173954.281795
2   2024-07-20 12:00:00  167.27  174.59  168.15  174.32  255530.158479
3   2024-07-20 06:00:00  167.74  170.40  170.01  168.13   51514.282641
4   2024-07-20 00:00:00  168.37  171.91  169.21  170.04   93366.416896
"""
# load SPX
# symbol = 'SPX'
# data = pd.read_json('E:\\coding\\python_for_finance\\data\\spxd.json').dropna()

# SOL data
data['start'] = pd.to_datetime(data['start'])
data.set_index('start', inplace=True)
data = data.drop(labels=['low', 'high', 'open', 'volume'], axis=1)
print(data.dtypes)
data = data.sort_index(ascending=True)

print(data.tail())
# SPX data
# data.set_index('date', inplace=True)
# data = data.drop(labels=['open', 'high', 'low', 'adjusted_close', 'volume'], axis=1)

data = data.rename({'close' : symbol}, axis=1)

data['returns'] = np.log(data / data.shift(1))
data = data.dropna()
"""
             SPX   returns
date
2023-09-06  9910  0.000808  <- initial spx level
2023-09-07  9882 -0.002829
2023-09-08  9984  0.010269
2023-09-11  9876 -0.010876
2023-09-12  9800 -0.007725
...          ...       ...
2024-08-28  7410  0.009492
2024-08-29  7700  0.038390
2024-08-30  7695 -0.000650
2024-09-02  7620 -0.009794
2024-09-03  7600 -0.002628  <- note that the spx lost money in this run
"""

mu = data.returns.mean() * 252              # annualized return as there are 252 trading days
print(mu)
# -0.2645885096469468
sigma = data.returns.std() * 252 ** 0.5     # annualized volatility 
print(sigma)
# 0.2638898840606999
r = 0.0                                     # risk free rate; simplified
f = (mu - r) / sigma ** 2                   # optimal kelly fraction
print(f)
# -3.7994916586484413                       # negative kelly since spx was on a downturn

equs = []
def kelly_strategy(f):
    global equs
    equ = 'equity_{:.2f}'.format(f)
    equs.append(equ)

    cap = 'capital_{:.2f}'.format(f)
    data[equ] = 1.0               # initial equity value = 1
    # side-note - https://pandas.pydata.org/pdeps/0006-ban-upcasting.html

    data[cap] = data[equ] * f   # capital initial value is the 1 * (optimal kelly fraction)

    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]     
        data.loc[t, cap] = data[cap].loc[t_1] * math.exp(data['returns'].loc[t])
        data.loc[t, equ] = data[cap].loc[t] - data[cap].loc[t_1] + data[equ].loc[t_1]
        data.loc[t, cap] = data[equ].loc[t] * f 

kelly_strategy(f * 0.5)     # half-kelly
kelly_strategy(f * 0.66)    # 2/3 kelly
kelly_strategy(f)           # full kelly
# print(data[equs].tail())
"""
            equity_-1.90  equity_-2.51  equity_-3.80
datelogit
2024-08-28      1.443798      1.543761      1.640657
2024-08-29      1.336453      1.392255      1.396694
2024-08-30      1.338102      1.394522      1.400140
2024-09-02      1.362878      1.428606      1.451990
2024-09-03      1.369674      1.438008      1.466470
"""

ax = data['returns'].cumsum().apply(np.exp).plot(legend=True, figsize=(10,6))
data[equs].plot(ax=ax, legend=True)
plt.show()
