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
data = data.sort_index(ascending=True)

# SPX data
# data.set_index('date', inplace=True)
# data = data.drop(labels=['open', 'high', 'low', 'adjusted_close', 'volume'], axis=1)

data = data.rename({'close' : symbol}, axis=1)

data['returns'] = np.log(data / data.shift(1))
data = data.dropna()

mu = data.returns.mean() * 365              # (for SOL) 365 days since crypto trades non stop
                                            # (for SPX) annualized return as there are 252 trading days
print(mu)
# 0.4785953143436269
sigma = data.returns.std() * 365 ** 0.5     # annualized volatility 
print(sigma)
# 0.45626009321926025
r = 0.0                                     # risk free rate; simplified
f = (mu - r) / sigma ** 2                   # optimal kelly fraction
print(f)
# 2.2990238286500344

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
print(data[equs].tail())
"""
                     equity_1.15  equity_1.52  equity_2.30
start
2024-07-20 00:00:00     8.333963    13.026132    23.216090
2024-07-20 06:00:00     8.226354    12.804115    22.616555
2024-07-20 12:00:00     8.574504    13.519406    24.530878
2024-07-20 18:00:00     8.538317    13.444091    24.323822
2024-07-21 00:00:00     8.511757    13.388888    24.172492  <- overfitting to say the least 
"""

ax = data['returns'].cumsum().apply(np.exp).plot(legend=True, figsize=(10,6))
data[equs].plot(ax=ax, legend=True)
plt.show()
