import pandas as pd
import numpy as np
from pylab import plt

import math

# symbol
# SPTR500N data
# load https://www.invesco.com/uk/en/financial-products/etfs/invesco-sp-500-ucits-etf-dist.html#Overview
# S&P 500® Total Return (Net) Index (SPTR500N)
symbol = 'SPTR'
data = pd.read_excel('E:\\coding\\python_for_finance\\data\\sp500_2011_2024_invesco_trimmed.xlsx').dropna()

data.set_index('Date', inplace=True)
data = data.rename({'Index' : symbol}, axis=1)
data = data.sort_index(ascending=True)
data['returns'] = np.log(data / data.shift(1))
data = data.dropna()

mu = data.returns.mean() * 252                # (for SPTR) annualized return as there are 252 trading days
print(mu)

sigma = data.returns.std() * 252 ** 0.5     # annualized volatility 
r = 0.0                                     # risk free rate; simplified
f = (mu - r) / sigma ** 2                   # optimal kelly fraction

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
            equity_2.15  equity_2.84  equity_4.30
Date
2024-09-06    27.795254    52.404398    96.909962
2024-09-09    28.491374    54.136825   101.764096
2024-09-10    28.766387    54.826599   103.728652
2024-09-11    29.426138    56.486417   108.486646
2024-09-12    29.900797    57.689142   111.986540   <-- 'slight' overfitting 
"""

ax = data['returns'].cumsum().apply(np.exp).plot(legend=True, figsize=(10,6))
data[equs].plot(ax=ax, legend=True)
plt.show()
