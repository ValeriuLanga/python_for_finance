import pandas as pd
import numpy as np
from pylab import mpl, plt

vix = pd.read_csv('E:\\coding\\python_for_finance\\data\\VIX.csv', index_col=0, parse_dates=True).dropna()
# bring dates in range
vix = vix.loc['2024-01-02':'2024-06-21']
# drop some columns and rename close to VIX
vix = vix.drop(labels=['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
vix = vix.rename({'Close': 'VIX'}, axis=1)
# print(vix)

spx = pd.read_json('E:\\coding\\python_for_finance\\data\\spxd.json').dropna()
spx.set_index('date', inplace=True)
spx = spx.drop(labels=['open', 'high', 'low', 'adjusted_close', 'volume'], axis=1)
spx = spx.rename({'close' : 'SPX'}, axis=1)
# print(spx)

# join in the same table
data = vix.join(spx)
# print(data)

# show them side-by-side
# data.plot(subplots=True, figsize=(10,6))

# show on the same table - secodnary y for the VIX, inverse correlation obvious
# data.plot(secondary_y='VIX', figsize=(10,6))
# plt.show()


# let's shift to log returns
returns = np.log(data / data.shift(1)).dropna()
# print(returns.head()) 

# show log returns in subplots
# returns.plot(subplots=True, figsize=(10,6))
# plt.show()

# let's change to a scatter matrix to better see the correlation
# pd.plotting.scatter_matrix(
#     returns, 
#     alpha=0.2,
#     diagonal='hist',
#     hist_kwds={'bins' : 35},
#     figsize=(10,6)
# )
# plt.show()

# ordinary least squares regression below

# this returns the series that is the least squares fit to the data y sampled at x
# i.e. the least sq fit of the value of VIX sampled at indiv close px of SPX
# reg = np.polyfit(
#     returns['SPX'], # x-coords are SPX log rets
#     returns['VIX'], # y coordinates are VIX log rets
#     deg=1   # polynomial degree is 1
# )
# # reg = [-2.18651493e+00  1.77562015e-03]

# # just a scatter plot with the SPX & VIX; note how high VIX means low SPX
# ax = returns.plot(
#     kind='scatter',
#     x='SPX',    # x-axis is log returns of SPX
#     y='VIX',    # y-axis is log returns of VIX
#     figsize=(10,6)
# )

# # to which we add the linear regression
# # linear regression tries to find the line that minimizes the sum of squared distances to the data
# ax.plot(returns['SPX'],     # x-axis - SPX values   
#         np.polyval(reg, returns['SPX']), # evaluate polynomial at specific values; in this case, SPX log returns
#         'r',                # in red
#         lw=2
# )

# plt.show()


# correlation time
print(returns.corr())
#          VIX      SPX
# VIX  1.00000 -0.30297
# SPX -0.30297  1.00000

ax = returns['SPX'].rolling(window=20).corr(returns['VIX']).plot(figsize=(10,6))    # rolling 20-day correlation
ax.axhline(returns.corr().iloc[0, 1], # static correlation value as horizontal line; this is the y-value (see above)
           c='r'
)    

plt.show()