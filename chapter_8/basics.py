import pandas as pd
import numpy as np
from pylab import mpl, plt

# load daily close for msft and tsla
data_msft = pd.read_json('E:\\coding\\python_for_finance\\msft.json')
data_msft.set_index('date', inplace=True)

# drop other columns & replace close with ticker
data_msft = data_msft.drop(labels=['open', 'high', 'low', 'adjusted_close', 'volume'], axis=1)
data_msft = data_msft.rename({'close' : 'MSFT'}, axis=1)

# same treatment for tsla
data_tsla = pd.read_json('E:\\coding\\python_for_finance\\tsla.json')
data_tsla.set_index('date', inplace=True)
data_tsla = data_tsla.drop(labels=['open', 'high', 'low', 'adjusted_close', 'volume'], axis=1)
data_tsla = data_tsla.rename({'close' : 'TSLA'}, axis=1)

# bring it into one data frame
data = data_tsla.join(data_msft)
# print(data.info())
# print(data)

# let's visualize really quick
# data.plot(figsize=(10,2), subplots=True)
# plt.show()

# summary statistics
# print(data.describe().round(2))
# print(data.mean())
# print(data.aggregate(['min', 'mean', 'std', 'median', 'max']).round(2))


# changes over time
# print(data.diff())  # absolute differences from prev TD
# print(data.diff().mean())

# pct chg / return is better
# print(data.pct_change().round(2))
# data.pct_change().mean().plot(kind='bar', figsize=(10,6))
# plt.show()


# log returns
returns = np.log(data / data.shift(1))  # https://www.allquant.co/post/magic-of-log-returns-concept-part-1
# data.shift(1) -> value from T+1 become value for T
# see formula from link - if Px at end is next day then we shift by 1 essentially 
# then we can just apply log and boom we have R - continous compound rate of return from prev day

# print(returns.round(2))
# returns.cumsum().apply(np.exp).plot(figsize=(10,6)) # add up all 1 day log changes into mega change 
# plt.show()

# resampling
print(data.resample('1W', label='right').last().head()) # downsampling i.e. from daily to wkly
# label='right' -> we are downsampling a range(left, right) so we want to keep the RIGHT lable as the index value; see note on p.217
returns.cumsum().apply(np.exp).resample('1M', label='right').last().plot(figsize=(10,6))
plt.show()


