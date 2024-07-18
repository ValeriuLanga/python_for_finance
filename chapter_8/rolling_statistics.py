import pandas as pd
import numpy as np
from pylab import mpl, plt

# import data from jsons
data = pd.read_json('E:\\coding\\python_for_finance\\data\\tsla.json')
data.set_index('date', inplace=True)
data = data.drop(labels=['open', 'high', 'low', 'adjusted_close', 'volume'], axis=1)
data = data.rename({'close' : 'TSLA'}, axis=1)

# if we had more stocks than just tsla, we could do the below
sym = 'TSLA'
data = pd.DataFrame(data[sym]).dropna()
print(data)

# let's get some rolling statistics
window = 20 # samples but in our case closing px
data['min'] = data[sym].rolling(window=window).min()    # remember that thsi will add a 'min' column to our DataFrame
data['mean'] = data[sym].rolling(window=window).mean()  # now we really need to get the TSLA column as we have more columns
data['std'] = data[sym].rolling(window=window).std()
data['median'] = data[sym].rolling(window=window).median()
data['max'] = data[sym].rolling(window=window).max()
data['ewma'] = data[sym].ewm(halflife=0.5, min_periods=window).mean()

print(data)

# plot rolling statistics for last 200 data points
# ax = data[['min', 'mean', 'max']].iloc[-200:].plot( # final 200 rows, 3 metrics
#     figsize=(10,6), style=['g--', 'r--', 'b--'],    # green dotted line, red dotted line, blue dotted line
#     lw=0.8
# )
# data[sym].iloc[-200:].plot(ax=ax, lw=2.0)   # add original series i.e. closing px - again, final 200 rows
# plt.show()

# SMA - Simple Moving Averages
data['SMA1'] = data[sym].rolling(window=5).mean()
data['SMA2'] = data[sym].rolling(window=30).mean()

# data[[sym, 'SMA1', 'SMA2']].plot(figsize=(10,6))
# plt.show()

# what if we were to create a trading strategy out of this
# 1 means we go long
# -1 means we go short
data.dropna(inplace=True)
data['positions'] = np.where(data['SMA1'] > data['SMA2'],   # short term SMA < long term SMA
                             1,         # we want to GO LONG
                             -1)        # we want to GO SHORT

ax = data[[sym, 'SMA1', 'SMA2', 'positions']].plot(figsize=(10,6), secondary_y='positions')
ax.get_legend().set_bbox_to_anchor((0.25,0.85))
plt.show()