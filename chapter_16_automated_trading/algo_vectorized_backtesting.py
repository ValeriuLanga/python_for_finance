import pandas as pd
import numpy as np
from pylab import plt

data = pd.read_csv("E:\\coding\\python_for_finance\\data\\fxcm_eur_usd_eod_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.sort_index(ascending=True)

data.columns = [x.lower() for x in data.columns]
# data = data.rename({'BidOpen': 'bidopen', 'BidHigh': 'bidhigh', 'BidLow': 'bidlow', 'BidClose': 'bidclose', 'AskOpen': 'askopen', 'AskHigh': 'askhigh', 'AskLow': 'asklow', 'AskClose': 'askclose', }, axis=1)
"""
                     BidOpen  BidHigh   BidLow  BidClose  AskOpen  AskHigh   AskLow  AskClose
Date
2013-01-01 22:00:00  1.31848  1.32935  1.31563   1.31850  1.31853  1.32940  1.31571   1.31860
2013-01-02 22:00:00  1.31850  1.31903  1.30468   1.30483  1.31860  1.31910  1.30471   1.30501
2013-01-03 22:00:00  1.30483  1.30897  1.29974   1.30679  1.30501  1.30898  1.29978   1.30697
2013-01-06 22:00:00  1.30679  1.31194  1.30168   1.31159  1.30697  1.31196  1.30168   1.31166
2013-01-07 22:00:00  1.31159  1.31398  1.30563   1.30805  1.31166  1.31400  1.30565   1.30815
...                      ...      ...      ...       ...      ...      ...      ...       ...
2017-12-25 22:00:00  1.18589  1.18788  1.18466   1.18581  1.18667  1.18791  1.18467   1.18587
2017-12-26 22:00:00  1.18581  1.19104  1.18551   1.18863  1.18587  1.19104  1.18552   1.18885
2017-12-27 22:00:00  1.18863  1.19591  1.18861   1.19424  1.18885  1.19592  1.18885   1.19426
2017-12-28 22:00:00  1.19424  1.20255  1.19362   1.20049  1.19426  1.20256  1.19369   1.20092
2017-12-31 22:00:00  1.20049  1.20121  1.19928   1.20105  1.20092  1.20144  1.19994   1.20144
"""

# take just one year
# data = data.loc['2016-12-31' : '2017-12-31']
# print(data)
"""
                     bidopen  bidhigh   bidlow  bidclose  askopen  askhigh   asklow  askclose
Date
2017-01-02 22:00:00  1.05174  1.05174  1.03402   1.04048  1.05295  1.05295  1.03406   1.04065
2017-01-03 22:00:00  1.04048  1.04999  1.03897   1.04864  1.04065  1.05004  1.03897   1.04922
2017-01-04 22:00:00  1.04864  1.06152  1.04807   1.06050  1.04922  1.06155  1.04811   1.06085
2017-01-05 22:00:00  1.06050  1.06220  1.05250   1.05319  1.06085  1.06241  1.05251   1.05372
2017-01-07 22:00:00  1.05319  1.05319  1.05267   1.05291  1.05372  1.05372  1.05300   1.05319
...                      ...      ...      ...       ...      ...      ...      ...       ...
2017-12-25 22:00:00  1.18589  1.18788  1.18466   1.18581  1.18667  1.18791  1.18467   1.18587
2017-12-26 22:00:00  1.18581  1.19104  1.18551   1.18863  1.18587  1.19104  1.18552   1.18885
2017-12-27 22:00:00  1.18863  1.19591  1.18861   1.19424  1.18885  1.19592  1.18885   1.19426
2017-12-28 22:00:00  1.19424  1.20255  1.19362   1.20049  1.19426  1.20256  1.19369   1.20092
2017-12-31 22:00:00  1.20049  1.20121  1.19928   1.20105  1.20092  1.20144  1.19994   1.20144
"""

spread = (data['askclose'] - data['bidclose']).mean()
data['midclose'] = (data['askclose'] + data['bidclose']) /2 
ptc = spread / data['midclose'].mean()  # avg proportional trans cost given avg spread and avg mid close price
# print(ptc)

# data['midclose'].plot(figsize=(10, 6), legend=True)
# plt.show()

data['returns'] = np.log(data['midclose'] / data['midclose'].shift(1))
data = data.dropna()
lags = 5

cols = []
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = data['returns'].shift(lag)
    cols.append(col)

data = data.dropna()
# digitize the lagged returns
data[cols] = np.where(data[cols] > 0, 1, 0)
# data['direction'] = np.sign(data['returns'])
data['direction'] = np.where(data['returns'] > 0, 1, -1)

# model time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model = SVC(C=1, kernel='linear', gamma='auto')
split = int(len(data) * 0.8)

# train the model
train = data.iloc[:split].copy()
model.fit(train[cols], train['direction'])
# in-sample accuracy
# print(accuracy_score(train['direction'], model.predict(train[cols])))
# 0.5702479338842975

# test the model
test = data.iloc[split:].copy()
test['position'] = model.predict(test[cols])
# out-sample accuracy
# print(accuracy_score(test['direction'], test['position']))
# 0.639344262295082

test['strategy'] = test['position'] * test['returns']
# print(sum(test['strategy'].diff() != 0))
# 61 position changes dictated by algo

# subtract avg transaction cost from returns every time we make a trade
test['strategy_tc'] = np.where(test['position'].diff() != 0, test['strategy'] - ptc, test['strategy'])  
# print(test[['returns', 'strategy', 'strategy_tc']].sum().apply(np.exp))
"""
returns        1.019071
strategy       1.062737
strategy_tc    1.054039 <- still beating the benchmark
"""

test[['returns', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()



