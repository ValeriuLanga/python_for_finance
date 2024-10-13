import pandas as pd
import numpy as np
from pylab import plt
import scipy.stats as scs


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
data = data.loc['2016-12-31' : '2017-12-31']
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

# test[['returns', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot(figsize=(10, 6))
# plt.show()


# let's calculate the optimal leverage acc to Kelly's crit now
mean = test[['returns', 'strategy_tc']].mean() * len(data) # * 12   # we're already looking at one year worth's
print(mean)

var = test[['returns', 'strategy_tc']].var() * len(data) # * 12   # we're already looking at one year worth's
# print(var)

vol = var ** 0.5    # annualised volatility
# print(vol)

optimal_leverage = mean / var
# print(optimal_leverage)

optimal_leverage_half = optimal_leverage * 0.5
# print(optimal_leverage_half)
"""
returns        10.518406
strategy_tc    31.035424
"""

to_plot = ['returns', 'strategy_tc']
for level in [10, 20, 30, 40, 50]:
    label = 'lstrategy_tc_{}'.format(level)
    test[label] = test['strategy_tc'] * level # scale strategy on diff levels

    to_plot.append(label)

# # show the performance of the trading strategy w txn costs for different leverage values
# test[to_plot].cumsum().apply(np.exp).plot(figsize=(10,6))
# plt.show()


# now let's do some risk analysis
# maximum drawdown - largest loss (dip) after a recent high
# largest drawdon period - longest period that the trading system needs to get back to a recent high

# we make some assumptions - initial position is 33k and the leverage ratio is 30 (as proven by the optimal 1/2 leverage)
equity = 3_333
risk = pd.DataFrame(test['lstrategy_tc_30'])    # log returns for 30 leverage ratio
risk['equity'] = risk['lstrategy_tc_30'].cumsum().apply(np.exp) * equity    # scale returns by initial equity

risk['cummax'] = risk['equity'].cummax()    # cummulative max values
risk['drawdown'] = risk['cummax'] - risk['equity'] # drawdown values over time

# print(risk)
"""
                     lstrategy_tc_30        equity        cummax     drawdown
Date
2017-10-18 21:00:00         0.156636   3898.174645   3898.174645     0.000000
2017-10-19 21:00:00         0.176748   4651.810681   4651.810681     0.000000
2017-10-21 21:00:00         0.038098   4832.454571   4832.454571     0.000000
2017-10-22 21:00:00        -0.047000   4610.584142   4832.454571   221.870429   <-- basically the delta between max px and current px
2017-10-23 21:00:00         0.032155   4761.244951   4832.454571    71.209620
"""

# print(risk['drawdown'].max())  # maxim global drawdown value (i.e. biggest dip)
t_max = risk['drawdown'].idxmax()   # when do we see the biggest drawdown
# print(t_max)
"""
11780.166721343003
2017-12-13 22:00:00
"""

"""
A new high <=> a drawdown value of 0
The drawdown period <=> the period between two such high
"""
temp = risk['drawdown'][risk['drawdown'] == 0]  # select only high point resets
periods = (temp.index[1:].to_pydatetime() - temp.index[:1].to_pydatetime())
# print(periods)
"""
[datetime.timedelta(days=1) datetime.timedelta(days=3)
 datetime.timedelta(days=6) datetime.timedelta(days=7)
 datetime.timedelta(days=8) datetime.timedelta(days=11)
 datetime.timedelta(days=12) datetime.timedelta(days=13)
 ...]
"""
t_per = periods.max()
# print(t_per)      # max drawdown period
"""
34 days, 1:00:00
"""

# risk[['equity', 'cummax']].plot(figsize=(10,6)) # equity vs cummax
# plt.axvline(t_max, c='r', alpha=0.5)    # mark the point of the biggest dip vs prev high
# plt.show()

#
#   Value at Risk (VaR)
#
# VaR is quoted as ccy ammount and represents the max loss to be expected given
# both a certain time horizon and a confidence level
percs = np.array([0.01, 0.1, 1., 2.5, 5.0, 10.0]) # precentile vals to be used
risk['returns'] = np.log(risk['equity'] / risk['equity'].shift(1))    # go to ln rets
VaR = scs.scoreatpercentile(equity * risk['returns'], percs)    # VaR given percentile values
# print(VaR)
"""
[-738.18412507 -735.83610588 -712.35591398 -629.81983659 -515.55375466
 -336.37495872]
"""

def print_VaR():
    print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
    print(33 * '-')
    for pair in zip(percs, VaR):
        print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))
        # translate percentile values into confidence levels i.e. 100 - percentile
        # translate VaR into a positive (it will be negative)

# print_VaR()
"""
Confidence Level    Value-at-Risk
---------------------------------
           99.99          738.184
           99.90          735.836
           99.00          712.356
           97.50          629.820
           95.00          515.554
           90.00          336.375
"""

# calculate VaR for time horizon of 1h by resampling the original DataFrame
# Node: we resample to 2D instead of 1D since 1D is our market data's frequency; in the book, the orig freq is 5m
hourly = risk.resample('2D', label='right').last()  
hourly['returns'] = np.log(hourly['equity'] / hourly['equity'].shift(1))

VaR = scs.scoreatpercentile(equity * hourly['returns'], percs)
print_VaR()
"""
Confidence Level    Value-at-Risk
---------------------------------
           99.99          990.560
           99.90          980.680
           99.00          881.881
           97.50          717.215
           95.00          492.591
           90.00          326.837   <- worth noting that all values got worse with a bigger resample

"""


# we can now persist the model for later use
import pickle
pickle.dump(model, open('algorithm.pkl', 'wb')) 


