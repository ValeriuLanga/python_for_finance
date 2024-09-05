import pandas as pd
import numpy as np
from pylab import mpl, plt

import statsmodels.formula.api as sm

from sklearn.linear_model import LinearRegression

symbol = 'SPX'

data = pd.read_json('E:\\coding\\python_for_finance\\data\\spxd.json').dropna()
data.set_index('date', inplace=True)
data = data.drop(labels=['open', 'high', 'low', 'adjusted_close', 'volume'], axis=1)
data = data.rename({'close' : symbol}, axis=1)
# print(data)

lags = 5
def create_lags(data, use_returns=False):
    global cols
    cols = [] 

    for lag in range(1, lags + 1):
        col = "lag_{}".format(lag)
        if not use_returns:
            data[col] = data[symbol].shift(lag)
        else:
            data[col] = data['returns'].shift(lag)

        cols.append(col)

create_lags(data)
# print(data.head(10))
"""
                SPX    lag_1    lag_2    lag_3    lag_4    lag_5
date
2024-01-02  44.0425      NaN      NaN      NaN      NaN      NaN
2024-01-03  43.7950  44.0425      NaN      NaN      NaN      NaN
2024-01-04  43.8800  43.7950  44.0425      NaN      NaN      NaN
2024-01-05  43.7400  43.8800  43.7950  44.0425      NaN      NaN
2024-01-08  43.8600  43.7400  43.8800  43.7950  44.0425      NaN
2024-01-09  44.1325  43.8600  43.7400  43.8800  43.7950  44.0425
"""
data = data.dropna()

# # OLS to figure out which of the lagging returns makes has the highest corr to current SPX price on T
# result = sm.ols(formula='SPX ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5', data=data).fit()

# print(result.params)
"""
Intercept    1.090577   
lag_1        0.904530   <- only lag_1 is relevant in terms of predicting the price on T
lag_2       -0.121877   
lag_3        0.235704
lag_4        0.012016
lag_5       -0.051817
"""

# let's move to OLS regression and prediction
data['returns'] = np.log(data[symbol] / data[symbol].shift(1))
data = data.dropna()

data['direction'] = np.sign(data['returns']).astype(int)
# print(data.head())

# data['returns'].hist(bins=25, figsize=(10, 6))
# plt.show()

# drop other indicators
data = data.drop(labels=cols, axis=1)
cols.clear()
# print(data.head())
"""
                SPX   returns  direction
date
2024-01-10  44.2750  0.003224          1
2024-01-11  44.1050 -0.003847         -1
2024-01-12  44.4425  0.007623          1
2024-01-15  44.4150 -0.000619         -1
2024-01-16  44.3775 -0.000845         -1
"""
# switch to ln returns from price
lags = 2
create_lags(data, True)
# for lag in range(1, lags + 1):
#     col = "lag_{}".format(lag)

#     # shift ln returns by lag
#     data[col] = data['returns'].shift(lag)
#     cols.append(col)

data = data.dropna()
# data.plot.scatter(x='lag_1', y='lag_2', c='returns', cmap='coolwarm', figsize=(10, 6), colorbar=True)
# plt.axvline(0, c='r', ls='--')
# plt.axhline(0, c='r', ls='--')

# plt.show()

#
# regression time
#

model = LinearRegression()

# try to predict both based on direction and returns
data['pos_ols_1'] = model.fit(data[cols], data['returns']).predict(data[cols])
data['pos_ols_2'] = model.fit(data[cols], data['direction']).predict(data[cols])

# print(data[['pos_ols_1', 'pos_ols_2']].head())

data[['pos_ols_1', 'pos_ols_2']] = np.where(
    data[['pos_ols_1', 'pos_ols_2']] > 0, 1, -1
)   # transform real value prediction to directional

# print(data[['pos_ols_1', 'pos_ols_2']].head())
"""
            pos_ols_1  pos_ols_2
date
2024-01-12          1          1
2024-01-15          1          1
2024-01-16          1          1
2024-01-17          1          1
2024-01-18          1          1
"""

# diff directional predictions between trend following vs return prediction
# print(data['pos_ols_1'].value_counts())
# print(data['pos_ols_2'].value_counts())

# number of trades we would have made
# .diff() does T - (T-1)
# print((data['pos_ols_1'].diff() != 0).sum())
# print((data['pos_ols_2'].diff() != 0).sum())
"""
Name: count, dtype: int64
39
41
"""

# ok let's compute actual returns of our mini-strategy
# -1 / 1 - we take the loss or make the profit
data['strat_ols_1'] = data['pos_ols_1'] * data['returns']
data['strat_ols_2'] = data['pos_ols_2'] * data['returns']

# print(data[['returns', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp))
"""
returns        1.150323 # benchmark
strat_ols_1    1.134811 # still positive but under benchmark
strat_ols_2    1.153341 # this beats the benchmark return
"""

# how many predictions were correct though? 
# print((data['direction'] == data['pos_ols_1']).value_counts())
# print((data['direction'] == data['pos_ols_2']).value_counts())

"""
Pretty narrow margins - interesting

dtype: float64
True     62
False    50
Name: count, dtype: int64
True     61
False    51
Name: count, dtype: int64
"""

# visual representation of strategies vs benchmark returns
# data[['returns', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot(figsize=(10,6))
# plt.show()

########################
# Clustering
########################
import warnings
warnings.filterwarnings('ignore')
# kmeans complains about some stupid memory leak 
# since this script runs for a few sec it's fine to ignore it

from sklearn.cluster import KMeans

model = KMeans(n_clusters=2, random_state=0)
model.fit(data[cols])

data['pos_clus'] = model.predict(data[cols])
data['pos_clus'] = np.where(data['pos_clus'] == 1, -1, 1)

# print(data['pos_clus'].values)
# plt.figure(figsize=(10,6))
# plt.scatter(data[cols].iloc[:, 0], data[cols].iloc[:, 1], c=data['pos_clus'], cmap='coolwarm')
# plt.show()

# let's look at returns as well
data['strat_clus'] = data['pos_clus'] * data['returns']
# print(data[['returns', 'strat_clus']].sum().apply(np.exp))  # go from log rets to normal rets
"""
returns       1.150323
strat_clus    0.963966
"""

# print((data['direction'] == data['pos_clus']).value_counts())   # how many correct predictions did we do
"""
False    59
True     53
"""
# interesting to see strategy would perform _worse_
# data[['returns', 'strat_clus']].cumsum().apply(np.exp).plot(figsize=(10,6))
# plt.show()


########################
# Frequency approach
########################
# four possible movements based on 2 binary features (transformed from real-values price movements)
# (0,1) (0,0) (1,0) (1,1)
# one feature is the price lagging by 1 day, the other by 2 days
# print(cols)
"""
['lag_1', 'lag_2']
"""
def create_bins(data, bins=[0]):
    global cols_bin
    cols_bin = []

    for col in cols:
        col_bin = col + '_bin'
        data[col_bin] = np.digitize(data[col], bins=bins)

        cols_bin.append(col_bin)

create_bins(data)
# print(data.head())
"""
                SPX   returns  direction     lag_1     lag_2  pos_ols_1  pos_ols_2  strat_ols_1  strat_ols_2  pos_clus  strat_clus  lag_1_bin  lag_2_bin
date
2024-01-12  44.4425  0.007623          1 -0.003847  0.003224          1          1     0.007623     0.007623         1    0.007623          0          1
2024-01-15  44.4150 -0.000619         -1  0.007623 -0.003847          1          1    -0.000619    -0.000619        -1    0.000619          1          0
2024-01-16  44.3775 -0.000845         -1 -0.000619  0.007623          1          1    -0.000845    -0.000845         1   -0.000845          0          1
2024-01-17  44.0650 -0.007067         -1 -0.000845 -0.000619          1          1    -0.007067    -0.007067         1   -0.007067          0          0
2024-01-18  44.2600  0.004416          1 -0.007067 -0.000845          1          1     0.004416     0.004416         1    0.004416          0          0
"""

grouped = data.groupby(cols_bin + ['direction'])
# print(grouped.size())
"""
lag_1_bin  lag_2_bin  direction
0          0          -1            8
                       1           12
           1          -1           12
                       1           18
1          0          -1           15
                       1           15
           1          -1           15
                       1           17
"""

res = grouped['direction'].size().unstack(fill_value=0)
# print(res.head())

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

res.style.apply(highlight_max, axis=1)  # what does this do??

data['pos_freq'] = np.where(data[cols_bin].sum(axis=1) == 2, -1, 1)
# print((data['direction'] == data['pos_freq']).value_counts())
# it's right more than it's wrong ...
"""
True     60
False    52
"""

data['strat_freq'] = data['pos_freq'] * data['returns']
# print(data[['returns', 'strat_freq']].sum().apply(np.exp))
# mild over-performance for strat_freq
"""
returns       1.150323
strat_freq    1.152548
"""

# data[['returns', 'strat_freq']].cumsum().apply(np.exp).plot(figsize=(10,6))
# plt.show()  # notice there's a period when it vastly outperforms benchmark

########################
# Classification - 2 binary features
########################

from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# fitting the models based on the binary feature values
# deriving the resulting position values

C = 1

# this holds INSTANCES of models
models = {
    'log_reg': linear_model.LogisticRegression(C=C),
    'gauss_nb': GaussianNB(),
    'svm': SVC(C=C)
}

# print(data.head())
def fit_models(data):
    mfit = { model: models[model].fit(data[cols_bin], 
                                      data['direction']) 
                                      for model in models.keys() }
    # print(mfit)

fit_models(data)

def derive_positions(data):
    for model in models.keys():
        data['pos_' + model] = models[model].predict(data[cols_bin])
    
derive_positions(data)

# print(data.head())
"""
                SPX   returns  direction     lag_1     lag_2  pos_ols_1  pos_ols_2  strat_ols_1  strat_ols_2  pos_clus  strat_clus  lag_1_bin  lag_2_bin  pos_freq  strat_freq  pos_log_reg  pos_gauss_nb  pos_svm
date
2024-01-12  44.4425  0.007623          1 -0.003847  0.003224          1          1     0.007623     0.007623         1    0.007623          0          1         1    0.007623            1             1        1
2024-01-15  44.4150 -0.000619         -1  0.007623 -0.003847          1          1    -0.000619    -0.000619        -1    0.000619          1          0         1   -0.000619            1             1        1
2024-01-16  44.3775 -0.000845         -1 -0.000619  0.007623          1          1    -0.000845    -0.000845         1   -0.000845          0          1         1   -0.000845            1             1        1
2024-01-17  44.0650 -0.007067         -1 -0.000845 -0.000619          1          1    -0.007067    -0.007067         1   -0.007067          0          0         1   -0.007067            1             1        1
2024-01-18  44.2600  0.004416          1 -0.007067 -0.000845          1          1     0.004416     0.004416         1    0.004416          0          0         1    0.004416            1             1        1
"""

# vectorized backtesting
def evaluate_strats(data):
    global sel
    sel = []
    for model in models.keys():
        col = 'strat_' + model
        data[col] = data['pos_' + model] * data['returns']

        sel.append(col)
    sel.insert(0, 'returns')

evaluate_strats(data)

# do this after calling evaluate_strats - 'sel' won't exist otherwise 
sel.insert(1, 'strat_freq')

# print(sel)
"""
['returns', 'strat_freq', 'strat_log_reg', 'strat_gauss_nb', 'strat_svm']
"""

# print(data[sel].sum().apply(np.exp))
"""
returns           0.772672
strat_freq        1.499017
strat_log_reg     1.928349
strat_gauss_nb    1.678130
strat_svm         1.928349  - almost 3x benchmark
"""

# data[sel].cumsum().apply(np.exp).plot(figsize=(10,6))
# plt.show()


########################
# Classification - 5 binary features
########################

# refresh data
data = pd.read_json('E:\\coding\\python_for_finance\\data\\spxd.json').dropna()
data.set_index('date', inplace=True)
data = data.drop(labels=['open', 'high', 'low', 'adjusted_close', 'volume'], axis=1)
data = data.rename({'close' : symbol}, axis=1)

data['returns'] = np.log(data / data.shift(1))
data['direction'] = np.sign(data['returns'])

lags = 5  
create_lags(data, True)
data = data.dropna()

create_bins(data)
data = data.dropna()
# print(data)

fit_models(data)
derive_positions(data)
evaluate_strats(data)

print(data[sel].sum().apply(np.exp))
"""
returns           0.775510
strat_log_reg     1.908406
strat_gauss_nb    2.152262
strat_svm         2.864895
"""
data[sel].cumsum().apply(np.exp).plot(figsize=(10,6))
plt.show()