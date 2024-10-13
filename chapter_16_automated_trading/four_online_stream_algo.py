import pandas as pd
import numpy as np
from pylab import plt
import scipy.stats as scs
import pickle

#
#   Online / stream algorithm
#

algorithm = pickle.load(open('algorithm.pkl', 'rb'))
symbol =  'EUR/USD'
amount = 100    # initial amount to trade
bar = '1D'      # 1 days bars (mkt data we have is 1 day)
position = 0    # initial position
lags = 5        # see chapter_16_automated_trading/three_kelly_algo_vectorized_backtesting.py
min_bars = lags + 1     # need to have at least 5 previous data points to be able to make a prediction
df = pd.DataFrame()     # we'll use this for resampled data


def automated_strategy(data, dataframe):
    global min_bars, position, df
    ldf = len(dataframe)
    df = dataframe.resample(bar, label='right').last().ffill()

    if ldf % 20 == 0:   # ?????
        print('%3d' % len(dataframe), end=',')
    
    if len(df) > min_bars:
        min_bars = len(df)
        
        df['Mid'] = df[['Bid', 'Ask']].mean(axis=1)
        df['Returns'] = np.log(df['Mid'] / df['Mid'].shift(1))
        df['Direction'] = np.where(df['Returns'] > 0, 1, -1)

        features = df['Direction'].iloc[-(lags + 1):-1] # ???? this should give us the lag features
        features = features.values.reshape(1, -1)   # reshape to model's interpretation
        signal = algorithm.predict(features)[0]

        print(signal)


# now we have to mock a live data feed which is relatively easy to do - we just load a different historical data set
# the model was trained on ['2016-12-31' : '2017-12-31']