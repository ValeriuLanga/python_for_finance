import numpy as np
import pandas as pd
import datetime as dt
from pylab import mpl, plt

def prepare_df(path: str, symbol: str) -> pd.DataFrame:

    raw = pd.read_json(path)
    raw.set_index('date', inplace=True)
    raw = raw.drop(labels=['open', 'high', 'low', 'adjusted_close', 'volume'], axis=1)
    raw = raw.rename({'close' : symbol}, axis=1)
    
    return pd.DataFrame(raw).dropna()

if __name__ == '__main__':

    # import data from jsons
    raw = prepare_df('E:\\coding\\python_for_finance\\data\\msft.json', 'MSFT')
    raw2 = prepare_df('E:\\coding\\python_for_finance\\data\\msft_to_begin_24.json', 'MSFT')
    
    data = pd.concat([raw, raw2])
    data.sort_index(axis=0, inplace=True)
    # print(data)

    symbol = 'MSFT'

    SMA1 = 5
    SMA2 = 30

    data['SMA1'] = data[symbol].rolling(SMA1).mean()
    data['SMA2'] = data[symbol].rolling(SMA2).mean()
    # print(data)

    # data.plot(figsize=(10, 6))
    # plt.show()

    data.dropna(inplace=True)
    data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1) # buy i.e. 1 if short term SMA > long term SMA; else sell

    # ax = data.plot(secondary_y='Position', figsize=(10,6))
    # ax.get_legend().set_bbox_to_anchor((0.25, 0.85))
    # plt.show()

    ###################
    # time for some backtesting
    data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))  # benchmark; convert to log returns
    data['Strategy'] = data['Position'].shift(1) * data['Returns']  # strategy; buy/sell
    data.dropna(inplace=True)

    cum_returns = np.exp(data[['Returns', 'Strategy']].sum())
    print(cum_returns)

    annualized_volatility = data[['Returns', 'Strategy']].std() * 252 ** 0.5
    print(annualized_volatility)

    ax = data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    data['Position'].plot(ax=ax, secondary_y='Position', style='--')
    plt.show()

    


