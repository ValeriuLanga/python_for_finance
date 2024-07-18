import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl
import scipy.stats as scs

def print_statistics(a1 : np.ndarray, a2 : np.ndarray):
    sta1 = scs.describe(a1)
    sta2 = scs.describe(a2)

    print('%14s %14s %14s' % ('statistic', 'data set 1', 'data set 2'))
    print(45 * "-")
    print('%14s %14.3f %14.3f' % ('size', sta1[0], sta2[0]))
    print('%14s %14.3f %14.3f' % ('min', sta1[1][0], sta2[1][0]))
    print('%14s %14.3f %14.3f' % ('max', sta1[1][1], sta2[1][1]))
    print('%14s %14.3f %14.3f' % ('mean', sta1[2], sta2[2]))
    print('%14s %14.3f %14.3f' % ('std', np.sqrt(sta1[3]),
                                np.sqrt(sta2[3])))
    print('%14s %14.3f %14.3f' % ('skew', sta1[4], sta2[4]))
    print('%14s %14.3f %14.3f' % ('kurtosis', sta1[5], sta2[5]))


if __name__ == '__main__':
    # Black Scholes Merton setup for option pricing 
    # Price at a future date T given a level S0 as of today 
    S0 = 100
    r = 0.05        # constant riskless short rate
    sigma = 0.25    # constant volatility i.e. standard deviation of returns
    T = 2.0         # time horizon in years
    I = 10_000        # number of iterations/simulations

    # actual equation below - standard normal distrib used 
    ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * npr.standard_normal(I))

    # plt.figure(figsize=(10,6))
    # plt.hist(ST1, bins=50)
    # plt.xlabel('index level')
    # plt.ylabel('frequency')

    # plt.show()

    ##################################
    # looking at the graph it looks to be a log normal distribution
    ST2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T,    # mean
                             sigma * np.sqrt(T),            # standard deviation
                             size=I)   
    # notice we remove '* npr.standard_normal(I)' and replace it w an I param

    # plt.figure(figsize=(10,6))
    # plt.hist(ST2, bins=50)
    # plt.xlabel('index level')
    # plt.ylabel('frequency')

    # plt.show()

    # print_statistics(ST1, ST2)
    
    ##################################
    # stochastic approach
    I = 10_000
    M = 50      # number of time intervals
    dt = T / M  # length of time interval in year fractions 

    S = np.zeros((M + 1, I))    # index levels as a 2-d array
    S[0,:] = S0                 # initial levels go on 1st row
    
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r- 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * npr.standard_normal(I))

    # plt.figure(figsize=(10,6))
    # plt.hist(S[-1], bins=50)
    # plt.xlabel('index level')
    # plt.ylabel('frequency')
    # plt.show()

    # let's show them side by side
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
    # ax1.hist(S[-1], bins=50)
    # ax1.set_title('index level - stochastic')
    # # ax1.label('index level - stochastic')
    # # ax1.ylabel('frequency')

    # ax2.hist(ST2, bins=50)
    # ax2.set_title('index level - static')
    # # ax2.label('index level - static')
    # # ax2.ylabel('frequency')
    # plt.show()

    # let's overlay the data
    # plt.figure(figsize=(10,6))
    # plt.hist([S[-1], ST2], 
    #          label=['index level - stochastic', 'index level - static'], 
    #          color=['b', 'g'], 
    #          bins=50,
    #          alpha=0.5)
    # plt.legend(loc=0)
    # plt.xlabel('frequency')
    # plt.ylabel('index value')
    # plt.show()

    print_statistics(S[-1], ST2)
    # let's see some simulated paths
    plt.figure(figsize=(10,6))
    plt.plot(S[:, :10], lw=1.5)
    plt.xlabel('time')
    plt.ylabel('index level')
    plt.show()

