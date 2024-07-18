import math
import numpy as np
import numpy.random as npr
import scipy.stats as scs

from pylab import plt, mpl
from monte_carlo import print_statistics

if __name__ == '__main__':
    S0 = 100 # initial index level
    r = 0.05    # riskless short rate
    sigma = 0.2 # 
    lamb = 0.75 # jump intensity
    mu = -0.6 # mean jump size
    delta = 0.25 # jump volatility
    rj = lamb * (np.exp(mu + 0.5 * delta ** 2) - 1) # drift correction

    T = 1.0
    M = 50
    I = 10_000
    dt = T / M

    S = np.zeros((M + 1, I))
    S[0] = S0

    sn1 = npr.standard_normal((M+1, I))
    sn2 = npr.standard_normal((M+1, I))
    poisson = npr.poisson(lamb * dt, (M + 1, I)) # poisson distrib

    for t in range (1, M + 1):
        S[t] = S[t - 1] * (
            np.exp((r - rj - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * sn1[t])
            + (np.exp(mu + delta * sn2[t]) - 1) 
            * poisson[t]
        )
        S[t] = np.maximum(S[t], 0)

    # plt.figure(figsize=(10, 6))
    # plt.hist(S[-1], bins=50)
    # plt.xlabel('value')
    # plt.ylabel('frequency')

    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(S[:,:20], lw=1.5)    # chart all values for last 20 iters
    plt.xlabel('time')
    plt.ylabel('index level')

    plt.show()