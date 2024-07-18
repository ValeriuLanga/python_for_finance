import math
import numpy as np
import numpy.random as npr
import scipy.stats as scs

from pylab import plt, mpl
from monte_carlo import print_statistics


def srd_euler(M : int, I : int, x0 : float, kappa : float, theta : float, dt : float) -> np.ndarray:
    xh = np.zeros((M + 1, I))
    x = np.zeros_like(xh)
    xh[0] = x0
    x[0] = x0

    for t in range(1, M + 1):
        xh[t] = (xh[t-1] +
                 kappa * (theta - np.maximum(xh[t-1], 0)) * dt +
                 sigma * np.sqrt(np.maximum(xh[t-1], 0)) *
                 math.sqrt(dt) * npr.standard_normal(I)
                 )  # simulation based on euler scheme
    
    x = np.maximum(xh, 0)
    return x


def srd_exact(M : int, I : int, x0 : float, kappa : float, theta : float, dt : float) -> np.ndarray:
    x = np.zeros((M + 1, I))
    x[0] = x0

    for t in range(1, M + 1):
        df = 4 * theta * kappa / sigma ** 2
        c = (sigma ** 2 * ( 1 - np.exp(-kappa * dt))) / (4 * kappa)
        nc = np.exp(-kappa * dt) / c * x[t - 1]
        x[t] = c * npr.noncentral_chisquare(df, nc, size=I)

    return x


if __name__ == '__main__':
    x0 = 0.05   # initial value
    kappa = 3.0 # mean reversion factor
    theta = 0.02    # long term mean value
    sigma = 0.1 # volatility factor
    I = 10_000  # number of iters
    M = 50  # number of time intervals for discretization
    T = 2   # horizon in year
    dt = T / M  # length of time in year fractions

    x1 = srd_euler(M=M, I=I, x0=x0, kappa=kappa, theta=theta, dt=dt)

    plt.figure(figsize=(10,6))

    # plt.hist(x1[-1], bins=50)
    # plt.xlabel('value')
    # plt.ylabel('frequency')
    # plt.show()

    # let's plot 10 paths as well
    # notice the negative average drift (due to x0 > theta)
    # as well as convergence to theta = 0.02

    # plt.plot(x1[:, :10], lw=1.5)
    # plt.xlabel('time')
    # plt.ylabel('index level')
    # plt.show()


    #############################
    # let's try an exact discretization scheme instead of the euler one
    x2 = srd_exact(M=M, I=I, x0=x0, kappa=kappa, theta=theta, dt=dt)
    # plt.hist(x2[-1], bins=50)
    # plt.xlabel('value')
    # plt.ylabel('frequency')
    # plt.show()

    # same graph charachteristic as before
    # negative avg drift and convergance to theta=0.02
    # plt.plot(x2[:, :10], lw=1.5)
    # plt.xlabel('time')
    # plt.ylabel('index level')
    # plt.show()

    print_statistics(x1[-1], x2[-1])
    # some implications re time perf - but in short, the euler approach is much quicker (1/2 the time)


