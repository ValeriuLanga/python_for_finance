import math
import numpy as np
import numpy.random as npr
import scipy.stats as scs

from pylab import plt, mpl
from monte_carlo import print_statistics

def gen_sn(M, I, anti_paths=True, mo_match=True):
    ''' Function to generate random numbers for simulation.

    Parameters
    ==========
    M: int
        number of time intervals for discretization
    I: int
        number of paths to be simulated
    anti_paths: boolean
        use of antithetic variates
    mo_math: boolean
        use of moment matching
    '''
    if anti_paths is True:
        sn = npr.standard_normal((M + 1, int(I / 2)))
        sn = np.concatenate((sn, -sn), axis=1)
    else:
        sn = npr.standard_normal((M + 1, I))
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std()
    return sn


def gbm_mcs_stat(K : float, I : int, S0 : np.ndarray, sigma : float, T : float, r : float):
    '''
    valuation of european call option in Black Scholes Merton by Monte carlo sim
    (of index level at maturity)
    '''
    sn = gen_sn(1, I)

    # simulate index level at maturity
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * sn[1])

    # payoff at maturity
    hT = np.maximum(ST, 0)
    # print(hT)

    # MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    
    return C0


def gbm_mcs_dyna(K: float, S0: np.ndarray, T: int, M: int, option='call'):
    '''
    valuation of european call option in Black Scholes Merton by Monte carlo sim
    (of index level paths)
    '''
    dt = T / M

    # simulate index level paths
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)

    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * sn[t])

    # case based calc of payoff
    if option == 'call':
        hT = np.maximum(S[-1] - K, 0)
    else:
        hT = np.maximum(K - S[-1], 0)

    # calculate MCS simulator
    C0 = math.exp(-r * T) * np.mean(hT)

    return C0


if __name__ == '__main__':
    S0 = 100.
    r = 0.05
    sigma = 0.25
    T = 1.0
    I = 50_000
    r = 0.05

    print(gbm_mcs_stat(K=105., I=I, S0=S0, sigma=sigma, T=T, r=r))

    M = 50 # number of time intervals for discretization
    print(gbm_mcs_dyna(K=110., S0=S0, T=T, M=M, option='call'))
    print(gbm_mcs_dyna(K=110., S0=S0, T=T, M=M, option='put'))

