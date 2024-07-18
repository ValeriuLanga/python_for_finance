import math
import numpy as np
import numpy.random as npr
import scipy.stats as scs

from pylab import plt, mpl
from monte_carlo import print_statistics

if __name__ == '__main__':
    S0 = 100
    r = 0.05
    sigma = 0.25
    T = 30 / 365
    I = 10_000

    # end of period values for geometric brownian motion
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * npr.standard_normal(I))
    R_gbm = np.sort(ST - S0)    # final inxed values - begininng index values i.e. PnL
    print('Returns', R_gbm)
    # plt.figure(figsize=(10,6))
    # plt.hist(R_gbm, bins=50)
    # plt.xlabel('absolute return')
    # plt.ylabel('frequency')

    # plt.show()

    percentages = [0.01, 0.1, 1., 2.5, 5., 10.]
    VaR = scs.scoreatpercentile(R_gbm, per=percentages)
    print('VaR', VaR)
    print('%16s %16s' % ('Confidence level', 'Value-at-risk - standard'))
    print(33 * '-')
    for pair in zip(percentages, VaR):
        print('%16.2f %16.3f' % (100 - pair[0], - pair[1])) # R_gbm has losses i.e. negatives  

    ##############################################################
    # jump diffusion approach - instead of brownian motion
    M = 50
    lamb = 0.75 # jump intensity
    mu = -0.6   # mean jump size
    delta = 0.25    # jump volatility

    dt = 30. / 365 / M
    rj = lamb * (np.exp(mu + 0.5 * delta ** 2) - 1) # drift correction 

    S = np.zeros((M + 1, I))
    S[0] = S0
    sn1 = npr.standard_normal((M + 1, I))
    sn2 = npr.standard_normal((M + 1, I))
    poisson = npr.poisson(lamb * dt,  (M + 1, I))

    for t in range (1, M + 1):
        S[t] = S [t - 1] * (np.exp((r - rj - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * sn1[t])
                            + (np.exp(mu + delta * sn2[t]) - 1) 
                            * poisson[t]
                            )
        S[t] = np.maximum(S[t], 0)

    R_jd = np.sort(S[-1] - S0)

    # print('Returns - jump diffusion', R_jd)
    # plt.figure(figsize=(10,6))
    # plt.hist(R_jd, bins=50)
    # plt.xlabel('absolute return')
    # plt.ylabel('frequency')

    # plt.show()

    VaR = scs.scoreatpercentile(R_jd, per=percentages)
    print('VaR', VaR)
    print('%16s %16s' % ('Confidence level', 'Value-at-risk - jump diff'))
    print(33 * '-')
    for pair in zip(percentages, VaR):
        print('%16.2f %16.3f' % (100 - pair[0], - pair[1])) # R_gbm has losses i.e. negatives  


    # side by side
    percs = list(np.arange(0.0, 10.1, 0.1))
    gbm_var = scs.scoreatpercentile(R_gbm, percs)
    jd_var = scs.scoreatpercentile(R_jd, percs)

    plt.figure(figsize=(10,6))
    plt.plot(percs, gbm_var, 'b', lw=1.5, label='GBM')
    plt.plot(percs, jd_var, 'r', lw=1.5, label='JD')

    plt.legend(loc=4)
    plt.xlabel('100 - confidence level [%]')
    plt.ylabel('VaR')
    plt.ylim(ymax=0.0)

    plt.show()