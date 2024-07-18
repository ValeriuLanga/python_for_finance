import math
import numpy as np
import numpy.random as npr
import scipy.stats as scs

from pylab import plt, mpl
from monte_carlo import print_statistics

if __name__ == '__main__':
    S0 = 100.
    r = 0.05
    v0 = 0.1    # initial volatility value
    kappa = 3.0
    theta = 0.25
    sigma = 0.1
    rho = 0.6   # fixed correlation between the 2 Brownian motions
    T = 1.0

    correlation_matrix = np.zeros((2,2))
    correlation_matrix[0, :] = [1.0, rho]
    correlation_matrix[1, :] = [rho, 1.0]
    cholesky_matrix = np.linalg.cholesky(correlation_matrix)

    # print(cholesky_matrix)

    M = 50
    I = 10_000
    dt = T / M

    random_number = npr.standard_normal((2, M + 1, I))  # 3D random number data set

    # first we look at volatility
    v = np.zeros_like(random_number[0])
    vh = np.zeros_like(v)
    v[0] = v0
    vh[0] = v0

    for t in range (1, M + 1):
        ran = np.dot(cholesky_matrix, random_number[:, t, :])   # pick relevant number subset & transform via cholesky
        vh[t] = (vh[t - 1] +
                 kappa * (theta - np.maximum(vh[t - 1], 0)) * dt +
                 sigma * np.sqrt(np.maximum(vh[t - 1], 0)) * 
                 math.sqrt(dt) * ran[1]
                )   # Euler scheme above

    v = np.maximum(vh, 0)


    # now we look at the index level itself
    S = np.zeros_like(random_number[0])
    S[0] =S0

    for t in range(1, M + 1):
        ran = np.dot(cholesky_matrix, random_number[:, t, :])
        S[t] = S[t - 1] * np.exp(
                                (r - 0.5 * v[t]) * dt +  np.sqrt(v[t]) * ran[0] * np.sqrt(dt)
                                )
    
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
    # ax1.hist(S[-1], bins=50)
    # ax1.set_xlabel('index level')
    # ax1.set_ylabel('frequency')
    
    # ax2.hist(v[-1], bins=50)
    # ax2.set_xlabel('volatility')
    # plt.show()

    print_statistics(S[-1], v[-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,6))
    ax1.plot(S[:, :10], lw=1.5)
    ax1.set_ylabel('index level')

    ax2.plot(v[:, :10], lw=1.5)
    ax2.set_ylabel('volatility')
    ax2.set_xlabel('time')

    plt.show()