import numpy as np
import pandas as pd
from pylab import plt, mpl

import scipy.interpolate as spi
import scipy.optimize as sco
import math

def Eu(p):
    a, b = p    # a, b are the ammounts of each stock that the investor decides to buy
    return - (0.5 * math.sqrt(a * 15 + b * 5) +
             0.5 * math.sqrt(a * 5 + b * 12))   
    # we're actually calculating the max so we need to negate the maximum utilty formula


if __name__ == '__main__':
    initial_px_a = 10
    initial_px_b = 10
    initial_funds = 100

    constraints = (
        {
            'type' : 'ineq',
            'fun' : lambda p: initial_funds - p[0] * initial_px_a - p[1]* initial_px_b
        }
    )
    bounds = ((0, 1000), (0, 1000))

    result = sco.minimize(
        Eu, 
        [5,5],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    print(result)
    print(result['x']) # optimal allocation
    print(-result['fun']) # remember that we flipped the func
    print(np.dot(result['x'], [initial_px_a, initial_px_b]))    # how much money did we spend

