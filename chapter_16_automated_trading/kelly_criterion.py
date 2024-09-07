import math
import time
import numpy as np
import pandas as pd
import datetime as dt
import cufflinks as cf
from pylab import plt

np.random.seed(1000)

p = .55         # probability to get heads
f = p - (1 - p) # probability to get tails
I = 50          # nr series to be simulated
n = 100         # trials per series

def run_simulation(f):
    c = np.zeros((n, I))

    c[0] = 100  # starting 'capital'
    for i in range(I):
        for t in range(1, n):
            o = np.random.binomial(1, p)    # toss the coin
            if o > 0:   # heads
                c[t, i] = (1 + f) * c[t - 1, i] # made some bucks
            else:   # tails
                c[t, i] = (1 - f) * c[t - 1, i] # lost some bucks
            
    return c

c_1 = run_simulation(f)
print(c_1.round(2))

# plt.figure(figsize=(10,6))
# plt.plot(c_1, 'b', lw=0.5)
# plt.plot(c_1.mean(axis=1), 'r', lw=2.5)

# plt.show()

# let's try diff values for the bet size
c_2 = run_simulation(0.05)
c_3 = run_simulation(0.25)
c_4 = run_simulation(0.5)

plt.figure(figsize=(10,6))
plt.plot(c_1.mean(axis=1), 'r', label='$f^*=0.1$')
plt.plot(c_2.mean(axis=1), 'b', label='$f=0.05$')
plt.plot(c_3.mean(axis=1), 'y', label='$f=0.25$')
plt.plot(c_4.mean(axis=1), 'm', label='$f=0.5$')
plt.legend(loc=0)

plt.show()



