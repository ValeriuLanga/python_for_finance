import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl

npr.seed(1000)
np.set_printoptions(precision=4)

print(npr.rand(10))
print(npr.rand(5, 5))

# to get numbers from a [a, b) range
a = 5
b = 10
print(npr.rand(10) * (b - a) + a) # scalar operation to be broadcasted to whole ndarray

# visualizing random data 
sample_size = 500

rn1 = npr.rand(sample_size, 3)  # cont
rn2 = npr.randint(0, 10, sample_size)   # discreet
rn3 = npr.sample(size=sample_size)  # cont
a = np.linspace(0, 100, 5)
rn4 = npr.choice(a, size=sample_size)   # discreet

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# ax1.hist(rn1, bins=25, stacked=True)
# ax1.set_title('rand')
# ax1.set_ylabel('frequency')

# ax2.hist(rn2, bins=25)
# ax2.set_title('randint')

# ax3.hist(rn3, bins=25)
# ax3.set_title('sample')
# ax3.set_ylabel('frequency')

# ax4.hist(rn4, bins=25)
# ax4.set_title('choice')

# plt.show()

#############################
# distributions as well
rn1 = npr.standard_normal(sample_size)
rn2 = npr.normal(100, 20, sample_size)
rn3 = npr.chisquare(df=0.5, size=sample_size)
rn4 = npr.poisson(lam=1.0, size=sample_size)    # only discreet one here


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

ax1.hist(rn1, bins=25, stacked=True)
ax1.set_title('standard normal')
ax1.set_ylabel('frequency')

ax2.hist(rn2, bins=25)
ax2.set_title('normal(100, 20)')

ax3.hist(rn3, bins=25)
ax3.set_title('chi square')
ax3.set_ylabel('frequency')

ax4.hist(rn4, bins=25)
ax4.set_title('poisson')

plt.show()