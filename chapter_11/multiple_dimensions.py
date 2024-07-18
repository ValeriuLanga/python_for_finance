import numpy as np
import pandas as pd
from pylab import plt, mpl

def fm(p):
    x, y = p
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2

x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x, y)    # generated 2D ndarray objects ('grids') out of the 1D ndarray objects

Z = fm((X, Y))
x = X.flatten() # yield 1D ndarray from 2D ndarray object
y = Y.flatten()
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(projection='3d')

# surf = ax.plot_surface(X, Y, Z, 
#                        rstride=2, 
#                        cstride=2,
#                        cmap='coolwarm',
#                        linewidth=0.5,
#                        antialiased=True
#                        )
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('f(x,y)')
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

# now let's try to approximate this bad boy
matrix = np.zeros((len(x), 6 + 1))
matrix[:, 6] = np.sqrt(y)   # look at how we create the fm() values
matrix[:, 5] = np.sin(x)    # same for x
matrix[:, 4] = y ** 2       # usual exponentials 
matrix[:, 3] = x ** 2
matrix[:, 2] = y
matrix[:, 1] = x
matrix[:, 0] = 1

reg = np.linalg.lstsq(matrix, fm((x,y)), rcond=None)[0] # going with least squares again, this time in 3d
print(reg)

RZ = np.dot(matrix, reg).reshape((20, 20))  # regression results are put in a grid structure (otherwise an array)
                                            # 20 x 20 because we know how large this linspace is; see above for x, y    

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(projection='3d')
surf1 = ax.plot_surface(X, Y, Z, 
                       rstride=2, 
                       cstride=2,
                       cmap='coolwarm',
                       linewidth=0.5,
                       antialiased=True
                       ) 

surf2 = ax.plot_wireframe(X, Y, RZ, 
                       rstride=2, 
                       cstride=2,
                       label='regression',
                       color='r'
                       )

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
fig.colorbar(surf1, shrink=0.5, aspect=5)

plt.show()


