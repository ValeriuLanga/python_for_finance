import numpy as np
import pandas as pd
from pylab import plt, mpl

import scipy.interpolate as spi
import scipy.optimize as sco

from approximation import create_plot

output = False

def fm(p):
    x, y = p
    z = (np.sin(x) + 0.05 * x ** 2
            + np.sin(y) + 0.05 * y ** 2)
    
    if output == True:
        print('%8.4f | %8.4f | %8.4f' % (x, y, z))
    
    return z

if __name__ == '__main__':
    ############################
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)

    X, Y = np.meshgrid(x, y)
    Z = fm((X, Y))

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, Z,
                        rstride=2, cstride=2,
                        cmap='coolwarm',
                        linewidth=0.5,
                        antialiased=True
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()


    ############################
    # global optimization
    output = True
    sco.brute(  # brute force optimization
        fm,     # function
        ((-10, 10.1, 5), (-10, 10.1, 5)),   # ranges
        finish=None
    ) 
    # min appears to be 0 - for x, y = 0

    output=False
    opt1 = sco.brute(
        fm, 
        ((-10, 10.1, 0.1), (-10, 10.1, 0.1)),
        finish=None
    )

    print(opt1)
    print(fm(opt1))

    # local optimization
    output = True
    opt2 = sco.fmin(
        fm,
        opt1, 
        xtol=0.001,
        ftol=0.001,
        maxiter=15,
        maxfun=20
    )
    print(opt2)
    print(fm(opt2))