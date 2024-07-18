import scipy.interpolate as spi
import numpy as np
from approximation import create_plot

def f(x):
    return np.sin(x) + 0.5 * x

x = np.linspace(-2 * np.pi, 2 * np.pi, 25)
ipo = spi.splrep(   # linear spline interpolation
    x,      # data points defining the curve f(x) = y
    f(x), 
    k=1     # the degree; recommended - cubic i.e. 3
)

iy = spi.splev( # derive the interpolated values
    x,  # points at which to return the value of the smoothed spline 
    ipo # spline
)

print(np.allclose(f(x), iy))

create_plot([x,x], [f(x), iy], ['b', 'r.'], ['f(x)', 'interpolation'], ['x', 'f(x)'])
