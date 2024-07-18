import scipy.integrate as sci
import numpy as np
from pylab import plt, mpl

from matplotlib.patches import Polygon 


def f(x):
    return np.sin(x) + 0.5 * x

if __name__ == '__main__':
    x = np.linspace(0, 10)
    y = f(x)

    a = 0.5 # left integration lim
    b = 9.5 # right integration lim

    Ix = np.linspace(a, b)  # integration interval values
    Iy = f(Ix)  # integration func values

    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(x, y, 'b', linewidth=2)
    plt.ylim(bottom=0)

    verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)]
    poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
    ax.add_patch(poly)
    
    plt.text(
        0.75 * (a + b),
        1.5,
        r"$\int_a^b f(x)dx$",
        horizontalalignment='center',
        fontsize=20
    )
    
    plt.figtext(0.9, 0.075, '$x$')
    plt.figtext(0.075, 0.9, '$f(x)$')

    ax.set_xticks((a,b))
    ax.set_xticklabels(('$a$', '$b$'))
    ax.set_yticks([f(a), f(b)])

    # plt.show()

    #######################
    # numerical integration
    print(sci.fixed_quad(f, a, b)[0])   # fixed Gaussian quadrature
    print(sci.quad(f, a, b)[0])         # adaptive quadrature
    print(sci.romberg(f, a, b))      # romberg integration   

    # list / array as inputs
    xi = np.linspace(a, b, 25)
    print(sci.trapezoid(f(xi), xi))
    print(sci.simpson(y=f(xi), x=xi))

    # integration by simulation
    for i in range(1, 20):
        np.random.seed(1000)
        x = np.random.random(i * 10) * (b - a) + a  # more x-es each time 
        print(np.mean(f(x)) * (b - a))



