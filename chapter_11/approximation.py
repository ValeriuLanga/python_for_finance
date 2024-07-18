import numpy as np
import pandas as pd
from pylab import plt, mpl

def f(x):
    return np.sin(x) + 0.5 * x


def create_plot(x, y, styles, lables, axlables):
    plt.figure(figsize=(10,6))

    for i in range(len(x)):
        plt.plot(x[i], y[i], styles[i], label=lables[i])
        plt.xlabel(axlables[0])
        plt.ylabel(axlables[1])
    
    plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    ######################
    # let's just see the func
    x = np.linspace(-2 * np.pi, 2 * np.pi, 50)  # we're only looking at [-2*pi, 2*pi]
    # create_plot([x], [f(x)], ['b'], ['f(x)'], ['x', 'f(x)'])

    ######################
    # linear regression time
    # regression_coefs = np.polyfit(
    #     x,      # 50 values of x 
    #     f(x),   # sin() + 0.5 * x at 50 points
    #     deg=1,  # monomial
    #     full=True   
    # )
    # print(regression_coefs)

    # regression_vals = np.polyval(regression_coefs[0], x)    # if full=False we could just pass regression_coefs
    # create_plot([x,x],      # plot 2 x-based funcs
    #             [f(x), regression_vals],    # classic f(x) AND regression values
    #             ['b', 'r.'],    # f(x) is in blue and will look like a sinusoid; red dots for regression values
    #             ['f(x)', 'regression'],     # labels
    #             ['x','f(x)'])   # axis lables

    ######################
    # higher level monomials to better approximate sin() part of f() 
    # regression_coefs = np.polyfit(x, f(x), deg=5)
    # # print(regression_coefs)

    # regression_vals = np.polyval(regression_coefs, x)
    # # print(regression_vals)

    # create_plot([x,x],
    #             [f(x), regression_vals],
    #             ['b', 'r.'],
    #             ['f(x)', 'regression'],
    #             ['x', 'f(x)']            
    # )

    ######################
    # let's crank up the degree of the monomial 
    # regression_coefs = np.polyfit(x, f(x), deg=7)   # notice how higher values will grant improvements
    # regression_vals =  np.polyval(regression_coefs, x)
    # print(np.allclose(f(x), regression_vals))   # did we hit the jackpot?
    # print(np.mean((f(x) - regression_vals) ** 2))   # Mean Squared Error - MSE

    # create_plot([x, x],
    #             [f(x), regression_vals],
    #             ['b', 'r.'],
    #             ['f(x)', 'regression'],
    #             ['x', 'f(x)']
    # )

    ######################
    # individual basis funcs
    # matrix = np.zeros((3 + 1, len(x)))  # 3rd degree monomial + leading 1s
    # matrix[3, :] = x ** 3
    # matrix[2, :] = x ** 2
    # matrix[1, :] = x ** 1   # same as x but want to be clear
    # matrix[0, :] = x 
    # print(matrix)

    # # computes the vector that approximately solves a(x) = b
    # reg = np.linalg.lstsq(matrix.T, # transpose matrix of coefs - a
    #                       f(x),     # func - b
    #                       rcond=None)[0]    
    # reg = reg.round(4)
    # print(reg)

    # regression_values = np.dot(reg, matrix) # dot product of 2 matrices

    # create_plot(
    #     [x,x],
    #     [f(x), regression_values],
    #     ['b', 'r.'],
    #     ['f(x)', 'regression'],
    #     ['x', 'f(x)']  
    # )

    # matrix[3, :] = np.sin(x)    # replace highest order monomial w sine func since this is the obeserved behaviour
    #                             # adding a brand new column will yield the same result actually but with more computing power
    # reg = np.linalg.lstsq(matrix.T, f(x), rcond=None)[0]
    # reg = reg.round(4)

    # regression_values = np.dot(reg, matrix) # since we have the coefs as a matrix, we need to use the dot product

    # print(np.allclose(regression_values, f(x)))         # we're spot on
    # print(np.mean((f(x) - regression_values) ** 2))     # no deviation

    # create_plot([x, x],
    #             [f(x), regression_values],
    #             ['b', 'r.'],
    #             ['f(x)', 'regression'],
    #             ['x', 'f(x)']
    # )


    ######################
    # dealing with noisy data
    # xn = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    # xn = xn + 0.15 * np.random.standard_normal(len(xn)) # introducing some noise to x
    # yn = f(xn) + 0.25 * np.random.standard_normal(len(xn))  # same for y

    # reg_coefs = np.polyfit(xn, yn, 7)
    # print(reg_coefs)

    # reg_y = np.polyval(reg_coefs, yn)
    # create_plot(
    #     [x,x],
    #     [f(x), reg_y],
    #     ['b', 'r.'],
    #     ['f(x)', 'regression'],
    #     ['x', 'f(x)']
    # )


    ######################
    # unsorted data
    xu = np.random.rand(50) * 4 * np.pi - 2 * np.pi # just random, unsorted numbers; no more linear space
    yu = f(xu)

    reg_coefs = np.polyfit(xu, yu, 7)
    reg_vals = np.polyval(reg_coefs, xu)

    # MSE
    # 0.03487037154670556 @ 5th degree
    # 0.0010523494551617084 @ 7th degree
    print(np.mean((yu - reg_vals) ** 2))

    create_plot(
        [xu, xu],
        [yu, reg_vals],
        ['b.', 'r.'],   # we're doing dots for both - more like a scatter plot of sorts
        ['f(x)', 'regression'],
        ['x', 'f(x)']
    )

