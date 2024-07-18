import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon


def single_dimensional_data():
    np.random.seed(1000)
    y = np.random.standard_normal(20)   
    x = np.arange(len(y))   # values from 0 to len(y) - [0,20)

    print(y)    # there's only one dimension to y

    plt.plot(x,y)
    plt.show()

    plt.plot(y) # this is the same since x in the prev example is just indices
    plt.show()

    plt.plot(y.cumsum())
    plt.show()

    plt.plot(y.cumsum())
    plt.grid(False)     # no grid overlay
    plt.axis('equal')   # equal scaling for axes
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(y.cumsum(), 'ro')  # plots the data as thick red dots; this will remove the line connecting them!
    plt.plot(y.cumsum(), 'b', lw=1.5) # this adds a line connecting the dots - in blue
    plt.xlabel('index')
    plt.ylabel('value')
    plt.title('A simple plot')
    plt.show()


def two_dimensional_data():
    y = np.random.standard_normal((20,2)).cumsum(axis=0)
    
    print(y)    # 2 deminesions here - as per the ndarray (20, 2) 

    plt.plot(y, lw=1.5)
    plt.plot(y, 'ro')

    plt.xlabel('index')
    plt.ylabel('values')
    plt.title('A simple plot')
    
    plt.show()

    # let's add some labels to the individual data sets and also a legend
    plt.plot(y[:, 0], lw=1.5, label='1st')
    plt.plot(y[:,1], lw=1.5, label='2nd')

    plt.plot(y, 'ro')   # r=red o=circle -> red circle
    plt.legend(loc=0)   # put legend in 'best' location

    plt.xlabel('index')
    plt.ylabel('values')
    plt.title('A simple plot')

    plt.show()

    # different amplitude of data might cause issues rendering
    y[:, 0] = y[:, 0] * 100 # change amplitude

    plt.plot(y[:, 0], lw=1.5, label='1st')
    plt.plot(y[:,1], lw=1.5, label='2nd')

    plt.plot(y, 'ro')   # r=red o=circle -> red circle
    plt.legend(loc=0)   # put legend in 'best' location

    plt.xlabel('index')
    plt.ylabel('values')
    plt.title('A simple plot')

    plt.show()

    # solution 1 - introduce an extra y-axis - plots effectively overlay
    fig, ax1 = plt.subplots()
    plt.plot(y[:, 0], 'b', lw=1.5, label='1st')
    plt.plot(y[:, 0], 'ro')
    plt.legend(loc=8)

    plt.xlabel('index')
    plt.ylabel('value 1st')
    plt.title('A simple plot')

    # ok so one thing to understand about pyplot is that it is stateful
    # whenever we call plt.(...) we change the state of the plot
    # that's why we don't need to explicitly call ax2.method(...)
    # see pyplot.py for more info
    ax2 = ax1.twinx()
    plt.plot(y[:, 1], 'g', lw=1.5, label='2nd')
    plt.plot(y[:,1], 'ro')
    plt.legend(loc=0)
    
    plt.ylabel('value 2nd')

    plt.show()

    # solution 2 - separate subplots
    plt.figure(figsize=(10,6))

    plt.subplot(211)    # 3 coordinates - (numrows, numcols, fignum)
                        # fignum in between [1, numrows * numcols] ; start from top-left
    plt.plot(y[:,0], 'b', lw=1.5, label='first')
    plt.plot(y[:,0], 'ro')
    plt.legend(loc=0)
    plt.ylabel('value')
    plt.title('2 separate subplots')

    plt.subplot(2, 1, 2)    # same 3 coords but now individually!
    plt.plot(y[:,1], 'g', lw=1.5, label='first')
    plt.plot(y[:,1], 'ro')
    plt.legend(loc=0)
    plt.xlabel('index')
    plt.ylabel('value')
    
    plt.show()

    # combining 2 different subplot types
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(y[:,0], lw=1.5, label='1st')
    plt.plot(y[:,0], 'ro')
    plt.legend(loc=0)
    plt.xlabel('index')
    plt.ylabel('value')
    plt.title('1st data set')

    plt.subplot(122)
    plt.bar(np.arange(len(y)), y[:,1], width=0.5, color='g', label='2nd')
    plt.legend(loc=0)
    plt.xlabel('index')
    plt.title('2nd data set')

    plt.show()


def histogram():
    y = np.random.standard_normal((1000,2))
    print(y)

    plt.figure(figsize=(10,6))
    plt.hist(y, label=['1st', '2nd'], bins=25)
    plt.legend(loc=0)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title('histogram')

    plt.show()

    # stacked display
    plt.figure(figsize=(10,6))
    plt.hist(y, label=['1st', '2nd'], color=['b', 'g'], stacked=True, bins=20, alpha=0.5)   # alpha = opacity; closer to 0 means more transparent
    plt.legend(loc=0)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title('histogram')

    plt.show()


def box_plot():
    y = np.random.standard_normal((100, 2))
    print(y)

    fig, ax = plt.subplots(figsize=(10,6))
    plt.boxplot(y)
    plt.setp(ax, xticklabels=['1st', '2nd'])
    plt.xlabel('data set')
    plt.ylabel('value')
    plt.title('Boxplot')

    plt.show()


def func(x):
    return 0.5 * np.exp(x) + 1


def integral_value():
    a, b = 0.5, 1.5 # 1/2 and 3/2
    x = np.linspace(0, 2)   # x values to plot the func
    y = func(x)

    Ix = np.linspace(a, b) # integral limits
    Iy = func(Ix)   # integral values at x pts

    # print(Iy)
    verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)] # coords to plot the polygon

    # math done, time to render
    fix, ax = plt.subplots(figsize=(10,6))
    plt.plot(x, y, 'b', linewidth=2)
    plt.ylim(bottom=0)

    poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
    ax.add_patch(poly)  # at this point we have the integral area as a polygon in gray !

    plt.text(0.5 * (a + b),
             1, 
             r'$\int_a^b f(x)\mathrm{d}x$', # black magic to show integral symbol - LaTex
             horizontalalignment='center',
             fontsize=20)
    plt.figtext(0.9, 0.075, '$x$')      # place axis labels
    plt.figtext(0.075, 0.9, '$y$')
    ax.set_xticks((a, b))   # only care about integral limits on x axis
    ax.set_xticklabels(('$a$', '$b$'))

    ax.set_yticks((func(a), func(b)))   # func values on y only
    ax.set_yticklabels(('$f(a)$', '$f(b)$'))

    plt.show()

# single_data()
# two_dimensional_data()
# histogram()
# box_plot()
integral_value()

