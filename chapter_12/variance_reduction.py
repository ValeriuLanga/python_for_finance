import numpy as np
import numpy.random as npr

if __name__ == '__main__':
    print('%15s %15s' % ('Mean', 'Std Deviation'))
    print(31 * '-')

    # standard normal
    for i in range(1, 32, 2):
        npr.seed(1000)
        sn = npr.standard_normal(i ** 2 * 1_000)

        print('%15.12f %15.12f' % (sn.mean(), sn.std()))

    # antithetic variates
    print('%15s %15s' % ('Mean', 'Std Deviation'))
    print(31 * '-')
    for i in range(1, 32, 2):
        npr.seed(1000)
        sn = npr.standard_normal(i ** 2 * int(1_000 / 2))
        sn = np.concatenate((sn, -sn))

        print('%15.12f %15.12f' % (sn.mean(), sn.std()))
    # std deviation still a bit all over the place

    # moment matching
    print('%15s %15s' % ('Mean', 'Std Deviation'))
    print(31 * '-')
    for i in range(1, 32, 2):
        npr.seed(1000)
        sn = npr.standard_normal(i ** 2 * 1_000)
        sn = (sn - sn.mean()) / sn.std()    # secret sauce here

        print('%15.12f %15.12f' % (sn.mean(), sn.std()))

