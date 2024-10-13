import math
import numpy as np
import timeit

def chapter_one() :
    S0 = 100.0 # initial index level
    K = 105.0 # strike price
    T = 1.0 # time to maturity
    r = 0.05 # riskless short rate
    sigma = 0.2 # volatility
    I = 100_000 # no. simulations

    # algo
    z = np.random.standard_normal(I)

    # index values at maturity
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
    hT = np.maximum(ST - K, 0) # payoff at maturity
    print(hT)

    C0 = math.exp(-r * T) * np.mean(hT) # Monte Carlo estimator

    print('Value of the european call option %5.3f.' % C0)

def chapter_two():
    # a = np.array([0, 0.5, 1.0, 1.5, 2.0])
    # print(a)

    # #b = np.array(['a','b','c'])
    # #print(b)

    # print( a.sum() )
    # print( a.std() )
    # print( a.cumsum() )

    # a = np.arange(8, dtype=float)
    # # print(a)
    # # print(2 * a) # vectorized ops as opposed to duplicating the items in the std lib
    # # print(a ** 2)

    # # n-dimensions
    # b = np.array([a, a * 2])
    # print("Array b={}", b)
    # print(b[:,2]) # for each 1st level, get item at index 2
    # print("Sum of array b={}", b.sum())

    # print("Sum along the first axis (i.e. x) = {}", b.sum(axis=0))
    # print("Sum along the second axis (i.e. y) = {}", b.sum(axis=1))

    # # generate 
    # c = np.zeros((2,3), dtype=int, order='C')
    # print(c)

    # c = np.ones((2,3,4), dtype=int, order='C')
    # d = np.zeros_like(c, dtype=float, order='C')
    # print(d)

    # e = np.empty((2,3,2))
    # print(e)

    # print(np.eye(10))

    # g = np.linspace(5, 15, 12) # from 5 to 15, evenly-space, 12 values
    # print(g)
    # print(g.dtype)

    # # reshaping
    # g= np.arange(15)
    # print(g)
    # print(g.shape)
    # print(np.shape(g))

    # h = g.reshape((3,5))    # reshape won't affect g; just a view unless alloc-ed
    # print(h)
    # print(h.shape)
    # print(np.shape(h))

    # h = h.reshape((5,3))
    # print(h)
    # print(h.T)
    # print(h.transpose())    

    # # resizing below
    # print(g)
    # print(np.resize(g, (3,1)))
    # print(np.resize(g, (1, 5)))
    # print(np.resize(g, (5, 4))) # upsizing, elems repeated from start

    # # stacking
    # # dimension of ndarrays must be the same
    # h = np.arange(15).reshape((5, 3))
    # print(h)
    # print(np.hstack((h, 2 *h)))
    # print(np.vstack((h, 0.5 * h)))

    # # flatten / ravel
    # print(h)
    # print(h.flatten())
    # print(h.flatten(order='C'))
    # print(h.flatten(order='F'))
    # print(h.ravel())

    ########
    # boolean arrays
    # h = np.arange(15).reshape((5, 3))
    # print(h)
    # print(h > 8)
    # print(h <= 7)

    # # these will also flatten the data
    # print(h[h > 8])
    # print(h[(h > 4) & (h < 7)])


    # # where(...) allows for more complicated actions based on true / false
    # print(np.where(h > 7, 1, 0))    # put 1 if true, 0 if false
    # print(np.where(h % 2 == 0, 'even', 'odd'))    # even / odd
    # print(np.where(h <= 7, h * 2, h / 2))
    
    # structured numpy arrays
    dt = np.dtype([('Name', 'S10'), ('Age', 'i4'), ('Height', 'f'), ('Children/Pets', 'i4', 2)])
    print(dt)

    s = np.array([('Smith', 45, 1.83, (0,1)), ('Jones', 53, 1.72, (2,2)),], dtype=dt)
    print(s)
    print(type(s))

    print(s['Name'])
    print(s['Height'].mean())
    print(s[1]['Age'])

def f(x):
    return 3 * x + 5

def vectorization():
    # # basic vectorization
    # np.random.seed(100)
    r = np.arange(12).reshape((4,3))
    # s = np.arange(12).reshape((4,3)) * 0.5

    # print(r)
    # print(s)
    
    # print(r+s)
    
    # # numpy broadcasting - apply scalar to whole array
    # print(r + 3)
    # print(2 * r)

    # can do diff shapes as well - throwback to algebra group basics
    s = np.arange(0, 12, 4)
    print(s)
    print(s.shape)
    print(r)
    print(r.shape)

    print(s + r)

    s = np.arange(0, 12, 3)
    # however below won't work 
    # print(s + r)
    
    print(s)
    sr = s.reshape(-1, 1)
    print(sr)
    print(r + sr)

    # can call funcs as well
    print(f(r))


def memory_layout():
    x = np.random.standard_normal((100_000, 5)) # (rows, columns)
    y = 2 * x + 3
    C = np.array((x,y), order='C')
    F = np.array((x,y), order='F') 

    x = 0.0
    y = 0.0 # free up memory

    # print(C[:2].round(2))  # get some numbers from C and round them

    dummy_perf(C.sum)
    dummy_perf(F.sum)
    
    dummy_perf(C.sum, 0)
    dummy_perf(C.sum, 1)

    dummy_perf(F.sum, 0)
    dummy_perf(F.sum, 1)


def dummy_perf(f, param=None):
    start_time = timeit.default_timer()
    for _ in range(7):
        for _ in range(100):
            f(param)
    print(timeit.default_timer() - start_time)


def merge(left, right):
    """Merge two sorted lists into one sorted list."""
    merged = []
    i = j = 0
    # Merge the two lists into merged[]
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    # Append any remaining elements
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def iterative_merge_sort(arr):
    """Sort the array using iterative (non-recursive) merge sort."""
    width = 1
    n = len(arr)
    while width < n:
        for i in range(0, n, 2 * width):
            left = arr[i : i + width]
            right = arr[i + width : i + 2 * width]
            arr[i : i + 2 * width] = merge(left, right)
        width *= 2
    return arr

print(iterative_merge_sort([2,4,5,63,3,654,74,32,423,546]))
# chapter_one()
# chapter_two()
# vectorization()
# memory_layout()