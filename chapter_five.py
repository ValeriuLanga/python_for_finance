import matplotlib.style
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

def first_steps():
    df = pd.DataFrame([10,20,30,40],
                      columns=['numbers'],  # column label
                      index=['a','b','c','d'])
    print(df)
    # print("df.index {}", df.index)
    # print("df.columns {}", df.columns)
    # print("df.loc['c'] {}", df.loc['c'])
    
    # print(df.loc[['a','d']])
    # print(df.iloc[1:3]) # (1,3]

    # print(df.sum())
    # print(df.apply(lambda x: x ** 2))   # vectorized application via apply()
    # print(df ** 2)  # direct vectorization as w numpy ndarray

    # enlarging data frames
    df['floats'] = (1.5, 2.5, 3.5, 4.5)
    print(df)
    print(df['floats'])

    df['names'] = pd.DataFrame(['Yves', 'Sandra', 'Lili', 'Henry'],
                               index=['d','b','c','a']) # notice ordering of indices 
    print(df)   # indices are aligned!

    # apending used to be possible - concat is the way now
    # df.concat({'numbers' : 100, 'floats' : 5.75, 'names' : 'Jil'}, ignore_indexes=True)
    concat = pd.DataFrame({'numbers' : [100], 'floats' : [5.75], 'names' : ['Jil']})
    print(concat)
    print(pd.concat([df, concat], ignore_index=True)) # index swapped from letter range to int range!

    concat = pd.DataFrame({'numbers' : [100], 'floats' : [5.75], 'names' : ['Jil']}, index=['y'])
    print(concat)
    df = pd.concat([df, concat])
    print(df)   # index is still letter range

    incomplete_concat = pd.DataFrame({'names' : ['Liz']}, index=['y'])
    print(incomplete_concat)
    df = pd.concat([df, incomplete_concat])
    print(df)   # numers & floats have NaN

    # missing values don't mean we can't do operations; NaN rows are just ignored
    print(df[['numbers', 'floats']].mean())
    print(df[['numbers', 'floats']].std())


def second_steps():
    np.random.seed(1000)
    a = np.random.standard_normal((9,4)) # remember - (rows, columns)
    print(a)

    df = pd.DataFrame(a)
    print(df)

    df.columns = ['No1','No2','No3','No4']
    print(df)
    print(df['No2'].mean())

    # let's add some date information
    dates = pd.date_range('2024-06-16', periods=9, freq='ME')
    print(dates)

    df.index = dates    # essentially we now have a time series
    print(df)

    # we can go back to numpy array 
    print(df.values)
    print(a)
    print(np.array(df))

def basic_analytics():
    np.random.seed(1000)
    a = np.random.standard_normal((9,4))
    df = pd.DataFrame(a)
    df.columns = ['No1','No2','No3','No4']
    df.index = pd.date_range('2024-06-16', periods=9, freq='ME')
    print(df)

    print("\tdf.info()\n", df.info())
    print("\tdf.describe()\n", df.describe())
    print("\tdf.sum()\n", df.sum())
    print("\tdf.cumsum()\n", df.cumsum())
    print("\tdf.mean()\n", df.mean())
    print("\tdf.mean(axis=0)\n", df.mean(axis=0))   # column-wise, y axis
    print("\tdf.mean(axis=1)\n", df.mean(axis=1))   # row-wise, x axis

    # numpy funcs also work
    print("\tnp.mean(df)\n", np.mean(df, axis=0))   # specify axis, otherwise it will flatten the data frame
    print("\tnp.log(df)\n", np.log(df))
    print("\tnp.sqrt(abs(df))\n", np.sqrt(abs(df)))
    print("\t100 * df + 100\n", 100 * df + 100)

    # plt.style.use('seaborn')
    mpl.rcParams['font.family'] = 'serif'
    df.cumsum().plot(lw=2.0, figsize=(10, 6))
    plt.show()

    # we can also show a plotbar
    df.plot.bar()
    plt.show()


def series_class():
    np.random.seed(1000)
    df = pd.DataFrame(np.random.standard_normal((9,4)))
    df.columns = ['No1','No2','No3','No4']
    df.index = pd.date_range('2024-06-16', periods=9, freq='ME')

    s = pd.Series(np.linspace(0,15,7), name='series')
    print(s)
    print(type(s))

    s = df['No1']
    print(s)
    print(type(s))

    print(s.mean())

    s.plot(figsize=(10,6))
    plt.show()


def group_by():
    np.random.seed(1000)
    df = pd.DataFrame(np.random.standard_normal((9,4)))
    df.columns = ['No1','No2','No3','No4']
    df.index = pd.date_range('2024-01-01', periods=9, freq='ME')

    # can add a column indicating the quarter
    df['Quarter'] = ['Q1','Q1','Q1',
                     'Q2','Q2','Q2',
                     'Q3','Q3','Q3',]
    print(df)

    groups = df.groupby('Quarter')
    print(groups.size())
    print(groups.mean())
    print(groups.max())
    print(groups.aggregate('min', 'max').round(2))

    df['Odd_Even'] = ['Odd', 'Even','Odd', 'Even','Odd', 'Even','Odd', 'Even','Odd']
    print(df)
    groups = df.groupby(['Quarter', 'Odd_Even'])
    print(groups.size())

    print(groups[['No1', 'No4']].aggregate(['sum', 'mean']))


def complex_selection():
    data = np.random.standard_normal((10,2))
    df = pd.DataFrame(data, columns=['x','y'])
    print(df.info())

    print(df.head())
    print(df.tail())

    print(df['x'] > 0.5)
    print((df['x'] > 0)  | (df['y'] < 0))

    # below are equivalent
    print(df.query('x > 0'))
    print(df[df['x'] > 0])
    print(df[(df.x > 0)])

    # can compare full Data Frame as well
    print(df)
    print(df > 0)   # numbers that didn't match are False
    print(df[df>0]) # notice how numbers that didn't match are NaN


def data_frame_ops():
    # concatenating
    df1 = pd.DataFrame(['100','200','300','400'],
                       index=['a','b','c','d'],
                       columns=['A'])
    print(df1)

    df2 = pd.DataFrame(['200','150','50'],
                       index=['f','b','d'],
                       columns=['B'])
    print(df2)

    # print(pd.concat((df1,df2), sort=False)) # NaN for missing values

    # joining
    # print(df1.join(df2))                # left join i.e. only if in df1
    # print(df1.join(df2, how='left'))    # left join i.e. only if in df1
    # print(df1.join(df2, how='right'))   # right join i.e. only if in df2

    # print(df1.join(df2, how='inner'))   # only values found in both
    # print(df1.join(df2, how='outer'))   # all index values
    
    # joining on empty data frames - basically a left join
    # df = pd.DataFrame()
    # df['A'] = df1['A']
    # print(df)

    # df['B'] = df2['B']  # again, similar to left join - will only have values that have indices in A
    # print(df)

    # interestingly this will be equiv to an outer join since columns are created simultanously
    # print(pd.DataFrame({'A':df1['A'], 'B':df2['B']}))

    # merging - we care about the values on a columns, not the indices
    c = pd.Series([250, 150, 50], index=['b', 'c', 'd'])
    print(c)
    df1['C'] = c
    df2['C'] = c

    print(df1)
    print(df2)

    print(pd.merge(df1, df2))
    print(pd.merge(df1, df2, how='inner'))  # same thing as above

    print(pd.merge(df1, df2, how='outer'))  # all values

    print(pd.merge(df1, df2, left_on='A', right_on='B'))    # match on values from df1.A == df2.B;
    print(pd.merge(df1, df2, left_index=True, right_index=True))


# first_steps()
# second_steps()
# basic_analytics()
# series_class()
# group_by()
# complex_selection()
data_frame_ops()


