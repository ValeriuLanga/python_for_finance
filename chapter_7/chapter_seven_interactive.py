import pandas as pd
import numpy as np
import cufflinks as cf
import plotly.offline as plyo

a = np.random.standard_normal((250, 5)).cumsum(axis=0)
print(a)

index = pd.date_range('2024-01-01', 
                      freq='B',         # business frequency
                      periods=len(a))   # len of a periods needed
print(index)

df = pd.DataFrame(100 + 5 * a,
                  columns=list('abcde'),    # column headers
                  index=index)
print(df.head)

plyo.iplot(
    df.iplot(asFigure=True),
    image='png',
    filename='ply_01'
)

plyo.iplot(
    df[['a', 'b']]
        .iplot(asFigure=True,
            theme='polar',
            title='A time series plot',
            xTitle='date',
            yTitle='value',
            mode={'a': 'markers', 'b': 'lines+markers'},
            symbol={'a': 'circle', 'b': 'diamond'},
            size=3.5,
            colors={'a': 'blue', 'b': 'magenta'}        
        ),
    image='png',
    filename='ply_02'
    )

plyo.iplot(
    df.iplot(kind='hist',
             subplots=True,
             bins=15,
             asFigure=True
            ),
    image='png',
    filename='ply_03'
)