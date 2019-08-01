import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self):
        self.s = pd.Series([1, 3, 5, np.nan, 6, 8])

    def ps(self):
        print(self.s)


pf = Portfolio()
pf.ps()