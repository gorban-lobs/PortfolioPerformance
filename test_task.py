import unittest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from task import Portfolio


class TestPortfolio(unittest.TestCase):

    def test__init_current_dataframes(self):
        dates = pd.to_datetime(['2016-01-05', '2016-01-06', '2016-01-07'])
        cur = pd.Series(data=['A', 'B'], index=['x', 'y'])
        ex = pd.DataFrame(data=[[2],[3],[1]], index=dates, columns=['A'])
        pr = pd.DataFrame(data=[[2, 1], [3, 4], [5, 6]], index=dates, columns=['x', 'y'])
        w = pd.DataFrame(data=[[7, 8], [9, 10], [11, 12]], index=dates, columns=['y', 'x'])
        pf = Portfolio(cur, ex, pr, w)
        with self.subTest(i=1):
            pf._init_current_dataframes('2016-01-06', '2016-01-07')
            assert_frame_equal(pf.current_exch, 
                               pd.DataFrame(data=[[3], [1]], 
                                            index=dates[1:3], columns=['A']))
            assert_frame_equal(pf.current_prices, 
                               pd.DataFrame(data=[[3, 4], [5, 6]], 
                                            index=dates[1:3], columns=['x', 'y']))
            assert_frame_equal(pf.current_weights, 
                               pd.DataFrame(data=[[9, 10], [11, 12]], 
                                            index=dates[1:3], columns=['y', 'x']))
        with self.subTest(i=2):
            pf._init_current_dataframes('2016-01-03', '2016-01-06')
            assert_frame_equal(pf.current_exch, 
                               pd.DataFrame(data=[[2], [3]], 
                                            index=dates[:2], columns=['A']))
            assert_frame_equal(pf.current_prices, 
                               pd.DataFrame(data=[[2, 1], [3, 4]], 
                                            index=dates[:2], columns=['x', 'y']))
            assert_frame_equal(pf.current_weights, 
                               pd.DataFrame(data=[[7, 8], [9, 10]], 
                                            index=dates[:2], columns=['y', 'x']))


if __name__ == '__main__':
    unittest.main()