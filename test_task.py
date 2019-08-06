import unittest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from task import Portfolio


class TestPortfolio(unittest.TestCase):
    
    def setUp(self):
        z_series = pd.Series([0])
        z_df = pd.DataFrame([0])
        self.pf = Portfolio(z_series, z_df, z_df, z_df)

    def test__init_current_dataframes(self):
        dates = pd.to_datetime(['2016-01-05', '2016-01-06', '2016-01-07'])
        cur = pd.Series(data=['A', 'B'], index=['x', 'y'])
        ex = pd.DataFrame(data=[[2],[3],[1]], index=dates, columns=['A'])
        pr = pd.DataFrame(data=[[2, 1], [3, 4], [5, 6]], index=dates, columns=['x', 'y'])
        w = pd.DataFrame(data=[[7, 8], [9, 10], [11, 12]], index=dates, columns=['y', 'x'])
        pf = Portfolio(cur, ex, pr, w)
        with self.subTest(i=0):
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
        with self.subTest(i=1):
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

    def test__fill_nan_vals(self):
        dates = pd.to_datetime(['2016-01-05', '2016-01-06', '2016-01-07'])
        test_tuple = (
            (
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                [dates[0], dates[1]],
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                [dates[0], dates[1]]
            ),
            (
                [[1.0, 2.0, np.nan], [np.nan, 2.0, 3.0]],
                [dates[0], dates[1]],
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                [dates[0], dates[1]],
            ),
            (
                [[np.nan, 2.0, np.nan], [2.0, np.nan, 2.0]],
                [dates[0], dates[1]],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                [dates[0], dates[1]]
            ),
            (
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [dates[0], dates[2]],
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dates
            ),
            (
                [[np.nan, np.nan, 3.0], [1.0, 2.0, np.nan]],
                [dates[0], dates[2]],
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                dates
            )
        )

        for i in range(len(test_tuple)):
            with self.subTest(i=i):
                test_df = pd.DataFrame(test_tuple[i][0], 
                                       index=test_tuple[i][1])
                res_df = pd.DataFrame(test_tuple[i][2],
                                      index=test_tuple[i][3])
                assert_frame_equal(self.pf._fill_nan_vals(test_df),
                                   res_df)


if __name__ == '__main__':
    unittest.main()