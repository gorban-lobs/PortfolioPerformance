import unittest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
from task import Portfolio


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        z_series = pd.Series([0])
        z_df = pd.DataFrame([0])
        self.pf = Portfolio(z_series, z_df, z_df, z_df)

    def test__init_current_dataframes(self):
        dates = pd.to_datetime(['2016-01-05', '2016-01-06', '2016-01-07'])
        cur = pd.Series(data=['A', 'B'], index=['x', 'y'])
        ex = pd.DataFrame(data=[[2], [3], [1]], index=dates, columns=['A'])
        pr = pd.DataFrame(
            data=[[2, 1], [3, 4], [5, 6]], index=dates, columns=['x', 'y'])
        w = pd.DataFrame(
            data=[[7, 8], [9, 10], [11, 12]], index=dates, columns=['y', 'x'])
        pf = Portfolio(cur, ex, pr, w)
        with self.subTest(i=0):
            pf._init_current_dataframes('2016-01-06', '2016-01-07')
            assert_frame_equal(pf.current_exch,
                               pd.DataFrame(
                                            data=[[3], [1]],
                                            index=dates[1:3], columns=['A']))
            assert_frame_equal(pf.current_prices,
                               pd.DataFrame(
                                            data=[[3, 4], [5, 6]],
                                            index=dates[1:3],
                                            columns=['x', 'y']))
            assert_frame_equal(pf.current_weights,
                               pd.DataFrame(
                                            data=[[9, 10], [11, 12]],
                                            index=dates[1:3],
                                            columns=['y', 'x']))
        with self.subTest(i=1):
            pf._init_current_dataframes('2016-01-03', '2016-01-06')
            assert_frame_equal(pf.current_exch,
                               pd.DataFrame(
                                            data=[[2], [3]],
                                            index=dates[:2], columns=['A']))
            assert_frame_equal(pf.current_prices,
                               pd.DataFrame(
                                            data=[[2, 1], [3, 4]],
                                            index=dates[:2],
                                            columns=['x', 'y']))
            assert_frame_equal(pf.current_weights,
                               pd.DataFrame(
                                            data=[[7, 8], [9, 10]],
                                            index=dates[:2],
                                            columns=['y', 'x']))

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
                test_df = pd.DataFrame(
                                       test_tuple[i][0],
                                       index=test_tuple[i][1])
                res_df = pd.DataFrame(
                                      test_tuple[i][2],
                                      index=test_tuple[i][3])
                assert_frame_equal(
                    self.pf._fill_nan_vals(test_df),
                    res_df)

    def test__calc_assets_returns(self):
        test_tuple = (
            ([1, 1, 1], [np.nan, 0, 0]),
            ([1, 2, 3, 4], [np.nan, 1, 1/2, 1/3]),
            ([2, -2, 2, -2], [np.nan, -2, -2, -2]),
            ([1, -3, 10, 4], [np.nan, -4, -13/3, -0.6]),
        )
        for i in range(len(test_tuple)):
            with self.subTest(i=i):
                test_df = pd.DataFrame(test_tuple[i][0])
                res_df = pd.DataFrame(test_tuple[i][1])
                assert_frame_equal(
                    self.pf._calc_assets_returns(test_df),
                    res_df)

    def test__calc_weighted_returns(self):
        test_tuple = (
            (
                [[1, 1], [0, 1]], ['A', 'B'],
                [[2, 1], [3, 1]], ['A', 'B'],
                [3, 1]
            ),
            (
                [[1, 1], [0, 1]], ['A', 'C'],
                [[2, 1], [3, 1]], ['A', 'B'],
                [],
            ),
            (
                [[1, 1], [0, 1]], ['B', 'A'],
                [[2, 1], [3, 1]], ['A', 'B'],
                [3, 3]
            ),
            (
                [[1, 2, 3], [0, 1, 0], [1, 2, 0]], ['B', 'A', 'C'],
                [[2, 1, 0], [3, 1, 2], [1, 1, 1]], ['A', 'B', 'C'],
                [5, 3, 3]
            ),
        )
        for i in range(len(test_tuple)):
            with self.subTest(i=i):
                test_df1 = pd.DataFrame(test_tuple[i][0],
                                        columns=test_tuple[i][1])
                test_df2 = pd.DataFrame(test_tuple[i][2],
                                        columns=test_tuple[i][3])
                res_df = pd.Series(test_tuple[i][4])
                assert_series_equal(
                    self.pf._calc_weighted_returns(
                                                test_df1,
                                                test_df2),
                    res_df)

    def test__calc_performance(self):
        dates = pd.to_datetime(['2016-01-05', '2016-01-06',
                                '2016-01-07', '2016-01-08'])
        test_tuple = (
            (
                [1, 2, 3, 4],
                [1, 3, 12, 60]
            ),
            (
                [1, 1, 1, 1],
                [1, 2, 4, 8]
            ),
            (
                [0.5, 0.25, 1, -0.5],
                [1, 1.25, 2.5, 1.25]
            ),
        )
        for i in range(len(test_tuple)):
            with self.subTest(i=i):
                test_df = pd.Series(
                                    test_tuple[i][0],
                                    index=dates)
                res_df = pd.Series(
                                   test_tuple[i][1],
                                   index=dates)
                assert_series_equal(
                    self.pf._calc_performance(
                                              test_df,
                                              dates[0]),
                    res_df)

    def test__convert_names_to_ids(self):
        columns = ['u1', 'a1', 'b1', 'a2']
        sample_df = pd.DataFrame(np.random.randn(2, 4),
                                 columns=columns)
        currencies = pd.Series(['USD', 'A', 'B', 'A'],
                               index=columns)
        test_tuple = (
            (
                [[2.0, 3.0], [4.0, 5.0]],
                ['A', 'B'],
                [[0.0, 2.0, 3.0, 2.0], [0.0, 4.0, 5.0, 4.0]],
            ),
            (
                [[0.0, 2.0], [3.0, 4.0]],
                ['B', 'A'],
                [[0.0, 2.0, 0.0, 2.0], [0.0, 4.0, 3.0, 4.0]]
            ),
        )
        for i in range(len(test_tuple)):
            with self.subTest(i=i):
                test_df = pd.DataFrame(test_tuple[i][0],
                                       columns=test_tuple[i][1])
                res_df = pd.DataFrame(test_tuple[i][2],
                                      columns=columns)
                assert_frame_equal(
                    self.pf._convert_names_to_ids(
                                                  test_df,
                                                  sample_df,
                                                  currencies),
                    res_df)

    def test__get_price_exch_mul(self):
        prices_columns = ['u1', 'a1', 'b1', 'a2']
        currencies = pd.Series(['USD', 'A', 'B', 'A'],
                               index=prices_columns)
        test_tuple = (
            (
                [[2.0, 1.0], [1.0, 2.0]],
                ['A', 'B'],
                [[1.0, 2.0, 3.0, 2.0], [1.0, 4.0, 5.0, 4.0]],
                [[1.0, 4.0, 3.0, 4.0], [1.0, 4.0, 10.0, 4.0]],
            ),
            (
                [[0.0, 2.0], [3.0, 4.0]],
                ['B', 'A'],
                [[1.0, 2.0, 0.0, 2.0], [1.0, 4.0, 3.0, 4.0]],
                [[1.0, 4.0, 0.0, 4.0], [1.0, 16.0, 9.0, 16.0]],
            ),
        )
        for i in range(len(test_tuple)):
            with self.subTest(i=i):
                test_ex = pd.DataFrame(test_tuple[i][0],
                                       columns=test_tuple[i][1])
                test_pr = pd.DataFrame(test_tuple[i][2],
                                       columns=prices_columns)
                res_df = pd.DataFrame(test_tuple[i][3],
                                      columns=prices_columns)
                assert_frame_equal(
                    self.pf._get_price_exch_mul(
                                                test_pr,
                                                test_ex,
                                                currencies),
                    res_df)

    def test_calculate_asset_performance(self):
        dates = pd.to_datetime(['2016-01-01', '2016-01-02', '2016-01-03'])
        c_cur = pd.Series(data=['EUR', 'USD'],
                          index=['DE0007164600 GR', 'US0527691069 US'])
        test_tuple = (
            (
                [[2], [3], [2]],
                [[2, 1], [1, 2], [1, 1]],
                [[1, 2], [1, 2], [2, 1]],
                [1, 1, 0.0]
            ),
            (
                [[2], [3], [2]],
                [[2, 2], [3, 2], [2, 1]],
                [[1, 0], [2, 2], [1, 1]],
                [1, 2, 1 / 3]
            )
        )
        for i in range(len(test_tuple)):
            with self.subTest(i=i):
                c_ex = pd.DataFrame(
                                    data=test_tuple[i][0],
                                    index=dates, columns=['EUR'])
                c_pr = pd.DataFrame(
                                    data=test_tuple[i][1],
                                    index=dates,
                                    columns=[
                                        'DE0007164600 GR',
                                        'US0527691069 US'])
                c_w = pd.DataFrame(
                                   data=test_tuple[i][2],
                                   index=dates,
                                   columns=[
                                       'US0527691069 US',
                                       'DE0007164600 GR'])
                pf = Portfolio(c_cur, c_ex, c_pr, c_w)
                res_series = pd.Series(data=test_tuple[i][3], index=dates)
                assert_series_equal(
                    pf.calculate_asset_performance(dates[0], dates[2]),
                    res_series)

    def test_calculate_currency_performance(self):
        dates = pd.to_datetime(['2016-01-01', '2016-01-02', '2016-01-03'])
        c_cur = pd.Series(data=['EUR', 'USD'],
                          index=['DE0007164600 GR', 'US0527691069 US'])
        test_tuple = (
            (
                [[2], [3], [2]],
                [[2, 1], [1, 2], [1, 1]],
                [[1, 2], [1, 2], [2, 1]],
                [1, 2, 4 / 3]
            ),
            (
                [[2], [4], [3]],
                [[2, 2], [3, 2], [2, 1]],
                [[1, 0], [2, 2], [1, 1]],
                [1, 3, 2.25]
            )
        )
        for i in range(len(test_tuple)):
            with self.subTest(i=i):
                c_ex = pd.DataFrame(
                                    data=test_tuple[i][0],
                                    index=dates, columns=['EUR'])
                c_pr = pd.DataFrame(data=test_tuple[i][1],
                                    index=dates,
                                    columns=[
                                        'DE0007164600 GR',
                                        'US0527691069 US'])
                c_w = pd.DataFrame(
                                   data=test_tuple[i][2],
                                   index=dates,
                                   columns=[
                                       'US0527691069 US',
                                       'DE0007164600 GR'])
                pf = Portfolio(c_cur, c_ex, c_pr, c_w)
                res_series = pd.Series(data=test_tuple[i][3], index=dates)
                assert_series_equal(
                    pf.calculate_currency_performance(dates[0], dates[2]),
                    res_series)

    def test_calculate_total_performance(self):
        dates = pd.to_datetime(['2016-01-01', '2016-01-02', '2016-01-03'])
        c_cur = pd.Series(data=['EUR', 'USD'],
                          index=['DE0007164600 GR', 'US0527691069 US'])
        test_tuple = (
            (
                [[2], [3], [2]],
                [[2, 1], [1, 2], [1, 1]],
                [[1, 2], [1, 2], [2, 1]],
                [1, 1.5, -1 / 3 * 1.5]
            ),
            (
                [[2], [4], [3]],
                [[2, 2], [3, 3], [2, 1]],
                [[1, 0], [2, 2], [1, 1]],
                [1, 6, -1 / 6 * 6]
            )
        )
        for i in range(len(test_tuple)):
            with self.subTest(i=i):
                c_ex = pd.DataFrame(
                                    data=test_tuple[i][0],
                                    index=dates, columns=['EUR'])
                c_pr = pd.DataFrame(
                                    data=test_tuple[i][1],
                                    index=dates,
                                    columns=[
                                        'DE0007164600 GR',
                                        'US0527691069 US'])
                c_w = pd.DataFrame(
                                   data=test_tuple[i][2],
                                   index=dates,
                                   columns=[
                                       'US0527691069 US',
                                       'DE0007164600 GR'])
                pf = Portfolio(c_cur, c_ex, c_pr, c_w)
                res_series = pd.Series(data=test_tuple[i][3], index=dates)
                assert_series_equal(
                    pf.calculate_total_performance(dates[0], dates[2]),
                    res_series)


if __name__ == '__main__':
    unittest.main()
