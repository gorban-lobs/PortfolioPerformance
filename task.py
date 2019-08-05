import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, currencies, exchanges, prices, weights):
        self.currencies = currencies
        self.exchanges = self._fill_nan_vals(exchanges)
        self.prices = self._fill_nan_vals(prices)
        self.weights = self._fill_nan_vals(weights)

    def _fill_nan_vals(self, dataframe):
        new_index = pd.date_range(start=dataframe.index.min(), 
                                  end=dataframe.index.max())
        result_dataframe = dataframe.reindex(new_index).fillna(method='ffill')
        if result_dataframe.isnull().any().any():
            return result_dataframe.fillna(method='bfill')
        return result_dataframe

    def _calc_assets_returns(self, curr_df):
        shifted_df = curr_df.shift(1)
        return curr_df.sub(shifted_df).div(shifted_df)

    def _calc_weighted_returns(self, returns):
        if returns.columns.difference(self.weights.columns).empty:
            return returns.sort_index(axis='columns').mul(
                self.weights.sort_index(axis='columns')).sum(axis='columns')
        else:
            print("Different column names in weights file and prices file")
            return pd.Series()

    def _calc_performance(self, series, start_date):
        date_list = np.array([start_date])
        perf_list = np.array([1])
        for ind in range(1, len(series)):
            date_list = np.append(date_list, series.index[ind])
            perf_list = np.append(
                                perf_list, 
                                (perf_list[ind - 1] * 
                                (1 + series.iat[ind])))
        return pd.Series(data=perf_list, index=date_list)

    def calculate_asset_performance(self, start_date, end_date):
        asset_returns = self._calc_assets_returns(self.prices)
        weighted_returns = self._calc_weighted_returns(asset_returns)
        return self._calc_performance(weighted_returns.loc[start_date:end_date],
                                      start_date)

    def _convert_names_to_ids(self, currency_returns):
        res_dataframe = pd.DataFrame()
        for cur_id in self.weights.columns:
            cur_name = self.currencies.at[cur_id]
            if cur_name == 'USD':
                res_dataframe.loc[:, cur_id] = np.array(
                    [1.0] * len(currency_returns))
            else:
                res_dataframe.loc[:, cur_id] = currency_returns.loc[:, cur_name]
        return res_dataframe

    def calculate_currency_performance(self, start_date, end_date):
        currency_returns = self._calc_assets_returns(self.exchanges)
        cur_returns_by_index = self._convert_names_to_ids(currency_returns)
        weighted_returns = self._calc_weighted_returns(cur_returns_by_index)
        return self._calc_performance(weighted_returns.loc[start_date:end_date], 
                                      start_date)

    def _get_price_exch_mul(self):
        result_dataframe = pd.DateFrame()
        for cur_id in self.prices.columns:
            cur_name = self.currencies.at[cur_id]
            new_col = (self.prices.loc[:, cur_id] *
                      self.exchanges.loc[:, cur_name])
            result_dataframe.loc[:, cur_id] = new_col
        return result_dataframe

    def calculate_total_performance(self, start_date, end_date):
        total_returns = self._calc_assets_returns(self._get_price_exch_mul)
        weighted_return = self._calc_weighted_returns(total_returns)
        return self._calc_performance(weighted_return.loc[start_date:end_date],
                                start_date)
