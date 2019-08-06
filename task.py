import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, currencies, exchanges, prices, weights):
        self.currencies = currencies
        self.exchanges = self._fill_nan_vals(exchanges)
        self.prices = self._fill_nan_vals(prices)
        self.weights = self._fill_nan_vals(weights)

    def _init_current_dataframes(self, start_date, end_date):
        self.current_exch = self.exchanges.loc[start_date:end_date, :]
        self.current_prices = self.prices.loc[start_date:end_date, :]
        self.current_weights = self.weights.loc[start_date:end_date, :]

    def _fill_nan_vals(self, dataframe):
        dataframe.index = pd.to_datetime(dataframe.index)
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
        if returns.columns.difference(self.current_weights.columns).empty:
            return returns.sort_index(axis='columns').mul(
                self.current_weights.sort_index(axis='columns')).sum(
                    axis='columns', skipna=False)
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

    def _convert_names_to_ids(self, currency_returns):
        res_dataframe = self.current_weights.copy()
        for cur_id in self.current_weights.columns:
            cur_name = self.currencies.at[cur_id]
            if cur_name == 'USD':
                res_dataframe.loc[:, cur_id] = np.array(
                    [1.0] * len(currency_returns))
            else:
                res_dataframe.loc[:, cur_id] = currency_returns.loc[:, cur_name]
        return res_dataframe

    def calculate_asset_performance(self, start_date, end_date):
        self._init_current_dataframes(start_date, end_date)
        asset_returns = self._calc_assets_returns(self.current_prices)
        weighted_returns = self._calc_weighted_returns(asset_returns)
        return self._calc_performance(weighted_returns, start_date)

    def calculate_currency_performance(self, start_date, end_date):
        self._init_current_dataframes(start_date, end_date)
        currency_returns = self._calc_assets_returns(self.current_exch)
        cur_returns_by_index = self._convert_names_to_ids(currency_returns)
        weighted_returns = self._calc_weighted_returns(cur_returns_by_index)
        return self._calc_performance(weighted_returns, start_date)

    def _get_price_exch_mul(self):
        result_dataframe = self.current_prices.copy()
        for cur_id in self.current_prices.columns:
            cur_name = self.currencies.at[cur_id]
            if cur_name == 'USD':
                result_dataframe.loc[:, cur_id] = self.current_prices.loc[:, cur_id]
            else:
                new_col = (self.current_prices.loc[:, cur_id] *
                          self.current_exch.loc[:, cur_name])
                result_dataframe.loc[:, cur_id] = new_col
        return result_dataframe

    def calculate_total_performance(self, start_date, end_date):
        self._init_current_dataframes(start_date, end_date)
        total_returns = self._calc_assets_returns(self._get_price_exch_mul())
        weighted_returns = self._calc_weighted_returns(total_returns)
        return self._calc_performance(weighted_returns, start_date)
