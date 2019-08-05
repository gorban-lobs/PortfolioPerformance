import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, currencies, exchanges, prices, weights):
        self.currencies = currencies
        self.exchanges = _fill_nan_vals(exchanges)
        self.prices = _fill_nan_vals(prices)
        self.weights = _fill_nan_vals(weights)

    def _fill_nan_vals(dataframe):
        result_dataframe = dataframe.fillna(method='ffill')
        if result_dataframe.isnull().any().any():
            return result_dataframe.fillna(method='bfill')
        return result_dataframe

    def _calc_assets_returns(self, curr_df):
        shifted_df = curr_df.shift(1)
        return curr_df.sud(shifted_df).div(shifted_df)

    def _calc_performance(self, series, start_date):
        date_list = np.array([start_date])
        perf_list = np.array([1])
        for ind in range(1, len(series)):
            np.append(date_list, series.index[ind])
            np.append(perf_list, perf_list[ind - 1] * (1 + series.at[ind]))
        return pd.Series(data=perf_list, index=date_list)

    def calculate_asset_performance(self, start_date, end_date):
        asset_returns = _calc_asset_returns(self.prices)
        weighted_return = asset_returns.mul(self.weights).sum(axis='columns')
        return _calc_performance(weighted_return[start_date:end_date], 
                                start_date)

    def calculate_currency_performance(self, start_date, end_date):
        currency_returns = _calc_asset_returns(self.exchanges)
        weighted_return = currency_returns.mul(self.weights).sum(axis='columns')
        return _calc_performance(weighted_return[start_date:end_date], 
                                start_date)

    def calculate_total_performance(self, start_date, end_date):
        total_returns = _calc_asset_returns(self.prices * self.currencies)
        weighted_return = total_returns.mul(self.weights).sum(axis='columns')
        return _calc_performance(weighted_return[start_date:end_date],
                                start_date)
