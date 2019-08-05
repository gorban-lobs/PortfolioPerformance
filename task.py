import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, currencies, exchanges, prices, weights):
        self.currencies = currencies
        self.exchanges = exchanges
        self.prices = prices
        self.weights = weights

    def _calc_assets_returns(self, curr_df):
        shifted_df = curr_df.shift(1)
        return (curr_df - shifted_df) / shifted_df

    def calculate_asset_performance(self, start_date, end_date):
        asset_returns = _calc_asset_returns(self.prices)
        weighted_return = asset_returns.mul(self.weights).sum(axis='columns')

    def calculate_currency_performance(self, start_date, end_date):
        currency_returns = _calc_asset_returns(self.currencies)
        weighted_return = currency_returns.mul(self.weights).sum(axis='columns')

    def calculate_total_performance(self, start_date, end_date):
        total_returns = _calc_asset_returns(self.prices * self.currencies)
        weighted_return = total_returns.mul(self.weights).sum(axis='columns')
