import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self):
        self.CURRENCIES_FILE = 'currencies.csv'
        self.EXCHANGES_FILE = 'exchanges.csv'
        self.PRICES_FILE = 'prices.csv'
        self.WEIGHTS_FILE = 'weights.csv'
        self.currencies = pd.read_csv(self.CURRENCIES_FILE, index_col=0)
        self.exchanges = pd.read_csv(self.EXCHANGES_FILE, index_col=0))
        self.prices = pd.read_csv(self.PRICES_FILE, index_col=0))
        self.weights = pd.read_csv(self.WEIGHTS_FILE, index_col=0)

    def _count_return(self, curr_df):
        shifted_df = curr_df.shift(1, fill_value=0)
        return curr_df.sub(shifted_df)

    def calculate_asset_performance(self, start_date, end_date):
        pass

    def calculate_currency_performance(self, start_date, end_date):
        pass

    def calculate_total_performance(self, start_date, end_date):
        pass


pf = Portfolio()
