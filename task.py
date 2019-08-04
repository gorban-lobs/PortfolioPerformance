import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self):
        self.CURRENCIES_FILE = 'currencies.csv'
        self.EXCHANGES_FILE = 'exchanges.csv'
        self.PRICES_FILE = 'prices.csv'
        self.WEIGHTS_FILE = 'weights.csv'
        self.currencies = pd.read_csv(self.CURRENCIES_FILE)
        self.exchanges = pd.read_csv(self.EXCHANGES_FILE)
        self.prices = pd.read_csv(self.PRICES_FILE)
        self.weights = pd.read_csv(self.WEIGHTS_FILE)
        
        self.s = pd.Series([1, 3, 5, np.nan, 6, 8])

    def calculate asset performance(self, start date, end date):
        pass

    def calculate currency performance(self, start date, end date):
        pass

    def calculate total performance(self, start date, end date):
        pass


pf = Portfolio()