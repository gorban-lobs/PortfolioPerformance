import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self):
        pass

    def _count_return(self, curr_df):
        shifted_df = curr_df.shift(1)
        return (curr_df - shifted_df) / shifted_df

    def calculate_asset_performance(self, start_date, end_date):
        pass

    def calculate_currency_performance(self, start_date, end_date):
        pass

    def calculate_total_performance(self, start_date, end_date):
        pass


pf = Portfolio()
