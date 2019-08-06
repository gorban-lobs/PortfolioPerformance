import numpy as np
import pandas as pd


class Portfolio:
    def __init__(
            self, 
            currencies: pd.Series, 
            exchanges: pd.DataFrame, 
            prices: pd.DataFrame,
            weights: pd.DataFrame
        ) -> None:
        """ 
        Initialize Portfolio class.  
  
        Parameters: 
        currencies: asset currency
                    columns: currency, rows: asset id
        exchanges: daily exchange rate (to dollar),
                   columns: currency names, rows: dates
        prices: daily asset prices 
                columns: asset ids, rows: dates
        weights: daily portfolio weights
                 columns: asset id, rows: dates
  
        Returns: 
        None
  
        """
 
        self.currencies = currencies
        self.exchanges = self._fill_nan_vals(exchanges)
        self.prices = self._fill_nan_vals(prices)
        self.weights = self._fill_nan_vals(weights)

    def _init_current_dataframes(
                self, 
                start_date: pd.Timestamp, 
                end_date: pd.Timestamp
        ) -> None:
        """ 
        Initialize data slices according to date interval.
        Create:
            self.current_exch from self.exchanges
            self.current_prices from self.prices
            self.current_weights from self.weights

        Parameters: 
        start_date: start of the interval
        end_date: end of the interval
  
        Returns: 
        None
  
        """

        self.current_exch = self.exchanges.loc[start_date:end_date, :]
        self.current_prices = self.prices.loc[start_date:end_date, :]
        self.current_weights = self.weights.loc[start_date:end_date, :]

    def _fill_nan_vals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """ 
        Fill NaN values and data gaps in dataframe with previous valid value.
        If there is no previous values fill it with next valid value.

        Parameters: 
        dataframe: current dataframe
  
        Returns: 
        Filled dataframe
  
        """

        dataframe.index = pd.to_datetime(dataframe.index)
        new_index = pd.date_range(start=dataframe.index.min(), 
                                  end=dataframe.index.max())
        result_dataframe = dataframe.reindex(new_index).fillna(method='ffill')
        if result_dataframe.isnull().any().any():
            return result_dataframe.fillna(method='bfill')
        return result_dataframe

    def _calc_assets_returns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """ 
        Calculate dataframe of assets returns

        Parameters: 
        dataframe: current dataframe
  
        Returns: 
        Dataframe with assets returns
  
        """

        shifted_df = dataframe.shift(1)
        return dataframe.sub(shifted_df).div(shifted_df)

    def _calc_weighted_returns(
            self, 
            returns: pd.DataFrame, 
            current_weights: pd.DataFrame
        ) -> pd.Series:
        """ 
        Calculate series of weighted sums of current dataframe rows.
        If column names in current dataframe and weights dataframe are not equal
        return empty series

        Parameters: 
        returns: current dataframe, 
        current_weights: current weights
  
        Returns: 
        Series of weighted sums
  
        """

        if returns.columns.difference(current_weights.columns).empty:
            return returns.sort_index(axis='columns').mul(
                current_weights.sort_index(axis='columns')).sum(
                    axis='columns', skipna=False)
        else:
            print("Different column names in weights file and prices file")
            return pd.Series([])

    def _calc_performance(
            self, 
            series: pd.Series, 
            start_date: pd.Timestamp
        ) -> pd.Series:
        """ 
        Calculate portfolio performance per days

        Parameters:
        series: current series, 
        start_date: first date of the interval
  
        Returns: 
        Series of portfolio performances
  
        """

        date_list = np.array([pd.to_datetime(start_date)])
        perf_list = np.array([1])
        for ind in range(1, len(series)):
            date_list = np.append(date_list, pd.to_datetime(series.index[ind]))
            perf_list = np.append(
                                perf_list, 
                                (perf_list[ind - 1] * 
                                (1 + series.iat[ind])))
        return pd.Series(data=perf_list, index=date_list)

    def _convert_names_to_ids(
            self, 
            currency_returns: pd.DataFrame, 
            sample_df: pd.DataFrame,
            currencies: pd.Series
        ) -> pd.DataFrame:
        """ 
        Create dataframe with currency indices instead of currency 
        as column names.
        If one name matches multiple indices there would be multiple columns 
        in the resulting datframe

        Parameters:
        currency_returns: current dataframe, 
        sample_df: sample dataframe with indices as column names,
        currencies: asset currency. Columns: currency, rows: asset id
  
        Returns: 
        Dataframe of with currency indices as column names
  
        """

        res_dataframe = sample_df.copy()
        for cur_id in sample_df.columns:
            cur_name = currencies.at[cur_id]
            if cur_name == 'USD':
                res_dataframe.loc[:, cur_id] = np.array(
                    [0.0] * len(currency_returns))
            else:
                res_dataframe.loc[:, cur_id] = currency_returns.loc[:, cur_name]
        return res_dataframe

    def calculate_asset_performance(
            self, 
            start_date: pd.Timestamp, 
            end_date: pd.Timestamp
        ) -> pd.Series:
        """ 
        Create series with asset portfolio performance per day

        Parameters:
        start_date: start of the interval
        end_date: end of the interval
  
        Returns: 
        Series with asset portfolio performance per day
  
        """

        self._init_current_dataframes(start_date, end_date)
        asset_returns = self._calc_assets_returns(self.current_prices)
        weighted_returns = self._calc_weighted_returns(asset_returns,
                                                       self.current_weights)
        return self._calc_performance(weighted_returns, start_date)

    def calculate_currency_performance(
            self, 
            start_date: pd.Timestamp, 
            end_date: pd.Timestamp
        ) -> pd.Series:
        """ 
        Create series with currency portfolio performance per day

        Parameters:
        start_date: start of the interval
        end_date: end of the interval
  
        Returns: 
        Series with currency portfolio performance per day
  
        """
        
        self._init_current_dataframes(start_date, end_date)
        currency_returns = self._calc_assets_returns(self.current_exch)
        cur_returns_by_index = self._convert_names_to_ids(currency_returns,
                                                        self.current_weights,
                                                        self.currencies)
        weighted_returns = self._calc_weighted_returns(cur_returns_by_index,
                                                       self.current_weights)
        return self._calc_performance(weighted_returns, start_date)

    def _get_price_exch_mul(
            self, 
            current_prices: pd.DataFrame, 
            current_exch: pd.DataFrame, 
            currencies: pd.DataFrame
        ) -> pd.DataFrame:
        """ 
        Create dataframe with currencies exchange rates and prices 
        multiplication

        Parameters:
        current_prices: current prices dataframe, 
        current_exch: current currencies exchange rates dataframe, 
        currencies: asset currency. Columns: currency, rows: asset id
  
        Returns: 
        Dataframe dataframe with currencies exchange rates and prices 
        multiplication
  
        """

        result_dataframe = current_prices.copy()
        for cur_id in current_prices.columns:
            cur_name = currencies.at[cur_id]
            if cur_name == 'USD':
                result_dataframe.loc[:, cur_id] = current_prices.loc[:, cur_id]
            else:
                new_col = (current_prices.loc[:, cur_id] *
                          current_exch.loc[:, cur_name])
                result_dataframe.loc[:, cur_id] = new_col
        return result_dataframe

    def calculate_total_performance(
            self, 
            start_date: pd.Timestamp, 
            end_date: pd.Timestamp
        ) -> pd.Series:
        """ 
        Create series with total portfolio performance per day

        Parameters:
        start_date: start of the interval
        end_date: end of the interval
  
        Returns: 
        Series with total portfolio performance per day
  
        """
        
        self._init_current_dataframes(start_date, end_date)
        total_returns = self._calc_assets_returns(self._get_price_exch_mul(
                                                  self.current_prices,
                                                  self.current_exch,
                                                  self.currencies))
        weighted_returns = self._calc_weighted_returns(total_returns,
                                                       self.current_weights)
        return self._calc_performance(weighted_returns, start_date)
