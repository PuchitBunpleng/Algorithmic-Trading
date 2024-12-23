from TradingAgent import TradingAgent
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler

class CatBoostAgent(TradingAgent):
    def __init__(self, name, column='returns', buy_threshold=0, sell_threshold=0, window_size=5, retrain_period=400):
        """
        Initializes the ExampleAgent instance with a given name.
        
        Parameters:
            name (str): Agent name
        """        
        super().__init__(name)
        self.model = CatBoostRegressor()
        self.buy_treshold = buy_threshold
        self.sell_treshold = sell_threshold
        self.retrain_period = retrain_period
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.column = column

    def generate_signals(self, data):
        """
        Generates trading signals based on the current and predicted prices.

        Parameters:
            data (DataFrame): The input data containing the prices or returns.

        Returns:
            int: The signal indicating whether to Hold, Buy, or Sell (0: Hold, 1: Buy, 2: Sell).
        """
        current = data[self.column].iloc[-1]
        scaled_data = self.scaler.transform(data[self.column].to_numpy().reshape(-1, 1)).squeeze()
        input = self.extract_feature(scaled_data)[-1]
        predict = self.model.predict(np.array(input.reshape(-1, self.window_size)))
        forecast = self.scaler.inverse_transform(predict.reshape(-1, 1))[0,0]
        if forecast - current > self.buy_treshold:
            return 1
        elif forecast - current < -self.sell_treshold:
            return 2
        else:
            return 0
    
    def train_model(self, data):
        """
        Trains the model using the provided data.

        Parameters:
            data (DataFrame): The input data containing the prices or returns.
        """
        scaled_data = pd.DataFrame({'data': self.scaler.fit_transform(data[self.column].to_numpy().reshape(-1, 1)).squeeze()})
        train_data, train_label = self.prepare_data(scaled_data['data'])
        train_pool = Pool(train_data, train_label)
        self.model.fit(train_pool, verbose=False)
    
    def prepare_data(self, data):
        """
        Prepare the training data

        Parameters:
            data (pandas.DataFrame): The input data containing the prices or returns.

        Returns:
            tuple: A tuple containing the training data and the corresponding labels.
                - train_data (pandas.DataFrame): The training data.
                - train_label (pandas.Series): The labels data.
        """
        features = self.extract_feature(data)
        train_label = data.shift(-self.window_size).dropna()
        train_data = np.array(features)
        return train_data, train_label
    
    def extract_feature(self, data):
        """
        Generates features based on the input data.

        Parameters:
            data (List): The input data for feature extraction.

        Returns:
            List: A list of features extracted from the input data.
        """
        features = []
        for i in range(len(data) - self.window_size):
            features.append(data[i:i+self.window_size])
        return np.array(features)

    def trade(self, data):
        """
        Trains the model using the provided data.

        Parameters:
            data (DataFrame): The input data containing the close prices.
        """
        timestamp = data.index[-1]
        if len(data) < self.retrain_period:
            print(f"{timestamp}: {self.name} - Collecting data...")
            return
        if len(data) % self.retrain_period == 0:
            data = data[-self.retrain_period:]  
            self.train_model(data)
            print(f"{timestamp}: {self.name} - Updated model")
        signal = self.generate_signals(data)
        price = data['close'].iloc[-1]
        if signal == 1 and self.position != 1 and self.cash > 0:
            self.holdings = self.cash / price
            self.cash = 0
            self.position = 1
            print(f"{timestamp}: {self.name} - Buy at {price}")
        elif signal == 2 and self.position != -1 and self.holdings > 0:
            self.cash = self.holdings * price
            self.holdings = 0
            self.position = -1
            print(f"{timestamp}: {self.name} - Sell at {price}")
        else:
            print(f"{timestamp}: {self.name} - Hold")
        self.history.append(self.get_portfolio_value(price))
    
    def get_portfolio_value(self, current_price):
        return super().get_portfolio_value(current_price)