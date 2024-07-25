from TradingAgent import TradingAgent
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class LinRegAgent(TradingAgent):
    def __init__(self, name, column='returns', buy_threshold=0, sell_threshold=0, window_size=5, retrain_period=200):
        super().__init__(name)
        self.model = LinearRegression()
        self.buy_treshold = buy_threshold
        self.sell_treshold = sell_threshold
        self.retrain_period = retrain_period
        self.scaler = MinMaxScaler()
        self.column = column
        self.window_size = window_size

    def generate_signals(self, data):
        current = data[self.column].iloc[-1]
        scaled_data = self.scaler.transform(data[self.column].to_numpy().reshape(-1, 1)).squeeze()
        input = self.extract_feature(scaled_data)[-1]
        predict = self.model.predict(input.reshape(-1, self.window_size))
        forecast = self.scaler.inverse_transform(predict.reshape(-1, 1))[0,0]
        if forecast - current > self.buy_treshold:
            return 1
        elif forecast - current < -self.sell_treshold:
            return 2
        else:
            return 0
    
    def train_model(self, data):
        scaled_data = pd.DataFrame({'data': self.scaler.fit_transform(data[self.column].to_numpy().reshape(-1, 1)).squeeze()})
        train_data, train_label = self.prepare_data(scaled_data['data'])
        self.model.fit(np.array(train_data), np.array(train_label))
    
    def prepare_data(self, data):
        features = self.extract_feature(data)
        train_label = data.shift(-self.window_size).dropna()
        train_data = np.array(features)
        return train_data, train_label
    
    def extract_feature(self, data):
        features = []
        for i in range(len(data) - self.window_size):
            features.append(data[i:i+self.window_size])
        return np.array(features)

    def trade(self, data):
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