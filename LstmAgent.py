from TradingAgent import TradingAgent
import numpy as np
from catboost import Pool, CatBoostRegressor
import keras
import requests
import pandas as pd

class LstmAgent(TradingAgent):
    def __init__(self, name):
        """
        Initializes the ExampleAgent instance with a given name.
        
        Parameters:
            name (str): Agent name
        """        
        super().__init__(name)
        self.retrain_period = 500
        self.window_size = 100
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the LSTM model.

        Returns:
            Sequential: The LSTM model.
        """
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(self.window_size, 1)))
        model.add(keras.layers.LSTM(units=50))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    

    def generate_signals(self, data):
        """
        Generates trading signals based on the current and predicted prices.

        Parameters:
            data (DataFrame): The input data containing the prices or returns.

        Returns:
            int: The signal indicating whether to Hold, Buy, or Sell (0: Hold, 1: Buy, 2: Sell).
        """
        current = data['returns'].iloc[-1]
        features = self.extract_feature(data)
        pred = self.model.predict(np.array(features).reshape(-1, self.window_size))[0]
        if pred > current:
            return 1
        elif pred < current:
            return 2
        else:
            return 0
    
    def train_model(self, data):
        """
        Trains the model using the provided data.

        Parameters:
            data (DataFrame): The input data containing the prices or returns.
        """
        data = data['returns']
        train_data, train_label = self.prepare_data(data)
        self.model.fit(train_data, train_label, epochs=50, batch_size=32, verbose=0)
    
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
        features = []
        for i in range(len(data) - self.window_size):
            features.append(data[i:i+self.window_size])
        return features

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
    
    def get_portfolio_value(self, current_price):
        return super().get_portfolio_value(current_price)
    


def fetch_historical_data(symbol, interval, limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

df_1m = fetch_historical_data('BTCUSDT', '1m')

def prepare_data(data):
    data['returns'] = data['close'].pct_change()
    data.dropna(inplace=True)
    return data

df_1m = prepare_data(df_1m)

def backtest(agent, data):
    agent.train_model(data)
    for timestamp, row in data.iterrows():
        agent.trade(data.loc[:timestamp])
    return agent.get_portfolio_value(row['close'])

agent_1m = LstmAgent('LSTM Agent 1m')
portfolio_value_1m = backtest(agent_1m, df_1m)
print(f"Portfolio Value for 1m Interval: {portfolio_value_1m}")
