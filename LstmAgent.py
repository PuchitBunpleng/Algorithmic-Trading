from TradingAgent import TradingAgent
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras

class LstmAgent(TradingAgent):
    def __init__(self, name):
        super().__init__(name)
        self.window_size = 1
        self.retrain_period = 500
        self.model = self.build_model()
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer((self.window_size, 1)))
        model.add(keras.layers.LSTM(128, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.LSTM(64))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['root_mean_squared_error'])
        return model

    def prepare_data(self, data):
        
        X = []
        y = []
        window_size = self.window_size
        for i in range(len(data)-window_size):
            row = data[i:i+window_size]
            X.append(row)
            label = data[i+window_size]
            y.append(label)
        return np.array(X), np.array(y)


    def train_model(self, data):
        X, y = self.prepare_data(data['returns'])
        X = self.X_scaler.fit_transform(X)
        y = y.reshape(-1, 1)
        print(y)
        self.y_scaler = self.y_scaler.fit(y)
        y = self.y_scaler.transform(y)


        self.model.fit(X, y, epochs=20)


    def generate_signals(self, data):

        current = data['returns'].iloc[-1]
        features = self.extract_feature(data['returns'])
        features = self.X_scaler.fit_transform(features)
        pred = self.model.predict(np.array(current).reshape(-1, 1), verbose=0)
        pred = self.y_scaler.inverse_transform(pred)[0,0]
        print(pred)
        if pred > current:
            return 1
        elif pred < current:
            return 2
        else:
            return 0

    def extract_feature(self, data):
        features = []
        for i in range(len(data) - self.window_size):
            features.append(data[i:i + self.window_size])
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
    


