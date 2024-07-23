from TradingAgent import TradingAgent
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras

class LstmAgent(TradingAgent):
    def __init__(self, name):
        super().__init__(name)
        self.window_size = 5
        self.retrain_period = 500
        self.model = self.build_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def build_model(self):
        model = keras.models.Sequential()
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
        data = data['returns']
        data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        X, y = self.df_to_X_y(data, self.window_size)
        return X, y

    def df_to_X_y(self, df, window_size):
        X = []
        y = []
        for i in range(len(df) - window_size):
            row = df[i:i + window_size]
            X.append(row)
            label = df[i + window_size]
            y.append(label)
        return np.array(X), np.array(y)

    def train_model(self, data):
        X, y = self.prepare_data(data)
        split= int(len(X) * 0.9)

        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        cp = keras.callbacks.ModelCheckpoint('lstm_model.keras', save_best_only=True)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, callbacks=[cp])
        self.model = keras.models.load_model('lstm_model.keras')


    def generate_signals(self, data):
        features = self.extract_feature(data)
        features = self.scaler.transform(features)
        pred = self.model.predict(np.array(features).reshape(-1, self.window_size, 1))[0, 0]
        current = data['returns'].iloc[-1]
        if pred > current:
            return 1
        elif pred < current:
            return 2
        else:
            return 0

    def extract_feature(self, data):
        features = []
        data = self.scaler.transform(data['returns'].values.reshape(-1, 1))
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
    


