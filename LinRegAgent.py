from TradingAgent import TradingAgent
import numpy as np
from sklearn.linear_model import LinearRegression

class LinRegAgent(TradingAgent):
    def __init__(self, name, buy_threshold=0.01, sell_threshold=0.01):
        super().__init__(name)
        self.model = LinearRegression()
        self.buy_treshold = buy_threshold
        self.sell_treshold = sell_threshold
        self.retrain_period = 200

    def generate_signals(self, data):
        # Change here
        data = data['returns']
        current = data.iloc[-1]
        pred = self.model.predict(np.array(current).reshape(-1, 1))[0]
        if pred - current > self.buy_treshold:
            return 1
        elif pred - current < -self.sell_treshold:
            return 2
        else:
            return 0
    
    def train_model(self, data):
        # Change here
        train_data = data['returns']
        train_data = train_data[:-1]
        train_label = data['returns'].shift(-1)
        train_label = train_label.dropna()
        self.model.fit(train_data.values.reshape(-1, 1), train_label.values.reshape(-1, 1))

    def trade(self, data):
        # Change here
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