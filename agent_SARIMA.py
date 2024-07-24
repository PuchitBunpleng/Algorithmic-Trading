import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from DummyAgent import TradingAgent  # Assuming you have a base TradingAgent class

class SARIMAXAgent(TradingAgent):
    def __init__(self, name, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), buy_threshold=0, sell_threshold=0):
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.model = None
        self.retrain_period = 500

    def train_model(self, data):
        print(self.order)
        self.model = SARIMAX(data['close'], order=self.order, seasonal_order=self.seasonal_order)
        self.results = self.model.fit(disp=False)
    
    def generate_signals(self, data):
        prediction = self.results.get_forecast(steps=1).predicted_mean.iloc[-1]
        current = data['close'].iloc[-1]
        if prediction - current> self.buy_threshold:
            return 1  # Buy signal
        elif prediction - current < -self.sell_threshold:
            return 2  # Sell signal
        return 0  # Hold

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
    
    def get_portfolio_value(self, current_price):
        return super().get_portfolio_value(current_price)
