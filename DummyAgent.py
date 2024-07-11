from TradingAgent import TradingAgent
import numpy as np

class DummyAgent(TradingAgent):
    def __init__(self, name):
        super().__init__(name)

    def generate_signals(self, data):
        return np.random.choice([0, 1, 2])
    
    def train_model(self, data):
        pass

    def trade(self, data):
        super().trade(data)
    
    def get_portfolio_value(self, current_price):
        return super().get_portfolio_value(current_price)