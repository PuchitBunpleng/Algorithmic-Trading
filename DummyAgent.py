from TradingAgent import TradingAgent
import numpy as np

class DummyAgent(TradingAgent):
    def __init__(self, name):
        super().__init__(name)

    def generate_signals(self, data):
        # Change here
        return np.random.choice([0, 1, 2])
    
    def train_model(self, data):
        # Change here
        pass

    def trade(self, data):
        # Change here
        pass
    
    def get_portfolio_value(self, current_price):
        return super().get_portfolio_value(current_price)