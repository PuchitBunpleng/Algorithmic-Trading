from TradingAgent import TradingAgent
import numpy as np

class DummyAgent(TradingAgent):
    def generate_signals(self, data):
        return np.random.choice([0, 1, 2])
    
    def train_model(self, data):
        pass