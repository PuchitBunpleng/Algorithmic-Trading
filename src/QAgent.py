import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator, MACD
from ta.volatility import BollingerBands
from TradingAgent import TradingAgent


class QLearningAgent(TradingAgent):
    def __init__(self, name, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.3, decay_rate=0.999, retrain_period=200):
        super().__init__(name)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.decay_rate = decay_rate  # Decay rate for exploration
        self.q_table = {}  # Q-table
        self.retrain_period = retrain_period
        self.previous_state = None
        self.previous_action = None

    def get_state(self, data):
        rsi = RSIIndicator(data['close']).rsi().iloc[-1]
        cci = CCIIndicator(data['high'], data['low'], data['close']).cci().iloc[-1]
        macd = MACD(data['close'])
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        bollinger = BollingerBands(data['close'])
        upper_band = bollinger.bollinger_hband().iloc[-1]
        lower_band = bollinger.bollinger_lband().iloc[-1]
        close_price = data['close'].iloc[-1]

        rsi_state = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
        cci_state = 'overbought' if cci > 100 else 'oversold' if cci < -100 else 'neutral'
        macd_state = 'strong bullish' if macd_line > signal_line and macd_line > 0 else \
                     'weak bullish' if macd_line > signal_line and macd_line < 0 else \
                     'strong bearish' if macd_line < signal_line and macd_line < 0 else \
                     'weak bearish' if macd_line < signal_line and macd_line > 0 else 'neutral'
        bollinger_state = 'above upper band' if close_price > upper_band else \
                          'below lower band' if close_price < lower_band else 'within bands'

        state = (rsi_state, cci_state, macd_state, bollinger_state)
        return state

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(['buy', 'sell', 'hold'])
        else:
            q_values = [self.get_q_value(state, action) for action in ['buy', 'sell', 'hold']]
            max_q = max(q_values)
            action = np.random.choice([action for action, q in zip(['buy', 'sell', 'hold'], q_values) if q == max_q])
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
        
        return action

    def update_q_table(self, previous_state, previous_action, reward, current_state):
        prev_q = self.get_q_value(previous_state, previous_action)
        max_future_q = max([self.get_q_value(current_state, action) for action in ['buy', 'sell', 'hold']])
        new_q = prev_q + self.alpha * (reward + self.gamma * max_future_q - prev_q)
        self.q_table[(previous_state, previous_action)] = new_q

    def calculate_reward(self, data, action):
        price_diff = data['close'].iloc[-1] - data['close'].iloc[-2]
        if action == 'buy':
            reward = price_diff
        elif action == 'sell':
            reward = -price_diff
        else:
            reward = 0
        return reward

    def generate_signals(self, data):
        current_state = self.get_state(data)
        action = self.choose_action(current_state)
        return action

    def train_model(self, data):
        pass

    def trade(self, data):
        timestamp = data.index[-1]
        if len(data) < 500:
            print(f"{timestamp}: {self.name} - Collecting data...")
            return
        if len(data) % self.retrain_period == 0:
            print(f"{timestamp}: {self.name} - Retraining...")

        current_state = self.get_state(data)
        action = self.choose_action(current_state)

        if self.previous_state is not None and self.previous_action is not None:
            reward = self.calculate_reward(data, self.previous_action)
            self.update_q_table(self.previous_state, self.previous_action, reward, current_state)

        price = data['close'].iloc[-1]
        if action == 'buy' and self.position != 1 and self.cash > 0:
            self.holdings = self.cash / price
            self.cash = 0
            self.position = 1
            print(f"{timestamp}: {self.name} - Buy at {price}")
        elif action == 'sell' and self.position != -1 and self.holdings > 0:
            self.cash = self.holdings * price
            self.holdings = 0
            self.position = -1
            print(f"{timestamp}: {self.name} - Sell at {price}")
        else:
            print(f"{timestamp}: {self.name} - Hold")
        self.history.append(self.get_portfolio_value(price))

        self.previous_state = current_state
        self.previous_action = action
