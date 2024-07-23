import pandas as pd
import keras
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pca import data_preparation


"""
def fetch_historical_data(symbol, interval, limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

data = fetch_historical_data('BTCUSDT', '1d')
print(data.head())

def prepare_data(data):
    data['returns'] = data['close'].pct_change()
    data.dropna(inplace=True)
    return data

data = prepare_data(data)
# Normalize the close price data
scaler = MinMaxScaler(feature_range=(0, 1))
data['returns'] = scaler.fit_transform(data['returns'].values.reshape(-1, 1))
"""

data = data_preparation()

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = df_as_np[i:i+window_size]
        X.append(row)
        label = df_as_np['returns'][i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

# Prepare the data for LSTM
window_size = 5
X, y = df_to_X_y(data, window_size)
print(X.shape, y.shape)

# Split the data into training, validation, and test sets
split_1 = int(len(X) * 0.8)
split_2 = int(len(X) * 0.9)

X_train, y_train = X[:split_1], y[:split_1]
X_val, y_val = X[split_1:split_2], y[split_1:split_2]
X_test, y_test = X[split_2:], y[split_2:]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

# Define a more sophisticated LSTM model
model2 = keras.models.Sequential()
model2.add(keras.layers.InputLayer((window_size, 1)))
model2.add(keras.layers.LSTM(128, return_sequences=True))
model2.add(keras.layers.Dropout(0.2))
model2.add(keras.layers.LSTM(64))
model2.add(keras.layers.Dropout(0.2))
model2.add(keras.layers.Dense(32, activation='relu'))
model2.add(keras.layers.Dense(1, activation='linear'))

model2.summary()

cp2 = keras.callbacks.ModelCheckpoint('model2/.keras', save_best_only=True)
model2.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=[keras.metrics.RootMeanSquaredError()])

# Train the model
history = model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, callbacks=[cp2])

# Load the best model
model3 = keras.models.load_model('model2/.keras')

# Predictions and plotting
def plot_predictions(data, title):
    plt.plot(data['Predictions'], label='Predictions')
    plt.plot(data['Actuals'], label='Actuals')
    plt.title(title)
    plt.legend()
    plt.show()

train_predictions = model3.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Predictions': train_predictions, 'Actuals': y_train.flatten()})
plot_predictions(train_results, 'Train Predictions vs Actuals')

val_predictions = model3.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Predictions': val_predictions, 'Actuals': y_val.flatten()})
plot_predictions(val_results, 'Validation Predictions vs Actuals')

test_predictions = model3.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Predictions': test_predictions, 'Actuals': y_test.flatten()})
plot_predictions(test_results, 'Test Predictions vs Actuals')

