import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('stock_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select relevant columns (adjust according to your dataset)
prices = data['Close'].values.reshape(-1, 1)
profit = data['Profit'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create training data for LSTM
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_sequences(scaled_prices)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Predicting stock prices using LSTM
predicted_prices = model.predict(X_train)
predicted_prices = scaler.inverse_transform(predicted_prices)

# ARIMA model
train_arima, test_arima = prices[:-100], prices[-100:]
arima_model = ARIMA(train_arima, order=(5, 1, 0))
arima_model_fit = arima_model.fit()

arima_pred = arima_model_fit.forecast(steps=100)
arima_mse = mean_squared_error(test_arima, arima_pred)

# LSTM Prediction Error
lstm_mse = mean_squared_error(prices[time_step:], predicted_prices)

# Compare the two models
print(f'LSTM MSE: {lstm_mse}')
print(f'ARIMA MSE: {arima_mse}')

# Visualize the predictions
plt.figure(figsize=(14, 5))
plt.plot(data.index[time_step:], scaler.inverse_transform(scaled_prices[time_step:]), color='blue', label='Actual Prices')
plt.plot(data.index[time_step:], predicted_prices, color='red', label='LSTM Predicted Prices')
plt.plot(data.index[-100:], arima_pred, color='green', label='ARIMA Predicted Prices')
plt.title('Stock Price Prediction (LSTM vs ARIMA)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate Fair Value based on EPS and Median P/E
forecasted_profit = np.mean(profit[-100:])  # Assuming future profit based on historical average
shares_outstanding = 1000000  # Example number of shares
forecasted_eps = forecasted_profit / shares_outstanding

# Median P/E calculation (using historical data)
median_pe = np.median(data['PE_Ratio'].values)
fair_value = median_pe * forecasted_eps
print(f'Forecasted EPS: {forecasted_eps}')
print(f'Fair Value: {fair_value}')
