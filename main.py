import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

crypto_currency = "ADA"
against_currency = "USD"

start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

data = web.DataReader(f"{crypto_currency}-{against_currency}", "yahoo", start, end)

print(data.head())

# Data

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

prediction_days = 60

x_train, y_train = [], []

for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i - prediction_days: i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# THE AI :( im still bad at ML
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss="mean_squared_error")
model.fit(x_train,y_train, epochs=25, batch_size=32)

# Tests

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(f"{crypto_currency}-{against_currency}", "yahoo", test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days :].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for i in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[i - prediction_days :i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color="black", label="Actual Prices")
plt.plot(prediction_prices, color="green", label="Predicted Prices")
plt.title(f"{crypto_currency} price prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()