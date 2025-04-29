import numpy as np
import pandas as pd
import yfinance as yf
import keras
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = keras.models.load_model('Stock Predictions Mod Model.keras')

# App title
st.markdown('''
# Stock Price Prediction App

Shown are the stock price data and predictions for selected companies!

**Credits**
- App built by [SSD](http://youtube.com/dataprofessor)
- Built using `Python`, `Streamlit`, `yfinance` and `keras`
''')
st.write('---')

# Sidebar parameters
st.sidebar.subheader('Select Dates')
start_date = st.sidebar.date_input("Start Date", datetime.date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())




st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'IOC.NS')

# Fetch stock data
df = yf.download(stock, start=start_date, end=end_date)

# Handle missing values
df.dropna(inplace=True)

st.subheader('Stock Data')
st.write(df)

# Data Preparation
sequence_length = 1000
data_training = pd.DataFrame(df["Close"][:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70):])

scaler = MinMaxScaler()
data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

x_train, y_train = [], []
for i in range(data_training_array.shape[0] - sequence_length):
    x_train.append(data_training_array[i: sequence_length + i, 0])
    y_train.append(data_training_array[sequence_length + i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Prepare test data
past_days = data_training.tail(sequence_length)
new_df = pd.concat([past_days, data_testing], ignore_index=True)

# Handle missing values in new_df
new_df.dropna(inplace=True)

input_data = scaler.fit_transform(new_df)

x_test, y_test = [], []
for i in range(sequence_length, input_data.shape[0]):
    x_test.append(input_data[i - sequence_length: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_predicted = model.predict(x_test)

# Ensure scale_factor is not zero
scale_factor = 1 / (scaler.scale_[0] if scaler.scale_[0] != 0 else 1)
y_predicted *= scale_factor
y_test *= scale_factor

# Future predictions
duration = 10
x_future = [input_data[-sequence_length:]]
future_predictions = []

for i in range(duration):
    x_future_array = np.array(x_future).reshape(1, sequence_length, 1)
    future_prediction = model.predict(x_future_array)
    future_predictions.append(future_prediction[0, 0])
    x_future[0] = np.concatenate((x_future[0][1:], future_prediction), axis=None)

future_predictions_array = np.array(future_predictions)
y_future = scaler.inverse_transform(future_predictions_array.reshape(-1, 1))

future_time_indices = pd.date_range(start=df.index[-1], periods=duration, freq="D")
st.markdown(y_future)
# Plot predictions
import plotly.express as px
import pandas as pd

# Ensure y_future is flat
if y_future.ndim > 1:
    y_future = y_future.flatten()

# Regenerate future_time_indices if needed
if len(future_time_indices) != len(y_future):
    future_time_indices = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=len(y_future), freq='D')

# Convert historical close to flat array if needed
historical_prices = df["Close"].values.flatten()

# Prepare historical DataFrame
df_hist = pd.DataFrame({
    "Date": df.index,
    "Price": historical_prices,
    "Type": "Historical"
})

# Prepare future prediction DataFrame
df_future = pd.DataFrame({
    "Date": future_time_indices,
    "Price": y_future,
    "Type": "Predicted"
})

# Combine both into one plot-friendly DataFrame
plot_df = pd.concat([df_hist, df_future], ignore_index=True)

# Create line chart
fig = px.line(
    plot_df,
    x="Date",
    y="Price",
    color="Type",
    title="📈 Stock Price: Historical vs Predicted",
    labels={"Price": "Stock Price", "Date": "Date"},
    template="plotly_white"
)

# Show in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Moving Averages Analysis
start = '2021-01-01'
end = '2024-12-31'
data1 = yf.download(stock, start=start, end=end)

# Handle missing values
data1.dropna(inplace=True)

st.subheader('Price vs MA50')
ma_50_days = data1.Close.rolling(50).mean()
fig1, ax1 = plt.subplots()
ax1.plot(ma_50_days, 'r')
ax1.plot(data1.Close, 'g')
ax1.legend(["MA50", "Price"], loc="upper left")
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data1.Close.rolling(100).mean()
fig2, ax2 = plt.subplots()
ax2.plot(ma_50_days, 'r')
ax2.plot(ma_100_days, 'b')
ax2.plot(data1.Close, 'g')
ax2.legend(["MA50", "MA100", "Price"], loc="upper left")
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data1.Close.rolling(200).mean()
fig3, ax3 = plt.subplots()
ax3.plot(ma_50_days, 'r')
ax3.plot(ma_200_days, 'b')
ax3.plot(data1.Close, 'g')
ax3.legend(["MA50", "MA200", "Price"], loc="upper left")
st.pyplot(fig3)
