import numpy as np
import pandas as pd
import yfinance as yf
import keras
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
model = keras.models.load_model('Stock Predictions Modified1 Model.keras')

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

stock = st.text_input('Enter Stock Symbol', 'VBL.NS')


data = yf.download(stock, start_date ,end_date)

st.subheader('Stock Data')
st.write(data)

# Choose the column Close to predict the Values
prices = data['Close'].values.reshape(-1, 1)

# Find the mean of the closing price bet 0 and 1
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)
# Provide the seq length and batch size
sequence_length = 90  # Number of previous time steps to use for prediction
#batch_size = 32

# Future predictions
num_future_days = 5
future_predictions = []
last_sequence = prices_scaled[-sequence_length:]

for _ in range(num_future_days):
    future_prediction_scaled = model.predict(last_sequence.reshape(1, sequence_length, 1))
    future_predictions.append(future_prediction_scaled[0, 0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = future_prediction_scaled

# Inverse transform
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
print(future_predictions)
st.write('Future Prices :- ')
st.markdown(future_predictions)
# Generate future dates
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=num_future_days)





# Plot future predictions
st.subheader('Future Stock Price Predictions')
fig4, ax4 = plt.subplots()
ax4.plot(data.index, prices, label='Actual Prices')
ax4.plot(future_dates, future_predictions, 'r', label='Future Predictions')
ax4.legend()
st.pyplot(fig4)



start = '2021-01-01'
end = '2024-12-31'
data1 = yf.download(stock, start,end)
st.subheader('Price vs MA50')
ma_50_days = data1.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data1.Close, 'g')
plt.legend(["MA50","Price"], loc="upper left")
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data1.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data1.Close, 'g')
plt.legend(["MA50","MA100","Price"], loc="upper left")
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data1.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data1.Close, 'g')
plt.legend(["MA50","MA200","Price"], loc="upper left")
plt.show()
st.pyplot(fig3)


