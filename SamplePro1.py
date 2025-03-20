import numpy as np
import pandas as pd
import yfinance as yf
import keras
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
model = keras.models.load_model('Stock Predictions Model.keras')

# App title
st.markdown('''
# Stock Price Prediction App

Shown are the stock price data and predictions for selected companies!

**Credits**
- App built by [SSD](http://youtube.com/dataprofessor)
- Built using `Python`, `Streamlit`, `yfinance`, `cufflinks`, and `keras`
''')
st.write('---')

# Sidebar parameters
st.sidebar.subheader('Query Parameters')
start_date = st.sidebar.date_input("Start Date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())




st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symnbol', 'VBL.NS')
'''start = '2015-01-01'
end = '2024-12-31'''

data = yf.download(stock, start_date ,end_date)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.legend(["MA50","Price"], loc="upper left")
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.legend(["MA50","MA100","Price"], loc="upper left")
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.legend(["MA100","MA200","Price"], loc="upper left")
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.legend(["Original Price","Predicted Price"], loc="upper left")
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
