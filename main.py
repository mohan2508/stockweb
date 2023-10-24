import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px

st.markdown('## Stock :red[Future Forecast]')
stock_symbol=st.text_input('Enter Stock Ticker','SBIN.NS')

START = "2013-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

data = yf.download(stock_symbol,START,TODAY)
data.reset_index(inplace=True)

data_load_state = st.text('Loading data...')
data_load_state.text('Loading data... done!')
st.markdown('## :green[RAW DATA]')
st.write(data.tail())
st.markdown('## :red[Description]')
st.write(data.describe())

n_years = st.slider('Years of prediction:', 1, 3)
period = n_years * 365
st.markdown('## Trend Prediction :red[ with 50d,200d MA]')

ma100=data.Close.rolling(50).mean()
ma200=data.Close.rolling(200).mean()

def plot_ma_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data.Close.rolling(50).mean(), name="50d MA"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data.Close.rolling(200).mean(), name="200d MA"))
    #fig.add_trace(go.Scatter(x=data['Date'], y=data.Close.rolling(200).mean(), name="stock_close")) 
	fig.layout.update(title_text='50d vs 200d MA', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_ma_data()

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='View the data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.markdown('## :green[Forecast data]')
st.write(forecast.tail())
    
st.markdown(f'## Forecast plot for :red[{n_years} years]')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

hide_streamlit_style = """
<style>
#MainMenu{visibility:hidden}
footer{visibility:hidden}
Manage app{visibility:hidden}
deploy{visibility:hidden}
Header{visibility:hidden}
footer{visibility:hidden}

</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)     






