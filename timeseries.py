import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load Time Series Dataset
df = pd.read_csv("timeseries.csv", parse_dates=['Date'], index_col='Date')  # Replace path

ts = df['Value']  # Replace 'Value' with your TS column name

# Train-Test Split
train = ts[:-12]
test = ts[-12:]

# ------------------ ARIMA ------------------
arima_model = ARIMA(train, order=(2,1,2))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=12)

print("ARIMA RMSE:", np.sqrt(mean_squared_error(test, arima_forecast)))

# ------------------ SARIMA ------------------
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.forecast(steps=12)

print("SARIMA RMSE:", np.sqrt(mean_squared_error(test, sarima_forecast)))
