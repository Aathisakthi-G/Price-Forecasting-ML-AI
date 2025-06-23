# Data based approaches

# Import libraries commonly used for data analysis
import numpy as np  # Used for numerical computations (potentially used later)
import pandas as pd  # Used for data manipulation (reading and working with DataFrame)
import matplotlib.pyplot as plt  # Used for data visualization (potentially used for plotting later)

from statsmodels.tsa.stattools import adfuller # used for stationary test
# Import classes for exponential smoothing models (forecasting)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing  # Implements Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import Holt  # Implements Holt's Exponential Smoothing (with trend)
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Implements general Holt-Winters Exponential Smoothing (with trend and seasonality)

from sklearn.metrics import mean_squared_error # for mean square error

from sqlalchemy import create_engine  # For interacting with databases
from urllib.parse import quote
import os # importing operating sysytem for file savings in perticular paths
import pickle # loading and saving the files

# Database connection credentials 
user = 'root'  # Username for database access
pw = quote('root')  # Password for database access
db = 'powertrading'  # Database name

# Create engine for database connection
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')

sql = 'SELECT * FROM power_tlb_imputed'# Load the data into a DataFrame called ele_price_forecast

df = pd.read_sql(sql, con=engine) # Load the data into a DataFrame of date_time_MCP
df.info()

# Function to check stationarity using ADF test
def adf_test(series):
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    return result[1]  # Return p-value

# Test for stationarity
p_value = adf_test(df['MCP'])

# If p-value > 0.05, apply differencing
if p_value > 0.05:
    df['MCP_diff'] = df['MCP'].diff().dropna()
    print("Data differenced to achieve stationarity.")
else:
    df['MCP_diff'] = df['MCP']
    print('else executed')

# Split data
Train = df.head(70174)
Test = df.tail(5526)

# Define MAPE function
def MAPE(pred, actual):
    return np.mean(np.abs((pred - actual) / actual)) * 100

# Define RMSE function
def RMSE(pred, actual):
    return np.sqrt(mean_squared_error(actual, pred))

# ------------------ Moving Average ------------------
window_size = 3 # Change this based on requirements
df['MCP_MA'] = df['MCP'].rolling(window=window_size).mean()

# Get moving average predictions for test data
ma_pred_test = df['MCP_MA'].tail(5526)

# Calculate error metrics
mape_ma = MAPE(ma_pred_test, Test['MCP'])
rmse_ma = RMSE(ma_pred_test, Test['MCP'])
print(f"RMSE: {rmse_ma:.2f}, MAPE: {mape_ma:.2f}%")

# Plot Moving Average
plt.figure(figsize=(10, 5))
df['MCP'].plot(label='Actual MCP')
df['MCP_MA'].plot(label=f'Moving Average (window={window_size})', linestyle='dashed')
plt.legend()
plt.title('MCP with Moving Average')
plt.show()

# ------------------ Exponential Smoothing Methods ------------------

# Simple Exponential Smoothing
ses_model = SimpleExpSmoothing(Train['MCP_diff'].dropna()).fit(optimized=True, use_brute=True)
pred_ses = ses_model.predict(start=Test.index[0], end=Test.index[-1])
mape_ses = MAPE(pred_ses, Test['MCP'].dropna())
rmse_ses = RMSE(pred_ses, Test['MCP'].dropna())

print(f"RMSE: {rmse_ses:.2f}, MAPE: {mape_ses:.2f}%")

# Holt’s Method
holt_model = Holt(Train['MCP_diff'].dropna(), initialization_method="estimated").fit()
pred_holt = holt_model.predict(start=Test.index[0], end=Test.index[-1])
mape_holt = MAPE(pred_holt, Test['MCP'].dropna())
rmse_holt = RMSE(pred_holt, Test['MCP'].dropna())
print(f"RMSE: {rmse_holt:.2f}, MAPE: {mape_holt:.2f}%")

# Holt-Winters Exponential Smoothing (Additive)
hwe_model_add = ExponentialSmoothing(Train['MCP_diff'].dropna(), seasonal="add", trend="add", seasonal_periods=4).fit()
pred_hwe_add = hwe_model_add.predict(start=Test.index[0], end=Test.index[-1])
mape_hwe_add = MAPE(pred_hwe_add, Test['MCP'].dropna())
rmse_hwe_add = RMSE(pred_hwe_add, Test['MCP'].dropna())
print(f"RMSE: {rmse_hwe_add:.2f}, MAPE: {mape_hwe_add:.2f}%")

# Holt-Winters Exponential Smoothing (Multiplicative)
hwe_model_mul = ExponentialSmoothing(Train['MCP_diff'].dropna(), seasonal="mul", trend="add", seasonal_periods=4).fit()
pred_hwe_mul = hwe_model_mul.predict(start=Test.index[0], end=Test.index[-1])
mape_hwe_mul = MAPE(pred_hwe_mul, Test['MCP'].dropna())
rmse_hwe_mul = RMSE(pred_hwe_mul, Test['MCP'].dropna())

# ------------------ Store Results ------------------
results = pd.DataFrame({
   'Method': ['Moving Average','Simple Exp Smoothing','Holt','HW Additive','HW Multiplicative'],
     'MAPE': [mape_ma, mape_ses, mape_holt, mape_hwe_add, mape_hwe_mul],
     'RMSE': [rmse_ma, rmse_ses, rmse_holt, rmse_hwe_add, rmse_hwe_mul]})

print(results)


# Forecast the next 100 values
# Ensure 'Date' column is in datetime format
if 'Datetime' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Datetime'])  # Convert if 'Date' column exists
    df.set_index('Datetime', inplace=True)  # Set as index if not already


# Forecast the next 100 values
forecast_steps = 10
future_forecast = []
last_values = list(df['MCP'].dropna().values[-window_size:])  # Get last 'window_size' values

for _ in range(forecast_steps):  # Forecast next 100 points
    next_value = np.mean(last_values)  # Compute mean of last 'window_size' values
    future_forecast.append(next_value)
    last_values.pop(0)  # Remove oldest value
    last_values.append(next_value)  # Append new predicted value

# Generate future timestamps at 15-minute intervals
last_datetime = df.index[-1]  # Get last timestamp from dataset
future_dates = pd.date_range(start=last_datetime + pd.Timedelta(minutes=15), periods=forecast_steps, freq='15T')

# Create DataFrame for future forecast
future_df = pd.DataFrame({'Date': future_dates, 'Forecasted_MCP': future_forecast})

# Plot Actual vs Forecasted
plt.figure(figsize=(12, 6))
plt.plot(df.index[-200:], df['MCP'].tail(200), label="Actual MCP", marker='o')
plt.plot(future_df['Date'], future_df['Forecasted_MCP'], label="Forecasted MCP (Moving Avg)", marker='x', linestyle='dashed')
plt.legend()
plt.title("Moving Average Forecast for Next 100 Values (15-Min Intervals)")
plt.xticks(rotation=45)
plt.show()

# Print forecasted values
print(future_df)

'''
The issue with moving average model is getting the same forecasted value 
for all 100 points is that the Moving Average method is a smoothing technique, 
not a true predictive model
'''