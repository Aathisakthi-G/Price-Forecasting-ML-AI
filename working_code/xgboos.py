import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler
from sqlalchemy import create_engine  # For interacting with databases
from urllib.parse import quote
import os  # Importing operating system for file savings in particular paths
import pickle  # Loading and saving the files
from datetime import timedelta

# Database connection credentials
user = 'root'  # Username for database access
pw = quote('root')  # Password for database access
db = 'powertrading'  # Database name

# Create engine for database connection
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')

# Load the data into a DataFrame called ele_price_forecast
sql = 'SELECT * FROM powertrading.power_tlb;'
df = pd.read_sql(sql, con=engine)

df.drop(columns=["Session ID", 'MCV (MW)'], inplace=True)
df.rename(columns={"MCP (Rs/MWh) *": "MCP", 'Purchase Bid (MW)':'Purchase Bid', 'Sell Bid (MW)':'Sell Bid','Final Scheduled Volume (MW)':'Final Scheduled Volume'}, inplace=True)
df["Datetime"] = df["Datetime"].astype("datetime64[ns]")

# Replace zero values with NaN
df.replace(0, np.nan, inplace=True)

# Forward fill to handle missing values
df.fillna(method='ffill', inplace=True)

# Ensure no zero values are found in the columns
zero_counts = (df == 0).sum()
print("Count of zero values in each column after forward fill:")
print(zero_counts)

# Ensure 'Datetime' column is in the correct format
df['Datetime'] = pd.to_datetime(df['Datetime'])

df.sort_values('Datetime', inplace=True)

# Create 96 Lag Features (Last 24 Hours MCP Values)
for i in range(1, 97):
    df[f'MCP_lag_{i}'] = df['MCP'].shift(i)

# Drop rows with NaN (from lag feature creation)
df.dropna(inplace=True)

# Define Features and Target
features = [col for col in df.columns if col not in ['Datetime', 'MCP']]
target = 'MCP'

# Train-Test Split (Time-Based)
# Split data
train = df.head(70174-96)
test = df.tail(5526)

X_train = train.drop(columns=['MCP','Datetime']).values
y_train = train['MCP'].values
X_test = test.drop(columns=['MCP', 'Datetime']).values
y_test = test['MCP'].values
test_dates = test['Datetime'].values  # Extract Datetime for plotting

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define model with specified hyperparameters
final_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=350,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f'RMSE: {rmse:.4f}, MAPE: {mape:.2f}%')

# Plot Results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual MCP', color='blue')
plt.plot(y_pred, label='Predicted MCP', color='red', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('MCP')
plt.legend()
plt.title('MCP Forecasting with XGBoost')
plt.show()

# Underfitting and Overfitting Analysis
train_pred = final_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Training Data Plot (with labels every 10th point)
def plot_with_labels(dates, actual, predicted, title):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual MCP', color='blue')
    plt.plot(dates, predicted, label='Predicted MCP', color='red', linestyle='dashed')

    step = 10
    for i in range(0, len(dates), step):
        plt.text(dates[i], actual[i], str(round(actual[i], 2)), fontsize=9, color='blue', rotation=45)
        plt.text(dates[i], predicted[i], str(round(predicted[i], 2)), fontsize=9, color='red', rotation=45)

    plt.xlabel('Datetime')
    plt.ylabel('MCP')
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(title)
    plt.show()

# Training Data Plot
train_dates = train['Datetime'].values
plot_with_labels(train_dates[:100], y_train[:100], train_pred[:100], 'Training Data - Underfitting/Overfitting Check')

# Test Data Plot
plot_with_labels(test_dates[:100], y_test[:100], y_pred[:100], 'Test Data - Underfitting/Overfitting Check')

# RMSE Bar Plot
plt.figure(figsize=(8, 6))
plt.bar(['Train RMSE', 'Test RMSE'], [train_rmse, test_rmse], color=['blue', 'red'])
plt.ylabel('RMSE')
plt.title('Training vs Testing RMSE')
plt.show()

print(f'Training RMSE: {train_rmse:.4f}, Testing RMSE: {test_rmse:.4f}')

# 95% Confidence Interval for Predictions
error_std = np.std(y_test - y_pred)
confidence_interval = 1.96 * error_std

# Forecast Next 200 Values
future_steps = 200
future_dates = []
future_predictions = []
upper_bound = []
lower_bound = []

last_known_datetime = test['Datetime'].iloc[-1]
last_known_lags = X_test[-1].tolist()

time_interval = (test['Datetime'].iloc[-1] - test['Datetime'].iloc[-2])

for i in range(future_steps):
    next_pred = final_model.predict(np.array(last_known_lags).reshape(1, -1))[0]
    upper_ci = next_pred + confidence_interval
    lower_ci = next_pred - confidence_interval

    future_predictions.append(next_pred)
    upper_bound.append(upper_ci)
    lower_bound.append(lower_ci)

    next_datetime = last_known_datetime + (i + 1) * time_interval
    future_dates.append(next_datetime)

    last_known_lags = last_known_lags[1:] + [next_pred]

future_df = pd.DataFrame({'Datetime': future_dates, 'Predicted_MCP': future_predictions,
                          'Upper_CI': upper_bound, 'Lower_CI': lower_bound})

# Plot Future Forecast with 95% Confidence Interval (with labels every 5th point)
plt.figure(figsize=(12, 6))
plt.plot(future_df['Datetime'], future_df['Predicted_MCP'], label='Predicted MCP', color='green', linestyle='dashed')
plt.fill_between(future_df['Datetime'], future_df['Lower_CI'], future_df['Upper_CI'], color='gray', alpha=0.3, label='95% CI')

for i in range(0, len(future_df), 10):
    plt.text(future_df['Datetime'].iloc[i], future_df['Predicted_MCP'].iloc[i],
             f"{future_df['Predicted_MCP'].iloc[i]:.2f}",
             fontsize=9, rotation=45)

plt.xlabel('Datetime')
plt.ylabel('MCP')
plt.xticks(rotation=45)
plt.legend()
plt.title('Future MCP Forecast for Next 200 Time Steps with 95% Confidence Interval')
plt.show()

# Display Future Predictions
print(future_df)

# Save the model using pickle
with open('final_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)

# Save the scaler using pickle
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save the future predictions and confidence intervals
future_df.to_csv('future_predictions.csv', index=False)