# Load Libraries
import pandas as pd                   # for data manipulation
import numpy as np                    # for numerical computation
import matplotlib.pyplot as plt       # for data visualization
from sqlalchemy import create_engine  # for interacting with databases
from urllib.parse import quote        # for URL encoding
from sklearn.linear_model import Lasso, Ridge, ElasticNet # for linear models of lasso, ridge, and elastic net
from sklearn.preprocessing import RobustScaler            # for robust scaling of features
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error # for model evaluation


# Database Credentials
user = 'root' # Username for MySQL
pw = quote('root')    # Password for MySQL
db = 'powertrading' # Database name

# Create Database Connection
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')

# Load Data
sql = 'SELECT * FROM preprocessed_power_tlb' # SQL query to fetch data
df = pd.read_sql(sql, con=engine) # Load data into a DataFrame

# Ensure Datetime is in correct format & sort it
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values(by='Datetime')

# Feature Engineering: Lag Features (Using last 96 lags)
for lag in range(1, 97):
    df[f'MCP_Lag_{lag}'] = df['MCP'].shift(lag)

# Drop NaN rows after lagging
df.dropna(inplace=True)

# Train-Test Split by Date
split_index = max(0, df.shape[0] - 5526)  # ensuring 2025 data as a test set so 5526 rows at the end
split_date = df['Datetime'].iloc[split_index] 

Train = df[df['Datetime'] < split_date] # train set 
Test = df[df['Datetime'] >= split_date] # test set

# Split Features & Target
X_train, y_train = Train.drop(columns=['MCP', 'Datetime']), Train['MCP']
X_test, y_test = Test.drop(columns=['MCP', 'Datetime']), Test['MCP']

# Apply RobustScaler to handle outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
lasso = Lasso(alpha=0.1).fit(X_train_scaled, y_train)                      # Train Lasso model
ridge = Ridge(alpha=0.1).fit(X_train_scaled, y_train)                       # Train Ridge model
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train_scaled, y_train)  # Train ElasticNet model

# Predictions
lasso_pred = lasso.predict(X_test_scaled) # Predictions for test_set using Lasso model
ridge_pred = ridge.predict(X_test_scaled) # Predictions for test_set using Ridge model
elastic_pred = elastic.predict(X_test_scaled) # Predictions for test_set using ElasticNet model

# Model Evaluation using RMSE and MAPE
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"{model_name} - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

# Evaluate Models
evaluate_model(y_test, lasso_pred, "Lasso") # Evaluate Lasso
evaluate_model(y_test, ridge_pred, "Ridge") # Evaluate Ridge
evaluate_model(y_test, elastic_pred, "ElasticNet") # Evaluate ElasticNet

# Forecast Next 100 MCP Values
def forecast_next_values(model, last_known_values, scaler, n_forecast=100):
    forecast_values = []
    current_input = last_known_values.copy()

    for _ in range(n_forecast):
        current_input_scaled = scaler.transform(current_input.reshape(1, -1))
        prediction = model.predict(current_input_scaled)[0]
        forecast_values.append(prediction)

        # Shift left & update last value
        if len(current_input) > 1:
            current_input = np.roll(current_input, -1)
            current_input[-1] = prediction
        else:
            break  # Avoid infinite loops

    return np.array(forecast_values)

# Get Last Known Features
if not X_test.empty:
    last_known_values = X_test.iloc[-1].values
else:
    raise ValueError("Test dataset is empty! Check the data split process.")

# Forecast
lasso_forecast = forecast_next_values(lasso, last_known_values, scaler) # Forecast using Lasso model 
ridge_forecast = forecast_next_values(ridge, last_known_values, scaler)    # Forecast using Ridge model
elastic_forecast = forecast_next_values(elastic, last_known_values, scaler) # Forecast using ElasticNet model

# Generate Future Time Intervals
last_time = df['Datetime'].iloc[-1]
next_times = pd.date_range(start=last_time, periods=100, freq='15T')

# Plot Forecast for Next 100 Time Intervals of MCP using Lasso, Ridge & ElasticNet
plt.figure(figsize=(12, 6)) # Set figure size
plt.plot(Test['Datetime'], y_test, label="Actual MCP", color='blue') # Plot Actual MCP
plt.plot(next_times, lasso_forecast, label="Lasso Prediction", color='red', linestyle='dashed') # Plot Lasso Forecast
plt.plot(next_times, ridge_forecast, label="Ridge Prediction", color='green', linestyle='dotted') # Plot Ridge Forecast
plt.plot(next_times, elastic_forecast, label="ElasticNet Prediction", color='purple', linestyle='dashdot') # Plot ElasticNet Forecast
plt.legend() # Add legend
plt.xlabel("Datetime") # Set x-axis label as Datetime
plt.ylabel("MCP") # Set y-axis label as MCP
plt.title("MCP Forecasting using Lasso, Ridge & ElasticNet") # Set title as MCP Forecasting using Lasso, Ridge & ElasticNet
plt.xticks(rotation=45) # Rotate x-axis labels for better visibility
plt.show() # Display the plot



# Plot Actual vs Predicted MCP Values (Test Data Only)
plt.figure(figsize=(14, 6))
plt.plot(Test['Datetime'], y_test, label="Actual MCP", color='blue', linewidth=2)
plt.plot(Test['Datetime'], lasso_pred, label="Lasso Prediction", color='red', linestyle='dashed')
plt.plot(Test['Datetime'], ridge_pred, label="Ridge Prediction", color='green', linestyle='dotted')
plt.plot(Test['Datetime'], elastic_pred, label="ElasticNet Prediction", color='purple', linestyle='dashdot')

plt.legend()
plt.xlabel("Datetime")
plt.ylabel("MCP")
plt.title("Actual vs Predicted MCP on Test Data")
plt.xticks(rotation=45)
plt.show()



# Function to calculate RMSE for Train & Test sets
def get_rmse(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    return train_rmse, test_rmse

# Get RMSE values
lasso_train_rmse, lasso_test_rmse = get_rmse(lasso, X_train_scaled, y_train, X_test_scaled, y_test)
ridge_train_rmse, ridge_test_rmse = get_rmse(ridge, X_train_scaled, y_train, X_test_scaled, y_test)
elastic_train_rmse, elastic_test_rmse = get_rmse(elastic, X_train_scaled, y_train, X_test_scaled, y_test)

# Plot Train vs Test RMSE for Each Model
models = ['Lasso', 'Ridge', 'ElasticNet']
train_rmse_values = [lasso_train_rmse, ridge_train_rmse, elastic_train_rmse]
test_rmse_values = [lasso_test_rmse, ridge_test_rmse, elastic_test_rmse]

plt.figure(figsize=(10, 6))
x = np.arange(len(models))
plt.bar(x - 0.2, train_rmse_values, width=0.4, label='Train RMSE', color='blue')
plt.bar(x + 0.2, test_rmse_values, width=0.4, label='Test RMSE', color='red')
plt.xticks(x, models)
plt.ylabel("RMSE")
plt.title("Train vs Test RMSE (Overfitting/Underfitting Check)")
plt.legend()
plt.show()






