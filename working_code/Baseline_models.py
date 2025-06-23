# Model based approaches

# Load the necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting graphs
from sqlalchemy import create_engine  # For interacting with databases
from urllib.parse import quote # For encoding passwords that might have special characters
import os  # For file operations  
import statsmodels.formula.api as smf # For statistical modeling
import pickle # For saving the model
from statsmodels.regression.linear_model import OLSResults # For loading the model

# Database connection credentials 
user = 'root'  # Username for database access
pw = quote('root')  # Password for database access
db = 'powertrading'  # Database name

# Create engine for database connection
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')
sql = 'SELECT * FROM  power_tlb_preprocessed'# Load the data into a DataFrame called ele_price_forecast
df1= pd.read_sql(sql, con=engine) # Load the data into a DataFrame of date_time_MCP
df1.info() # Display the column names and data types

df1['t00'] = df1['t00'].astype('bool')
df1['t15'] = df1['t15'].astype('bool')
df1['t30'] = df1['t30'].astype('bool')
df1['t45'] = df1['t45'].astype('bool')
df1.head() 

# Split data into training and testing sets
Train = df1.head(70174)
Test = df1.tail(5526)

# Define a function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Linear model
linear_model = smf.ols('MCP ~ t', data=Train).fit() # Fit the linear model
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t']))) # Make predictions on the test set

# Calculate the RMSE and MAPE for the linear model
rmse_linear = np.sqrt(np.mean((np.array(Test['MCP']) - np.array(pred_linear)) ** 2))
print(f"Linear Model - RMSE: {rmse_linear}") # Print the RMSE for the linear model

# Calculate the MAPE for the linear model
mape_linear = mean_absolute_percentage_error(Test['MCP'], pred_linear)
print(f"Linear Model - RMSE: {rmse_linear}, MAPE: {mape_linear}")

# Exponential model
exp_model = smf.ols('log_MCP ~ t', data=Train).fit() # Build the exponential model
pred_exp = pd.Series(exp_model.predict(pd.DataFrame(Test['t']))) # Make predictions on the test set
rmse_exp = np.sqrt(np.mean((np.array(Test['MCP']) - np.array(np.exp(pred_exp))) ** 2)) # Calculate the RMSE for the exponential model
mape_exp = mean_absolute_percentage_error(Test['MCP'], np.exp(pred_exp)) # Calculate the MAPE for the exponential model

print(f"Exponential Model - RMSE: {rmse_exp}, MAPE: {mape_exp}") # Print the RMSE and MAPE for the exponential model

# Quadratic model
quad_model = smf.ols('MCP ~ t + t_sq', data=Train).fit() # Fit the quadratic model
pred_quad = pd.Series(quad_model.predict(Test[["t", "t_sq"]])) # Make predictions on the test set
rmse_quad = np.sqrt(np.mean((np.array(Test['MCP']) - np.array(pred_quad)) ** 2)) # Calculate the RMSE for the quadratic model
mape_quad = mean_absolute_percentage_error(Test['MCP'], pred_quad) # Calculate the MAPE for the quadratic model

print(f"Quadratic Model - RMSE: {rmse_quad}, MAPE: {mape_quad}") # Print the RMSE and MAPE for the quadratic 

# Additive seasonality model
add_sea_model = smf.ols("MCP ~ t00 + t15 + t30", data=Train).fit() # Fit the additive seasonality model
pred_add_sea = pd.Series(add_sea_model.predict(Test[['t00', 't15', 't30']])) # Make predictions on the test set
rmse_add_sea = np.sqrt(np.mean((np.array(Test['MCP']) - np.array(pred_add_sea)) ** 2)) # Calculate the RMSE for the additive seasonality model
mape_add_sea = mean_absolute_percentage_error(Test['MCP'], pred_add_sea) # Calculate the MAPE for the additive seasonality model

print(f"Additive Seasonality Model - RMSE: {rmse_add_sea}, MAPE: {mape_add_sea}") # Print the RMSE and MAPE for the additive seasonality model

# multiplicative seasonality model
mul_sea_model = smf.ols("log_MCP ~ t00 + t15 + t30", data=Train).fit() # Fit the multiplicative seasonality model
pred_mul_sea = pd.Series(mul_sea_model.predict(Test[['t00', 't15', 't30', 't45']])) # Make predictions on the test set
rmse_mul_sea = np.sqrt(np.mean((np.array(Test['MCP']) - np.array(np.exp(pred_mul_sea))) ** 2)) # Calculate the RMSE for the multiplicative seasonality model
mape_mul_sea = mean_absolute_percentage_error(Test['MCP'], np.exp(pred_mul_sea)) # Calculate the MAPE for the multiplicative seasonality model

print(f"Multiplicative Seasonality Model - RMSE: {rmse_mul_sea}, MAPE: {mape_mul_sea}") # Print the RMSE and MAPE for the multiplicative seasonality model

# Additive seasonality with quadratic trend model
add_sea_quad_model = smf.ols('MCP ~ t + t_sq + t00 + t15 + t30', data=Train).fit() # Fit the additive seasonality with quadratic trend model
pred_add_sea_quad = pd.Series(add_sea_quad_model.predict(Test[['t', 't_sq', 't00', 't15', 't30']])) # Make predictions on the test set
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['MCP']) - np.array(pred_add_sea_quad)) ** 2)) # Calculate the RMSE for the additive seasonality with quadratic trend model
mape_add_sea_quad = mean_absolute_percentage_error(Test['MCP'], pred_add_sea_quad) # Calculate the MAPE for the additive seasonality with quadratic trend model

print(f"Additive Seasonality with Quadratic Trend Model - RMSE: {rmse_add_sea_quad}, MAPE: {mape_add_sea_quad}") # Print the RMSE and MAPE for the additive seasonality with quadratic trend model

# Multiplicative seasonality with exponential trend 
mul_sea_exp_model = smf.ols('log_MCP ~ t + t_sq + t00 + t15 + t30', data=Train).fit() # Fit the multiplicative seasonality with exponential trend model
pred_mul_sea_exp = pd.Series(mul_sea_exp_model.predict(Test[['t', 't_sq', 't00', 't15', 't30']])) # Make predictions on the test set
rmse_mul_sea_exp = np.sqrt(np.mean((np.array(Test['MCP']) - np.array(np.exp(pred_mul_sea_exp))) ** 2)) # Calculate the RMSE for the multiplicative seasonality with exponential trend model
mape_mul_sea_exp = mean_absolute_percentage_error(Test['MCP'], np.exp(pred_mul_sea_exp)) # Calculate the MAPE for the multiplicative seasonality with exponential trend 

print(f"Multiplicative Seasonality with Exponential Trend Model - RMSE: {rmse_mul_sea_exp}, MAPE: {mape_mul_sea_exp}") # Print the RMSE and MAPE for the multiplicative seasonality with exponential trend model

# Additive seasonality with exponential trend model
add_sea_exp_model = smf.ols('log_MCP ~ t + t00 + t15 + t30', data=Train).fit() # Fit the additive seasonality with exponential trend model
pred_add_sea_exp = pd.Series(add_sea_exp_model.predict(Test[['t', 't00', 't15', 't30']])) # Make predictions on the test set
rmse_add_sea_exp = np.sqrt(np.mean((np.array(Test['MCP']) - np.array(np.exp(pred_add_sea_exp))) ** 2)) # Calculate the RMSE for the additive seasonality with exponential trend model
mape_add_sea_exp = mean_absolute_percentage_error(Test['MCP'], np.exp(pred_add_sea_exp)) # Calculate the MAPE for the additive seasonality with exponential trend model

print(f"Additive Seasonality with Exponential Trend Model - RMSE: {rmse_add_sea_exp}, MAPE: {mape_add_sea_exp}") # Print the RMSE and MAPE for the additive seasonality with exponential trend model

# Multiplicative seasonality with quadratic trend model
mul_sea_quad_model = smf.ols('MCP ~ t + t_sq + t00 + t15 + t30', data=Train).fit() # Fit the multiplicative seasonality with quadratic trend model
pred_mul_sea_quad = pd.Series(mul_sea_quad_model.predict(Test[['t', 't_sq', 't00', 't15', 't30']])) # Make predictions on the test set
rmse_mul_sea_quad = np.sqrt(np.mean((np.array(Test['MCP']) - np.array(pred_mul_sea_quad)) ** 2)) # Calculate the RMSE for the multiplicative seasonality with quadratic trend model
mape_mul_sea_quad = mean_absolute_percentage_error(Test['MCP'], pred_mul_sea_quad) # Calculate the MAPE for the multiplicative seasonality with quadratic trend model

print(f"Multiplicative Seasonality with Quadratic Trend Model - RMSE: {rmse_mul_sea_quad}, MAPE: {mape_mul_sea_quad}") # Print the RMSE and MAPE for the multiplicative seasonality with quadratic trend model
print('\n')

# Mape Table for all models
# Creating a DataFrame to store the RMSE and MAPE values for all the models
data = {'Model': ['Linear', 'Exponential', 'Quadratic', 'Additive Seasonality', 'Multiplicative Seasonality', 'Additive Seasonality with Quadratic Trend', 'Multiplicative Seasonality with Exponential Trend', 'Additive Seasonality with Exponential Trend', 'Multiplicative Seasonality with Quadratic Trend'], \
        'RMSE': [rmse_linear, rmse_exp, rmse_quad, rmse_add_sea, rmse_mul_sea, rmse_add_sea_quad, rmse_mul_sea_exp, rmse_add_sea_exp, rmse_mul_sea_quad], \
         'MAPE': [mape_linear, mape_exp, mape_quad, mape_add_sea, mape_mul_sea, mape_add_sea_quad, mape_mul_sea_exp, mape_add_sea_exp, mape_mul_sea_quad]}
mape_table = pd.DataFrame(data) # Create a DataFrame from the data
print(mape_table) # Display the DataFrame


# The best model is the one with the lowest RMSE and MAPE values.  
# In this case, the best model is the Multiplicative Seasonality with Exponential Trend Model.
print('\n')
print("The best model is the additive Seasonality with Exponential Trend Model.")
print(f'RMSE: {rmse_mul_sea_exp}, MAPE: {mape_mul_sea_exp}')

# plotting the actual and predicted values
plt.figure(figsize=(12, 6)) # Set the figure size
plt.plot(Test['MCP'].reset_index(drop=True), label='Actual', color='red') # Plot the actual values
plt.plot(np.exp(pred_add_sea_exp).reset_index(drop=True), label='Predicted', color='blue') # Plot the predicted values
plt.legend() # Display the legend
plt.xlabel('Time') # Label for x-axis
plt.ylabel('MCP') # Label for y-axis
plt.title('Actual vs Predicted MCP') # Title of the plot
plt.grid(True) # Show grid
plt.show()


# Plotting the actual and predicted values with zoomed-in view
plt.figure(figsize=(12, 6))  # Set the figure size
plt.plot(Test['MCP'].reset_index(drop=True), label='Actual', color='red')  # Plot the actual values
plt.plot(np.exp(pred_add_sea_exp).reset_index(drop=True), label='Predicted', color='blue')  # Plot the predicted values
plt.legend()  # Display the legend
plt.xlabel('Time')  # Label for x-axis
plt.ylabel('MCP')  # Label for y-axis
plt.title('Actual vs Predicted MCP')  # Title of the plot
plt.grid(True)  # Show grid

# Set the x-axis and y-axis limits to zoom in
plt.xlim(0, 100)  # Adjust these values to zoom in on the desired range of the x-axis
plt.ylim(0, 5000)  # Adjust these values to zoom in on the desired range of the y-axis

plt.show()  # Display the plot

# building the best model for the entire dataset
add_sea_exp_model_full_model = smf.ols('log_MCP ~ t + t00 + t15 + t30', data = df1).fit() # Fit the additive seasonality with exponential trend model

# Read newdata for forecasting
predict_data = pd.read_excel(r"C:\Users\aathi\Downloads\PROJECT_360_2\Dataset\Extention.xlsx")
predict_data = pd.DataFrame(predict_data)
predict_data.info() # Display the column names and data types
df1.info()
# changing '00', '15', '30', '45' to booliean
predict_data['t00'] = predict_data['t00'].astype('bool')
predict_data['t15'] = predict_data['t15'].astype('bool')
predict_data['t30'] = predict_data['t30'].astype('bool')
predict_data['t45'] = predict_data['t45'].astype('bool')
predict_data.head() # Display the first few rows of the DataFrame

# Generate forecasts for the new data using the trained model ('model_full')
forecast = add_sea_exp_model_full_model.predict(predict_data)  # Make predictions on the new data
forecast = np.exp(forecast) # Take the exponential of the predictions to get the actual values
predict_data['Forecasted_MCP'] = pd.Series(forecast) # Add the forecasted values to the new data DataFrame

# Save the model
# Save the model to a file using joblib or pickle
add_sea_exp_model_full_model.save("add_sea_exp_model_full_model.pkl") # Save the model to a file
print("Model saved successfully!") # Print a message indicating that the model has been saved

## loading the model
model = OLSResults.load("add_sea_exp_model_full_model.pkl") # Load the model from the file
print("Model loaded successfully!") # Print a message indicating that the model has been

# Residuals might have additional information
# Autoregression model
# Calculating Residuals from best model applied on full data
# AV - FV = Residuals
full_res = df1['MCP'] - add_sea_exp_model_full_model.predict(df1) # Calculate the residuals

# ACF plot on residuals to check for autocorrelation
import statsmodels.graphics.tsaplots as tsa_plots # Import the necessary library for plotting ACF
tsa_plots.plot_acf(full_res, lags = 4) # Plot the ACF for the residuals
plt.show() # Display the ACF plot 

# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of Y with lags of the residuals of the time series 
tsa_plots.plot_pacf(full_res, lags = 4)
plt.show()

# AR Autoregressive model
# Build an AutoRegressive (AR) model with lag 1 using Statsmodels
from statsmodels.tsa.ar_model import AutoReg # Import the AutoReg class from statsmodels

model_ar = AutoReg(full_res, lags = [1]) # Fit an AR model with lag 1
model_fit = model_ar.fit() # Fit the model to the data

# Print the estimated coefficients of the AR(1) model
print('Coefficients: %s' % model_fit.params)
# Generate predictions using the fitted AR(1) model
pred_res = model_fit.predict(start = len(full_res), end = len(full_res) + len(predict_data) - 1, dynamic = False)
# Convert predictions to a Pandas Series and remove index for easier handling
pred_res.reset_index(drop = True, inplace = True)

# Combine forecasts from the previous model ('pred_new') and AR(1) model ('pred_res')
final_pred = forecast + pred_res

# Display the final predictions after combining forecasts
final_pred

os.chdir(r"C:\Users\aathi\Downloads\PROJECT_360_2\Dataset")
# Save the final predictions to an Excel file
final_pred.to_excel("final_predictions.xlsx", index = False) # Save the final predictions to an Excel file

# adding final prediction to predicted data dataframe
predict_data['Final_Predictions'] = final_pred # Add the final predictions to the new data DataFrame
predict_data.head() # Display the first few rows of the DataFrame

# sending predicted data to excel file for further analysis
predict_data.to_excel("final_forecasted_data.xlsx", index = False) # Save the predicted data to an Excel file


# plot for final predictions with 95% confidence interval
# Plot the forecasted MCP values with 95% confidence interval  
plt.figure(figsize=(12, 6))  # Set the figure size  
plt.plot(predict_data['date'], final_pred, label='Forecast', color='red')  # Plot the forecasted values
plt.fill_between(predict_data['date'], final_pred - 2, final_pred + 2, color='pink', alpha=0.5)  # Add a 95% confidence interval   

# Annotate the forecasted data points
for i, txt in enumerate(final_pred):
    plt.annotate(f'{txt:.2f}', (predict_data['date'][i], final_pred[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.legend()  # Display the legend
plt.xlabel('date')  # Label for x-axis   
plt.ylabel('MCP')  # Label for y-axis
plt.title('Forecasted MCP with 95% Confidence Interval')  # Title of the plot
plt.grid(True)  # Show grid
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()  # Display the plot

# ARIMA model
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Function to check stationarity
def adf_test(series):
    result = adfuller(series.dropna())  # Remove NaN values before applying the test
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] > 0.05:
        print("Data is non-stationary. Applying differencing.")
        return False
    else:
        print("Data is stationary.")
        return True

# Apply ADF test on the original 'MCP' column
if not adf_test(Train['MCP']):
    # First differencing (d=1)
    Train['MCP_diff'] = Train['MCP'].diff().dropna()
    print("\nADF Test After Differencing:")
    adf_test(Train['MCP_diff'])
    train_series = Train['MCP_diff']
else:
    train_series = Train['MCP']

# Plot ACF and PACF for differenced data
plt.figure(figsize=(12, 6))

plt.subplot(121)
plot_acf(train_series.dropna(), ax=plt.gca(), lags=10)
plt.title("Autocorrelation Function (ACF)")

plt.subplot(122)
plot_pacf(train_series.dropna(), ax=plt.gca(), lags=10)
plt.title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.show()

# Fit the ARIMA model with updated order (p=1, d=2, q=4 based on ACF/PACF)
model = ARIMA(Train['MCP'], order=(1, 2, 4))  
res = model.fit()

# Forecasting on the test set
forecast = res.predict(start=len(Train), end=len(Train)+len(Test)-1, dynamic=False)

# RMSE Calculation
rmse = np.sqrt(mean_squared_error(Test['MCP'], forecast))
print(f'RMSE: {rmse}')

# MAPE Calculation
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(Test['MCP'], forecast)
print(f'MAPE: {mape}%')

# Visualizing forecast vs. actual values
plt.figure(figsize=(10, 5))
plt.plot(Test['MCP'], label='Actual')
plt.plot(forecast, color='red', linestyle="--", label='Predicted')
plt.legend()
plt.title("Actual vs Predicted MCP Values")
plt.show()


#SARIMA model
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Fit the SARIMA model on the training data
model = SARIMAX(Train['MCP'], order=(1, 2, 4), seasonal_order=(1, 1, 1, 96))  
res = model.fit()

# Forecast on the test set
forecast = res.predict(start=len(Train), end=len(Train) + len(Test) - 1, dynamic=False)

# RMSE Calculation
rmse = np.sqrt(mean_squared_error(Test['MCP'], forecast))
print(f'RMSE: {rmse:.4f}')

# MAPE Calculation
mape = np.mean(np.abs((Test['MCP'] - forecast) / Test['MCP'])) * 100
print(f'MAPE: {mape:.2f}%')

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(Test['MCP'], label='Actual MCP', color='blue')
plt.plot(forecast, label='SARIMA Prediction', color='red', linestyle='dashed')
plt.legend()
plt.title("MCP Forecasting using SARIMA")
plt.show()



