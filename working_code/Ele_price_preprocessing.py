#*Model-based Preprocessing*

# Importing necessary libraries
import pandas as pd  # Used for data manipulation and analysis
import numpy as np  # Used for numerical computations
import matplotlib.pyplot as plt  # Used for plotting graphs
from sqlalchemy import create_engine  # Allows interaction with databases
from urllib.parse import quote # Handles special character encoding in passwords
import os  # Used for file handling operations

user = 'root' # Database credentials for connection
pw = quote('root') # Encode the password to handle special characters
db = 'powertrading' # Name of the database
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}') # Create engine for database connection
# Retrieve the data from the database and load it into a DataFrame called ele_price_forecast
sql = 'SELECT * FROM power_tlb;'
df = pd.read_sql(sql, con=engine) # Load the data into a DataFrame containing date_time_MCP
df.head() # Display the first few rows of the DataFrame
df.info() # Display the column names and data types of the DataFrame

# Renaming the target column for better readability
df.rename(columns={"MCP (Rs/MWh) *": "MCP"}, inplace=True)

# Selecting only the relevant feature for model-based approaches
df2 = df[["Datetime", "MCP"]]
df2.info()


# Filtering out zero or negative values in the MCP column
zero_values = df2[df2["MCP"] <= 0] # Identify and filter out zero or negative values
zero_values # Display the zero or negative values in the MCP column

# Replace zero values with NaN
df2["MCP"] = df2["MCP"].replace(0, np.nan)

# Impute missing values using forward fill
df2["MCP"] = df2["MCP"].fillna(method="ffill")
# Impute missing values using backward fill
df2["MCP"] = df2["MCP"].fillna(method="bfill")

# Verify the absence of missing values in the dataframe
df2.isnull().sum()

# Re-check for any remaining zero or negative values in the MCP column
zero_values = df2[df2["MCP"] <= 0]
zero_values # Display any remaining zero or negative values in the MCP column

# Save the imputed data back to the database
df2.to_sql('power_tlb_imputed', con = engine, if_exists = 'replace', index = False) 

# Retrieve the imputed data from the database
sql = 'SELECT * FROM power_tlb_imputed'
df3 = pd.read_sql(sql, con=engine) # Load the data into a DataFrame containing date_time_MCP
df3.head() # Display the first few rows of the DataFrame    
df3.info() # Display the column names and data types of the DataFrame

# Create new columns to capture time and MCP trends
df3["t"] = np.arange(1, 75700 + 1) # Create a column 't' with sequential integers
df3["t_sq"] = df3["t"] * df3["t"] # Add a column 't_sq' with the square of 't'
df3["log_MCP"] = np.log(df3["MCP"]) # Add a column 'log_MCP' with the logarithm of 'MCP'

df3['Datetime'] = pd.to_datetime(df3['Datetime']) # Convert 'Datetime' to datetime format
df3["Datetime"] = df3["Datetime"].dt.minute.astype(int) # Extract the minute part from 'Datetime' column

# Generate dummy variables for 15-minute time intervals
min_dummies = pd.get_dummies(df3["Datetime"]) # Perform one-hot encoding for the minutes
df3 = pd.concat([df3, min_dummies], axis=1) # Concatenate the dummy variables with the original DataFrame

# Rename columns '00', '15', '30', '45' to more readable names 't00', 't15', 't30', 't45'
df3.rename(columns={0: 't00', 15: 't15', 30: 't30', 45: 't45'}, inplace=True)
df3.head() # Display the first few rows of the DataFrame

# Create a time series plot for the MCP column
df3.MCP.plot() # Plot the MCP column to visualize the trend
plt.show() # Display the plot

# Save the preprocessed data to the database
df3.to_sql('power_tlb_preprocessed', con = engine, if_exists = 'replace', index = False)
sql = 'SELECT * FROM power_tlb_preprocessed'# Load the data into a DataFrame
df3_preprocessed = pd.read_sql(sql, con=engine) # Load the data into a DataFrame containing date_time_MCP
df3_preprocessed.info() # Display the column names and data types

# Change directory to the location where you want to save the file
os.chdir(r"C:\Users\aathi\Downloads\PROJECT_360_2")
df3_preprocessed.to_excel("power_tlb_preprocessed_model_based.xlsx", index=False) # Save the preprocessed data to an Excel file
df3_preprocessed = df3_preprocessed.drop(columns=["Datetime"]) # Drop the original 'Datetime' column

# Linear trend fit
linear_fit = np.polyfit(df3_preprocessed["t"], df3_preprocessed["MCP"], 1)
df3_preprocessed["linear_fit"] = np.polyval(linear_fit, df3_preprocessed["t"])

# Exponential trend fit
exp_fit = np.polyfit(df3_preprocessed["t"], df3_preprocessed["log_MCP"], 1)
df3_preprocessed["exp_fit"] = np.exp(np.polyval(exp_fit, df3_preprocessed["t"]))

# Quadratic trend fit
quad_fit = np.polyfit(df3_preprocessed["t"], df3_preprocessed["MCP"], 2)
df3_preprocessed["quad_fit"] = np.polyval(quad_fit, df3_preprocessed["t"])

# Plotting the trends
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].plot(df3_preprocessed["t"], df3_preprocessed["MCP"], label="MCP", color="blue", alpha=0.6)
axes[0].plot(df3_preprocessed["t"], df3_preprocessed["linear_fit"], label="Linear Trend", color="red")
axes[0].set_title("Linear Trend")
axes[0].legend()
axes[0].grid()

axes[1].plot(df3_preprocessed["t"], df3_preprocessed["MCP"], label="MCP", color="blue", alpha=0.6)
axes[1].plot(df3_preprocessed["t"], df3_preprocessed["exp_fit"], label="Exponential Trend", color="green")
axes[1].set_title("Exponential Trend")
axes[1].legend()
axes[1].grid()

axes[2].plot(df3_preprocessed["t"], df3_preprocessed["MCP"], label="MCP", color="blue", alpha=0.6)
axes[2].plot(df3_preprocessed["t"], df3_preprocessed["quad_fit"], label="Quadratic Trend", color="purple")
axes[2].set_title("Quadratic Trend")
axes[2].legend()
axes[2].grid()

plt.xlabel("Time")
plt.show()
