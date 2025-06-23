# AI-Powered Electricity Price Forecasting

## 1. Understanding Business Problem, Business Objectives & constraints, Success Criterias
### Business Problem
In power trading, price volatility and demand-supply imbalances create significant financial risks for market participants. Accurately forecasting electricity prices, demand patterns, and renewable energy generation can enhance decision-making for utilities, grid operators, and power traders. The challenge lies in integrating multiple data sources, handling market uncertainties, and optimizing trading strategies in real-time.
### Business Objective
Minimizing financial risks and procurement costs in power trading.

### Business Constraint
Maximize trading returns.

### Success Criteria
- **Business Success Criteria**: At least 15 - 20% reduction in financial risks due to price volatility, leading to increased profitability for power traders and utilities.
- **Machine Learning Success Criteria**: Achieving a Mean Absolute Percentage Error (MAPE) below 10% compared to baseline models.
- **Economic Success Criteria**: Optimized trading strategies leading to 5 - 10% savings in energy procurement costs and a 10 - 15% increase in returns from power trading.

## 2. Project architecture (HLD)
Using Draw.io, created project architecture. Architecture contains CRISP-ML(Q) type flow.

## 3. Articles Collection
Collected project related articles from differenet websites like google scholor, medium etc. Collect useful insights like models, evaluation metrics, datasets etc.

## 4. Data Collection 
Firstly, we identified the features related to business problem and we collect data from the web(primary Data collection). Took data from the IEX website and data sending and retreving to sql database.

## 5. Environment Creation
creating environment in command prompt and installing required libraries

## 6. EDA (Exploratory Data Analysis)
By usinng python programming language in Vscode IDE, performed EDA.              
- **Key Points in EDA**
              1. Descriptive Statistics
              2. Duplicates
              3. Missing values
              4. Null values and 0 values
              5. Variance Analysis
              6. Correlation Analysis
              7. Distribution Analysis
              8. Outliers Analysis
              9. Q-Q Analysis
              10. Time Series Analysis
                     -- Trend 
                     -- seasonality
                     -- Residuals Analysis
                    
## 7. Datapreprocessing
Based on the models data preprocessing will be done
- **Model Based Approches preprocessing**
            - Handling 0 values and null values
            - Removed unwanted columns
            - Changing column names(if required)
            - Type casting
            - Feature Engineering of linear, quadratic, Exponential 
            - Dummy variable creation of Time intervals
- **Data Based Approches preprocessing**
            - Handling 0 values and null values
            - Removed unwanted columns
            - Changing column names(if required)
            - Type casting

- **Lasso, Ridge, Elastic Net, Bayesian Ridge preprocessing**
            - Handling 0 values and null values
            - Removed unwanted columns
            - Changing column names(if required)
            - Type casting
            - feature engineering(96 lags of MCP)
            - feature scaling(robust scaling)
            

- **LSTM, XGB, RandomForest preprocessing**
            - Handling 0 values and null values
            - Removed unwanted columns
            - Changing column names(if required)
            - Type casting
            - feature engineering(96 lags of MCP)
            - feature scaling(robust scaling)
            - Train test split
            - Hyperparameter tuning for XGB
            - Hyperparameter tuning for RandomForest
            - Hyperparameter tuning for LSTM

## partitioning data
            - for Training(Upto end of 2024)  remaining Testing is 5,526 rows (Staring of 2025).

## 8. Model Building
- **Model Based Approches**
              1. Linear
              2. Exponential
              3. Quadratic
              4. Additive Seasonality
              5. Multiplicative Seasonality
              6. Additive Seasonality with quadratic trend
              7. Multiplicative Seasinality with exponential trend
              8. Additive seasonality with exponential trend 
              9. Multiplicative seasonality with quadratic trend
- **Data Based Approches**
              1. Simple Exponential Smoothing
              2. Holt Method
              3. "Holts winter exponential smoothing with
               additive seasonality and additive trend "
              4. "Holts winter exponential smoothing 
              with multiplicative seasonality and additive trend"
              5. Moving Average
- **ARIMA**
- **auto_ARIMA**
- **SARIMA**
- **LASSO**
- **Ridge**
- **ElasticNet**
- **Bayesian Ridge Regression**
- **LSTM**
- **Random Forest**
- **XGB**

# Model Evaluation 
              1.RMSE
              2.MAPE
       
# Model Deployment
              
              1.Streamlit


              
             






