# Import necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
from sqlalchemy import create_engine
from urllib.parse import quote
from PIL import Image
import os
import pymysql
import warnings

warnings.filterwarnings("ignore")
# Set working directory
os.chdir(r"C:\Users\aathi\Downloads\PROJECT_360_2\deployment\deployment")

# Load the trained XGBoost model and scaler
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit App Layout
st.set_page_config(page_title="MCP Forecasting", layout="wide")

def preprocess_data(df):
    """Preprocess user-uploaded data for forecasting."""
    df["Datetime"] = df["Datetime"].astype('datetime64[ns]')
    df.drop(columns=["Session ID", "MCV (MW)", "Unnamed: 0"], errors="ignore", inplace=True)
    df.rename(columns={"MCP (Rs/MWh) *": "MCP"}, inplace=True)
    df.replace(0, np.nan, inplace=True)
    df.fillna(method="bfill", inplace=True)
    
    # Create lag features
    for i in range(1, 97):
        df[f'MCP_lag_{i}'] = df['MCP'].shift(i)
    
    df.dropna(inplace=True)
    return df

def forecast_future_values(last_known_values, steps, confidence_level):
    """Forecast future MCP values using the last available window"""
    future_predictions = []
    upper_bounds = []
    lower_bounds = []
    
    current_window = last_known_values.copy()
    error_std = np.std([val[-1] for val in current_window] - model.predict(current_window))
    conf_multiplier = {95: 1.96, 90: 1.645, 99: 2.576}[confidence_level]
    
    for _ in range(steps):
        # Predict next value
        scaled_pred = model.predict(current_window[-1].reshape(1, -1))[0]
        
        # Calculate confidence intervals
        upper = scaled_pred + conf_multiplier * error_std
        lower = scaled_pred - conf_multiplier * error_std
        
        future_predictions.append(scaled_pred)
        upper_bounds.append(upper)
        lower_bounds.append(lower)
        
        # Update window without unnecessary scaling
        new_window = np.roll(current_window[-1], -1)
        new_window[-1] = scaled_pred
        current_window = np.vstack([current_window, new_window])
    
    return future_predictions, upper_bounds, lower_bounds

def main():
    """Streamlit application for MCP forecasting."""
    image = Image.open("AiSPRY logo.jpg")
    st.sidebar.image(image)

    st.markdown("""
    <style>
    .title-box {
        background-color: #FF5733;
        color: white;
        text-align: center;
        padding: 15px;
        border-radius: 15px;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    </style>
    <div class="title-box">Electricity Price Forecasting</div>
    """, unsafe_allow_html=True)

    # User inputs for database credentials and table name
    db_user = st.sidebar.text_input("Database User", "root")  
    db_password = st.sidebar.text_input("Password", "root", type="password")  
    db_name = st.sidebar.text_input("Database Name", "powertrading")  
    table_name = st.sidebar.text_input("Table Name", "results")  

    forecast_steps = st.sidebar.number_input("Forecast Steps", 1, 500, 200)
    confidence_level = st.sidebar.selectbox("Confidence Level", [95, 90, 99], 0)
    
    uploaded_file = st.sidebar.file_uploader("Upload Data", ["csv", "xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        processed_df = preprocess_data(df)
        
        # Get last available window
        X = processed_df.drop(columns=["MCP", "Datetime"]).values[-96:]
        predictions, upper, lower = forecast_future_values(X, forecast_steps, confidence_level)
        last_date = processed_df["Datetime"].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_steps+1, freq="15T")[1:]
        # Create results dataframe
        forecast_df = pd.DataFrame({
            "Datetime": future_dates,
            "Predicted_MCP": predictions,
            "Upper_CI": upper,
            "Lower_CI": lower
        })
        # vizualized future dates with a fixed interval of 15 minutes ('15T')
        if st.button("Generate Forecast"):

            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_df["Datetime"], 
                y=forecast_df["Predicted_MCP"],
                mode="lines+markers", 
                name="Forecast"
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["Datetime"], 
                y=forecast_df["Upper_CI"],
                line=dict(dash="dot"), 
                name="Upper Bound"
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["Datetime"], 
                y=forecast_df["Lower_CI"],
                fill="tonexty", 
                line=dict(dash="dot"), 
                name="Lower Bound"
            ))
            st.plotly_chart(fig)
            
            # Show results
            st.write("### Forecasted Values")
            st.dataframe(forecast_df)

        if st.button("💾 Save to Database"):
            if db_user and db_password and db_name and table_name:
                    try:
                        # Create a database engine with provided credentials
                        engine = create_engine(f"mysql+pymysql://{db_user}:{quote(db_password)}@localhost/{db_name}")
                        forecast_df.to_sql(table_name, con=engine, if_exists="replace", index=False)  # Append new data instead of replacing existing records
                        st.success("✅ Results successfully stored in the database.")
                        st.write(f"Data has been appended to the table `{table_name}` in database `{db_name}`.")
                    except Exception as e:
                        st.error(f"❌ Error saving to database: {e}")

if __name__ == "__main__":
    main()
# End of script