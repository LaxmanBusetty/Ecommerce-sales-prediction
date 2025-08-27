import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/Users/laxmankumarbusetty/Downloads/Ecommerce-sales-prediction/data/data.csv"  # your CSV file
MODEL_PATH = os.path.join(BASE_DIR, 'rf_sales_model.pkl')

# --------------------------
# Load or train model
# --------------------------
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.write("Training model... please wait")
    
    # Load dataset
    data = pd.read_csv(DATA_PATH, encoding='latin1')  # adjust encoding if needed

    # Feature engineering
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['DayOfWeek'] = data['InvoiceDate'].dt.dayofweek
    data['Month'] = data['InvoiceDate'].dt.month
    data['WeekOfYear'] = data['InvoiceDate'].dt.isocalendar().week
    TotalSales = data['Quantity'] * data['UnitPrice']
    data['Lag1'] = data['TotalSales'].shift(1)
    data['RollingMean3'] = data['TotalSales'].rolling(window=3).mean()
    data = data.dropna()

    # Features and target
    X = data[['DayOfWeek', 'Month', 'WeekOfYear', 'Lag1', 'RollingMean3']]
    y = TotalSales

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, MODEL_PATH)
    st.write("Model trained and saved!")

# --------------------------
# Streamlit App
# --------------------------
st.title("E-commerce Daily Sales Predictor")

st.sidebar.header("Input Features")
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 0)
month = st.sidebar.slider("Month", 1, 12, 1)
week_of_year = st.sidebar.slider("Week of Year", 1, 52, 1)
lag1 = st.sidebar.number_input("Lag1 (previous day sales)", 0.0)
rolling_mean3 = st.sidebar.number_input("Rolling Mean 3 days", 0.0)

# Input dataframe
input_data = pd.DataFrame({
    "DayOfWeek": [day_of_week],
    "Month": [month],
    "WeekOfYear": [week_of_year],
    "Lag1": [lag1],
    "RollingMean3": [rolling_mean3]
})

# Prediction
prediction = model.predict(input_data)[0]
st.write(f"Predicted Total Sales: ${prediction:.2f}")


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# y_test -> true values
# y_pred -> predictions from your trained model

r2 = r2_score(TotalSales, prediction)
mae = mean_absolute_error(TotalSales,prediction)
mse = mean_squared_error(TotalSales,prediction)
rmse = np.sqrt(mse)

print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
