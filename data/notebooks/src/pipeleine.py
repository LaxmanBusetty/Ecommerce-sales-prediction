# pipeline.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 1. Load Data
# ===============================
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "../Ecommerce.csv")  # adjust if filename differs
    df = pd.read_csv(data_path, encoding="latin1")
    
    # Parse dates if needed
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    
    return df


# ===============================
# 2. Preprocess Data
# ===============================
def preprocess_data(df):
    # Example feature engineering (you can adjust to your dataset)
    df = df.dropna()

    # Create target variable - assume 'TotalSales'
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["TotalSales"] = df["Quantity"] * df["UnitPrice"]

    # Features and target
    X = df[["Quantity", "UnitPrice"]]  # adjust with more features
    y = df["TotalSales"]

    return X, y


# ===============================
# 3. Train & Save Model
# ===============================
def train_and_save_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation:\nMSE: {mse:.2f}\nR²: {r2:.2f}")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "rf_sales_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Model saved at {model_path}")


# ===============================
# 4. Run Pipeline
# ===============================
if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)
    train_and_save_model(X, y)
