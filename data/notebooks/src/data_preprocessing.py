import pandas as pd
from sklearn.preprocessing import StandardScaler

import pandas as pd

def load_data(filepath="/Users/laxmankumarbusetty/Downloads/Ecommerce-sales-prediction/data/data.csv"):
    df = pd.read_csv(filepath, encoding="latin1")   # or encoding="ISO-8859-1"
    return df


def preprocess_data(df):
    # Drop duplicates and missing values
    df = df.drop_duplicates().dropna()

    # Convert date column
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Aggregate daily sales per product
    df["TotalSales"] = df['Quantity'] * df['UnitPrice']
    daily_sales = df.groupby(['StockCode', 'InvoiceDate']).agg({'TotalSales':'sum'}).reset_index()

    # Time-based features
    daily_sales['DayOfWeek'] = daily_sales['InvoiceDate'].dt.dayofweek
    daily_sales['Month'] = daily_sales['InvoiceDate'].dt.month
    daily_sales['WeekOfYear'] = daily_sales['InvoiceDate'].dt.isocalendar().week

    # Lag features for time series
    daily_sales['Lag1'] = daily_sales.groupby('StockCode')['TotalSales'].shift(1).fillna(0)
    daily_sales['RollingMean3'] = daily_sales.groupby('StockCode')['TotalSales'].transform(lambda x: x.rolling(3,1).mean())

    return daily_sales

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler
import pandas as pd

# Path to your dataset
data_path = "/Users/laxmankumarbusetty/Downloads/Ecommerce-sales-prediction/data/data.csv"

# Load your dataset
df = pd.read_csv(data_path)

# Add TotalSales column (Quantity × Price)
df["TotalSales"] = df["Quantity"] * df["Price"]   # 👈 replace with actual column names

# Save updated dataset (overwrite or new file)
df.to_csv("/Users/laxmankumarbusetty/Downloads/Ecommerce-sales-prediction/data/data_with_totalsales.csv", index=False)

print("✅ Done! TotalSales column added for all 500k rows.")

