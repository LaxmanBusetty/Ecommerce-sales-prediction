import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model(df):
    feature_cols = ['DayOfWeek', 'Month', 'WeekOfYear', 'Lag1', 'RollingMean3']
    target_col = 'TotalSales'

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators':[100,200],
        'max_depth':[5,10,None],
        'min_samples_split':[2,5],
        'min_samples_leaf':[1,2]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Params: {grid_search.best_params_}")
    print(f"RMSE: {rmse:.2f}, R2: {r2:.2f}")

    joblib.dump(best_model, 'rf_sales_model.pkl')
    return best_model
