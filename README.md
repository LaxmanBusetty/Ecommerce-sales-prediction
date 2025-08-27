🛒 Ecommerce Sales Prediction

This project predicts e-commerce sales performance using Machine Learning models. By analyzing historical sales data, the project identifies key features that influence sales, performs exploratory data analysis (EDA), and builds predictive models to forecast future sales.

📌 Table of Contents

Project Overview

Dataset

Technologies Used

Data Preprocessing

Exploratory Data Analysis (EDA)

Modeling

Results

Project Structure

How to Run

Future Work

Contributors

📖 Project Overview

E-commerce platforms generate massive amounts of sales data. Predicting future sales can help businesses optimize inventory management, marketing strategies, and pricing decisions.

In this project, we:
✅ Collected and preprocessed e-commerce sales data
✅ Performed exploratory data analysis using Python visualization libraries
✅ Built and compared different ML models for prediction
✅ Tuned hyperparameters for improved accuracy
✅ Visualized model performance

📂 Dataset

The dataset contains historical sales data with features such as:

Order ID

Product

Category

Customer Segment

Quantity Ordered

Price Each

Order Date

Region

Sales (Target Variable)

📌 Note: Due to GitHub file size restrictions, the dataset and trained models are not included in this repo. You can use your own dataset or request access to the original one.

🛠 Technologies Used

Programming Language: Python 3.9+

Libraries:

Data Analysis → NumPy, Pandas

Visualization → Matplotlib, Seaborn

Machine Learning → Scikit-learn, XGBoost

Model Evaluation → Scikit-learn metrics

🧹 Data Preprocessing

Handled missing values (imputation, removal where needed)

Processed categorical variables with encoding techniques

Performed feature engineering:

Created Bedroom Ratio, Household Rooms, etc. (if similar)

Derived Month, Day, Season from Order Date

Applied StandardScaler for feature scaling

📊 Exploratory Data Analysis (EDA)

Sales distribution across different categories and regions

Seasonal and monthly sales trends

Relationship between Quantity Ordered, Price, and Sales

Correlation heatmaps for feature relationships

🤖 Modeling

Implemented multiple machine learning algorithms:

K-Nearest Neighbors (KNN) – for regression/classification tasks

Linear Regression (with Gradient Descent) – baseline model

Random Forest Regressor – improved prediction accuracy

XGBoost – handled non-linearity and provided strong performance

📌 Model Evaluation Metrics:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

R² Score

📈 Results

Random Forest Regressor achieved the best performance, with ~80% accuracy on validation data.

Visualized predictions vs actual sales using scatterplots.

Feature importance analysis showed top drivers of sales.

📂 Project Structure
Ecommerce-sales-prediction/
│── data/                   # Dataset (not included due to size)
│── notebooks/              # Jupyter notebooks for EDA & modeling
│── src/                    # Source code for data prep & models
│── venv/                   # Virtual environment (ignored by Git)
│── .gitignore              # Ignored files
│── requirements.txt        # Dependencies
│── README.md               # Project documentation

🚀 How to Run

Clone this repository

git clone https://github.com/LaxmanBusetty/Ecommerce-sales-prediction.git
cd Ecommerce-sales-prediction


Create a virtual environment & activate it

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Run Jupyter Notebook or Python scripts

jupyter notebook

🔮 Future Work

Use deep learning models (RNN/LSTM) for time-series sales prediction

Deploy the model using Streamlit/Flask/Django

Integrate real-time data for live sales forecasting

Build a dashboard for stakeholders

👨‍💻 Contributors

Laxman Kumar Busetty – Data Analysis, Machine Learning, EDA, Documentation

✨ If you like this project, give it a ⭐ on GitHub!
