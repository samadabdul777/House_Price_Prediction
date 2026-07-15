# ==========================
# House Price Prediction
# California Housing Dataset
# ==========================

# Import libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# ==========================
# Load Dataset
# ==========================

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

# ==========================
# Explore Dataset
# ==========================

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nDataset Shape")
print(df.shape)

print("\nStatistical Summary")
print(df.describe())

# ==========================
# Missing Values
# ==========================

print("\nMissing Values")
print(df.isnull().sum())

# ==========================
# Correlation
# ==========================

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix")
print(correlation)

print("\nCorrelation with Price")
print(correlation["Price"].sort_values(ascending=False))

# ==========================
# Histograms
# ==========================

df.hist(figsize=(12,8))
plt.tight_layout()
plt.show()

# ==========================
# Split Features & Target
# ==========================

X = df.drop("Price", axis=1)
y = df["Price"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Feature Shape:", X_train.shape)
print("Testing Feature Shape:", X_test.shape)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed")

# ==========================
# Train Model
# ==========================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# ==========================
# Predictions
# ==========================

predictions = model.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nFirst 10 Predictions")
print(results.head(10))

# ==========================
# Evaluation
# ==========================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ==========================
# Coefficients
# ==========================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients")
print(coefficients)

# ==========================
# Actual vs Predicted Graph
# ==========================

plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# ==========================
# Save Model
# ==========================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/house_price_model.pkl")

print("\nModel Saved Successfully")

# ==========================
# Load Model
# ==========================

loaded_model = joblib.load("models/house_price_model.pkl")

print("Model Loaded Successfully")

# ==========================
# Predict New House
# ==========================

new_house = [[
    8.5,        # MedInc
    20.0,       # HouseAge
    6.5,        # AveRooms
    1.0,        # AveBedrms
    1200.0,     # Population
    3.0,        # AveOccup
    34.05,      # Latitude
    -118.25     # Longitude
]]

new_house = scaler.transform(new_house)

predicted_price = loaded_model.predict(new_house)

print("\nPredicted House Price:", predicted_price[0])