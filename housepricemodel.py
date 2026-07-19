

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


#1: Load data
df = pd.read_csv("house_prices_kaggle.csv")
print(f"Dataset shape: {df.shape}")


#2: Select features & handle missing values
features = [
    "OverallQual", "GrLivArea", "GarageCars", "GarageArea",
    "TotalBsmtSF", "1stFlrSF", "FullBath", "YearBuilt", "YearRemodAdd"
]
target = "SalePrice"

df[features] = df[features].fillna(df[features].median(numeric_only=True))

X = df[features]
y = df[target]


#3: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#4: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


#5: Predict house prices
y_pred = model.predict(X_test)


#6: Evaluate R² score (and other metrics)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
print(f"R² Score : {r2:.3f}")
print(f"MSE      : {mse:,.2f}")
print(f"RMSE     : {rmse:,.2f}")
print(f"MAE      : {mae:,.2f}")

print("\nFeature coefficients:")
for f, c in zip(features, model.coef_):
    print(f"  {f}: {c:,.2f}")


#7: Plot predicted vs actual values
sns.set_style("whitegrid")
plt.figure(figsize=(8, 7))

plt.scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolor="k", linewidth=0.3)

# perfect-prediction reference line (y = x)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Prediction (y=x)")

plt.xlabel("Actual Sale Price ($)")
plt.ylabel("Predicted Sale Price ($)")
plt.title(f"Predicted vs Actual House Prices  (R² = {r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
print("\nSaved chart -> predicted_vs_actual.png")


# 8: Save predictions to CSV
results = pd.DataFrame({
    "Actual_Price": y_test.values,
    "Predicted_Price": y_pred.round(2)
})
results.to_csv("house_price_predictions.csv", index=False)
print("Saved predictions -> house_price_predictions.csv")