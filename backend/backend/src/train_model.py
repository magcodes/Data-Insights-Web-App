import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os

# Load dataset
df = pd.read_csv("backend/data/sales_data.csv")

# Features and target
X = df[["Month", "Marketing_Spend", "Customers", "Discount", "Website_Visits"]]
y = df["Sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Feature importance (coefficients)
coefficients = dict(zip(X.columns, model.coef_))

# Monthly summary
monthly_sales = df.groupby("Month")["Sales"].sum().reset_index()
best_month_row = monthly_sales.loc[monthly_sales["Sales"].idxmax()]

# Prepare JSON output
results = {
    "summary": {
        "total_sales": round(df["Sales"].sum(), 2),
        "average_sales": round(df["Sales"].mean(), 2),
        "best_month": int(best_month_row["Month"]),
        "best_month_sales": round(best_month_row["Sales"], 2)
    },
    "model_metrics": {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4)
    },
    "feature_importance": {
        key: round(value, 4) for key, value in coefficients.items()
    },
    "chart_data": [
        {"month": int(row["Month"]), "sales": round(row["Sales"], 2)}
        for _, row in monthly_sales.iterrows()
    ],
    "sample_predictions": [
        {
            "actual": round(float(actual), 2),
            "predicted": round(float(pred), 2)
        }
        for actual, pred in zip(y_test.head(10), y_pred[:10])
    ]
}

# Save JSON
os.makedirs("backend/output", exist_ok=True)

with open("backend/output/sales_insights.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Model trained successfully!")
print("📁 JSON saved at backend/output/sales_insights.json")