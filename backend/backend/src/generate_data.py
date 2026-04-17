import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 120

months = np.tile(np.arange(1, 13), 10)[:n]
marketing_spend = np.random.randint(2000, 20000, n)
customers = np.random.randint(50, 500, n)
discount = np.random.randint(0, 31, n)
website_visits = np.random.randint(1000, 15000, n)

seasonality = (
    np.where(np.isin(months, [11, 12]), 5000, 0)
    + np.where(np.isin(months, [6, 7]), 2000, 0)
)

sales = (
    5000
    + 2.8 * marketing_spend
    + 45 * customers
    - 120 * discount
    + 0.9 * website_visits
    + seasonality
    + np.random.normal(0, 3000, n)
)

df = pd.DataFrame({
    "Month": months,
    "Marketing_Spend": marketing_spend,
    "Customers": customers,
    "Discount": discount,
    "Website_Visits": website_visits,
    "Sales": sales.round(2)
})

os.makedirs("backend/data", exist_ok=True)
df.to_csv("backend/data/sales_data.csv", index=False)

print("✅ Dataset created successfully!")
print(df.head())