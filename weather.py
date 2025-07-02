import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('austin_weather.csv')

# Precipitation Prediction

data['PrecipitationSumInches'] = data['PrecipitationSumInches'].replace('T', 0.01)

for col in data.columns:
    if col not in ['Date', 'Events']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=['PrecipitationSumInches'])

X = data.drop(columns=['Date', 'Events', 'PrecipitationSumInches'])
y = data['PrecipitationSumInches']


X.fillna(X.mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:,.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Precipitation (inches)")
plt.ylabel("Predicted Precipitation (inches)")
plt.title("Actual vs Predicted Precipitation")
plt.grid(True)
plt.show()

# Seasonal Pattern Analysis
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month

monthly_avg = data.groupby('Month')[['TempAvgF', 'PrecipitationSumInches']].mean()

plt.figure(figsize=(8, 5))
monthly_avg['TempAvgF'].plot(marker='o')
plt.title("Average Temperature by Month")
plt.xlabel("Month")
plt.ylabel("Temperature (°F)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
monthly_avg['PrecipitationSumInches'].plot(marker='o', color='skyblue')
plt.title("Average Precipitation by Month")
plt.xlabel("Month")
plt.ylabel("Precipitation (inches)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Month', y='TempAvgF', data=data)
plt.title("Temperature Distribution")
plt.grid(True)
plt.show()