import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Correct Windows path (raw string)
df = pd.read_csv(r"C:\Users\hp\Downloads\ML-P4-Salary_Data.csv")

print(df.head())
print(df.describe())

# Features & target
X = df.iloc[:, 0].values.reshape(-1, 1)   # YearsExperience
Y = df.iloc[:, 1].values                  # Salary

# Model
salary_model = LinearRegression()
salary_model.fit(X, Y)

print("Intercept:", salary_model.intercept_)
print("Slope:", salary_model.coef_[0])

# Prediction & score
Y_pred = salary_model.predict(X)
print("R2 Score:", r2_score(Y, Y_pred))

# Plot
plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_pred, label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.legend()
plt.show()



