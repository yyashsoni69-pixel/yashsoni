# ===============================
# EXPERIMENT 04: LINEAR REGRESSION
# ===============================

# ---------- IMPORT LIBRARIES ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ======================================================
# TASK 1: SIMPLE LINEAR REGRESSION USING NUMPY
# ======================================================

# Given data
x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([11, 16, 18, 30, 22, 38])

# Scatter plot
plt.scatter(x, y)
plt.xlabel("BMI")
plt.ylabel("Cholesterol")
plt.title("BMI vs Cholesterol")
plt.show()

# Function to calculate b0 and b1
def linear_regression_coefficients(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    b1 = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
    b0 = mean_y - b1 * mean_x
    return b0, b1

b0, b1 = linear_regression_coefficients(x, y)

print("Task 1 - NumPy Implementation")
print("Intercept (b0):", b0)
print("Slope (b1):", b1)

# Prediction for x = 27
x_test = 27
y_test = b0 + b1 * x_test
print("Predicted y for x = 27:", y_test)

# Regression line
y_line = b0 + b1 * x
plt.scatter(x, y, label="Data")
plt.plot(x, y_line, label="Regression Line")
plt.legend()
plt.show()


# ======================================================
# TASK 2: SIMPLE LINEAR REGRESSION USING SKLEARN
# ======================================================

x_reshaped = x.reshape(-1, 1)

model = LinearRegression()
model.fit(x_reshaped, y)

y_pred = model.predict(x_reshaped)

print("\nTask 2 - Sklearn Implementation")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])
print("R2 Score:", r2_score(y, y_pred))


# ======================================================
# TASK 3: SALARY DATASET (CSV UPLOADED IN JUPYTER)
# ======================================================

# Load dataset (already uploaded)
df = pd.read_csv("ML-P4-Salary_Data.csv")

print("\nDataset Head:")
print(df.head())

print("\nDataset Description:")
print(df.describe())

# Feature and target
X = df.iloc[:, 0].values.reshape(-1, 1)  # Years of Experience
Y = df.iloc[:, 1].values                # Salary

# Train model
salary_model = LinearRegression()
salary_model.fit(X, Y)

Y_salary_pred = salary_model.predict(X)

print("\nTask 3 - Salary Dataset")
print("Intercept:", salary_model.intercept_)
print("Slope:", salary_model.coef_[0])
print("R2 Score:", r2_score(Y, Y_salary_pred))

# Plot
plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_salary_pred, label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.legend()
plt.show()

# Cross verification using custom function
b0_salary, b1_salary = linear_regression_coefficients(X.flatten(), Y)
print("Custom Intercept:", b0_salary)
print("Custom Slope:", b1_salary)


# ======================================================
# TASK 4: MULTIPLE LINEAR REGRESSION (IF APPLICABLE)
# ======================================================

if df.shape[1] > 2:
    X_multi = df.iloc[:, :-1]
    Y_multi = df.iloc[:, -1]

    multi_model = LinearRegression()
    multi_model.fit(X_multi, Y_multi)

    print("\nTask 4 - Multiple Linear Regression")
    print("Intercept:", multi_model.intercept_)
    print("Coefficients:", multi_model.coef_)
else:
    print("\nTask 4 skipped (dataset has only one independent variable)")
