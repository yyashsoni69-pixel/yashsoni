git init
import pandas as pd

df = pd.read_csv("/mnt/data/ML-P4-Salary_Data.csv")
df.head()
df.describe()
X = df.iloc[:, 0].values.reshape(-1, 1)  # YearsExperience
Y = df.iloc[:, 1].values               # Salary
salary_model = LinearRegression()
salary_model.fit(X, Y)
print("Intercept:", salary_model.intercept_)
print("Slope:", salary_model.coef_[0])

Y_pred = salary_model.predict(X)
print("R2 Score:", r2_score(Y, Y_pred))
plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_pred, color='red', label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.legend()
plt.show()
b0_salary, b1_salary = linear_regression_coefficients(X.flatten(), Y)
print("Custom Intercept:", b0_salary)
print("Custom Slope:", b1_salary)
# streamlit_app.py
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("ML-P4-Salary_Data.csv")
X = df.iloc[:, 0].values.reshape(-1, 1)
Y = df.iloc[:, 1].values

model = LinearRegression()
model.fit(X, Y)

st.title("Salary Prediction App")

experience = st.number_input("Enter Years of Experience", 0.0, 50.0)
salary = model.predict(np.array([[experience]]))

st.write("Predicted Salary:", salary[0])
streamlit run streamlit_app.py
