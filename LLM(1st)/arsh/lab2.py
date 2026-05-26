#Program to Implement Linear Regression for Single Variable using Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1) 
y = np.array([30, 35, 45, 50, 60, 65, 75, 80, 90, 95])

# 2. Create and Train the Model
model = LinearRegression()
model.fit(X, y) 


y_pred = model.predict(X)


print(f"Intercept (b0): {model.intercept_:.2f}")
print(f"Coefficient/Slope (b1): {model.coef_[0]:.2f}")
print(f"R-squared score: {model.score(X, y):.2f}") 


plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Test Scores')
plt.title('Linear Regression with Scikit-learn')
plt.legend()
plt.show()