import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\onedrive\Desktop\LLM\pytorch\Salary_dataset.csv")

df = pd.DataFrame(data)
print(df.head())

x = df.iloc[:, :-1]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(f"{x_test.iloc[0]} | {y_pred[0]:.2f}")

plt.scatter(x_test.iloc[:,0], y_test.values)        # actual data
plt.plot(x_test, y_pred)          # prediction line
plt.show()
