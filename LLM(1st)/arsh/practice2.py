from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

sh=np.array([[1],[2],[3],[4],[5]])
marks=np.array([30,40,50,60,70])

model=LinearRegression()

model.fit(sh,marks)

u=int(input("Enter how many hours you studied:"))

prediction=model.predict([[u]])
print("Estimated prediction by model :",prediction)
if prediction>100:
    prediction=100

print("Actual or possible outcome :",prediction)

data=pd.DataFrame({
    "Hours Studied": sh.flatten(),
    "Marks":marks
})
print(data)
