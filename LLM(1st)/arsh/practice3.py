from sklearn.linear_model import LinearRegression
import numpy as np 

x=np.array([[1],[2],[3],[4],[5]])
y=np.array([30,60,90,120,150])

model=LinearRegression()

model.fit(x,y)

prediction=model.predict([[9]])

print(prediction)