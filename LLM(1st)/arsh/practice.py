from sklearn.linear_model import LinearRegression
import numpy as np 

x=np.array([[1],[2],[3],[4],[5]])
y=np.array([2,4,6,8,10])

model=LinearRegression()

model.fit(x,y)

prediction=model.predict([[7]])
print(prediction)