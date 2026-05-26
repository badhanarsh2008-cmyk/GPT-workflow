from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score

data=pd.read_csv("StudentPerformanceFactors.csv")

X=data.drop("Exam_Score",axis=1)
Y=data["Exam_Score"]

X=pd.get_dummies(X)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.2)

model=LinearRegression()

model.fit(X_train,Y_train)

prediction=model.predict(X_test)

mse = mean_squared_error(Y_test, prediction)
r2 = r2_score(Y_test, prediction)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

sample = X_test.iloc[0:1]
predicted_score = model.predict(sample)

print("\nPredicted Exam Score:", predicted_score)
print("Actual Exam Score:", Y_test.iloc[0])
