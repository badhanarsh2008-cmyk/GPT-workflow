import pandas as pd 

data=pd.read_csv("StudentPerformanceFactors.csv")
data2=data.head(data)
table=pd.DataFrame(data2)

print(table)