import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
csv_path = Path(__file__).with_name("phone_addiction.csv")
data = pd.read_csv(csv_path)

# Show dataset
print("Dataset:")
print(data.head())

# Input and output
X = data[['ScreenTime', 'SocialMediaHours', 'SleepHours']]
y = data['Addicted']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Visualize dataset
colors = data['Addicted'].map({0: 'green', 1: 'red'})

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data['ScreenTime'], data['SleepHours'], c=colors)
plt.xlabel('Screen Time')
plt.ylabel('Sleep Hours')
plt.title('Screen Time vs Sleep Hours')

plt.subplot(1, 2, 2)
plt.scatter(data['SocialMediaHours'], data['SleepHours'], c=colors)
plt.xlabel('Social Media Hours')
plt.ylabel('Sleep Hours')
plt.title('Social Media Hours vs Sleep Hours')

plt.tight_layout()
plt.show()

# Visualize feature importance
plt.figure(figsize=(7, 4))
plt.bar(X.columns, model.feature_importances_, color=['skyblue', 'orange', 'lightgreen'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.show()

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Custom prediction
print("\nCustom Prediction")

screen_time = float(input("Enter Screen Time: "))
social_media = float(input("Enter Social Media Hours: "))
sleep = float(input("Enter Sleep Hours: "))

custom_data = pd.DataFrame(
    [[screen_time, social_media, sleep]],
    columns=['ScreenTime', 'SocialMediaHours', 'SleepHours']
)
result = model.predict(custom_data)

if result[0] == 1:
    print("Person is Smartphone Addicted")
else:
    print("Person is Not Addicted")

