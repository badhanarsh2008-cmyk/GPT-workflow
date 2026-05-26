import numpy as np 
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================================
# 1. LOAD DATASET
# =====================================================================
# Dynamically locate phone_addiction.csv in the same directory as this script
csv_path = Path(__file__).with_name("phone_addiction.csv")
data = pd.read_csv(csv_path)

print("--- Dataset Preview ---")
print(data.head())

# Input (Features) and Output (Target)
X = data[['ScreenTime', 'SocialMediaHours', 'SleepHours']]
y = data['Addicted']

# =====================================================================
# 2. TRAIN & EVALUATE MODEL
# =====================================================================
# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Decision Tree Classifier
# (max_depth=3 keeps the visual plot clean and readable!)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Calculate testing accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# =====================================================================
# 3. MATPLOTLIB VISUALIZATIONS
# =====================================================================
print("\nGenerating Visualizations...")

# Plot 1: The Decision Tree Logic Blueprint
plt.figure(figsize=(14, 8))
plot_tree(
    model, 
    feature_names=X.columns, 
    class_names=['Not Addicted', 'Addicted'], 
    filled=True, 
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Trained Logic Blueprint", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Plot 2: Feature Importance Bar Chart
importances = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(X.columns, importances, color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
plt.xlabel('Importance Weight Score')
plt.ylabel('User Behavior Metrics')
plt.title('Which Feature Dictates Phone Addiction the Most?', fontsize=14, fontweight='bold')
plt.xlim(0, 1.0)
plt.tight_layout()
plt.show()

# Plot 3: Scatter Pattern Check (Screen Time vs. Sleep Hours)
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    data['ScreenTime'], 
    data['SleepHours'], 
    c=data['Addicted'], 
    cmap='bwr',       # Blue = Not Addicted, Red = Addicted
    alpha=0.75, 
    edgecolors='k'
)
plt.xlabel('Daily Screen Time (Hours)', fontsize=12)
plt.ylabel('Nightly Sleep Hours', fontsize=12)
plt.title('Dataset Clusters: Screen Time vs. Sleep', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ticks=[0, 1], label="0: Not Addicted | 1: Addicted")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =====================================================================
# 4. CUSTOM USER PREDICTION
# =====================================================================
print("\n--- Custom Live Prediction ---")
screen_time = float(input("Enter Screen Time (Hours): "))
social_media = float(input("Enter Social Media Hours: "))
sleep = float(input("Enter Sleep Hours: "))

# Build input into a DataFrame matching training columns to avoid feature name warnings
custom_data = pd.DataFrame(
    [[screen_time, social_media, sleep]],
    columns=['ScreenTime', 'SocialMediaHours', 'SleepHours']
)

result = model.predict(custom_data)

print("\n--- Diagnostic Result ---")
if result[0] == 1:
    print("Result: Person is Smartphone Addicted")
else:
    print("Result: Person is Not Addicted")