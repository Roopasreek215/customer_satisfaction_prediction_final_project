# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load Data
df = pd.read_csv("customer_data.csv")  # Replace with your actual file path
print(df.head())

# Step 3: EDA (basic)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Step 4: Data Cleaning and Preprocessing
df.dropna(inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Visual 1: Customer Satisfaction Distribution
sns.countplot(x="Satisfaction", data=df)
plt.title("Customer Satisfaction Distribution")
plt.xlabel("Satisfaction Level")
plt.ylabel("Number of Customers")
plt.show()

# Feature and target separation
X = df.drop("Satisfaction", axis=1)
y = df["Satisfaction"]

# Optional: Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visual 2: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 8: Save Results
results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
results.to_csv("prediction_results.csv", index=False)
df.to_csv("cleaned_customer_data.csv", index=False)
