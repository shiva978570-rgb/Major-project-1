import pandas as pd
from preprocess import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
df = load_data("data/Telco-Customer-Churn.csv")

# 2. Handle Missing
df = handle_missing(df)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Convert Target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 🔥 NEW FEATURES ADD
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
df["SeniorCitizen"] = df["SeniorCitizen"]

df["Contract"] = df["Contract"].map({
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
})

# Select Features
df = df[[
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "SeniorCitizen",
    "Contract",
    "Churn"
]]

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)

model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

# Accuracy
acc1 = accuracy_score(y_test, model1.predict(X_test))
acc2 = accuracy_score(y_test, model2.predict(X_test))

print("Logistic Regression Accuracy:", acc1)
print("Decision Tree Accuracy:", acc2)

# Best model
best_model = model1 if acc1 > acc2 else model2

# Save
joblib.dump(best_model, "model/model.pkl")

print("✅ Updated model saved!")