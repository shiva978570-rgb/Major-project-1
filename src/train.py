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

# 🔥 FIX TotalCharges (FINAL)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 3. Convert Target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 🔥 IMPORTANT: Only required features
df = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]]

# 4. Split Features & Target
X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 6. Model 1
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)

# 7. Model 2
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

# 8. Prediction
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)

# 9. Accuracy
acc1 = accuracy_score(y_test, y_pred1)
acc2 = accuracy_score(y_test, y_pred2)

print("Logistic Regression Accuracy:", acc1)
print("Decision Tree Accuracy:", acc2)

# 10. Best Model
best_model = model1 if acc1 > acc2 else model2

# 11. Save Model
joblib.dump(best_model, "model/model.pkl")

print("✅ Best model saved!")