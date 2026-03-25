from preprocess import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load data
df = load_data("data/Telco-Customer-Churn.csv")

df = handle_missing(df)

# 🔥 IMPORTANT (same as train.py)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = encode_data(df)

# Split features & target
X, y = split_features_target(df, "Churn")

X, scaler = scale_features(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Load model
model = joblib.load("model/model.pkl")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))