# model/train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 Load dataset
df = pd.read_csv("data/employee_data.csv", skipinitialspace=True)

# 🔹 Clean data
df = df.dropna(subset=['education'])
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)
df.drop(columns=['fnlwgt', 'native-country', 'education'], inplace=True)
df.drop_duplicates(inplace=True)

# 🔹 Define features and target
X = df.drop(columns=['income'])
y = df['income']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# 🔹 Preprocessing and model
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train and Evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 🔹 Save the trained model
joblib.dump(model, "model/rf_model.pkl")
print("\n💾 Model saved to model/rf_model.pkl")
import pickle
import os

# ✅ Save trained model
os.makedirs('model', exist_ok=True)
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
