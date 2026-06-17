import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json

# Data load karo
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop('customerID', axis=1, inplace=True)

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, drop_first=True)
df = df.fillna(False)
df = df.astype(int)

# X aur y
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Feature names save karo
with open('feature_names.json', 'w') as f:
    json.dump(X.columns.tolist(), f)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline banao — SMOTE + Scaler + Model
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train karo
pipeline.fit(X_train, y_train)

# Score dekho
score = pipeline.score(X_test, y_test)
print(f"Pipeline Accuracy: {score*100:.2f}%")

# Save karo
joblib.dump(pipeline, 'pipeline.pkl')
print("Pipeline saved! ✅")