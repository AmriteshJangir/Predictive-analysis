# student_data_cleaning.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# 1. Load Dataset
# -------------------------------
# Update the path if needed
df = pd.read_csv("student_data.csv")

print("Initial Shape:", df.shape)
print(df.head())
print(df.info())

# -------------------------------
# 2. Handle Missing Values
# -------------------------------

# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

# Fill numeric missing values with median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical missing values with mode
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# -------------------------------
# 3. Remove Duplicates
# -------------------------------
df.drop_duplicates(inplace=True)

# -------------------------------
# 4. Handle Outliers (IQR Method)
# -------------------------------
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# -------------------------------
# 5. Encode Categorical Variables
# -------------------------------
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------------------------------
# 6. Feature Scaling
# -------------------------------
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# -------------------------------
# 7. Final Check
# -------------------------------
print("Final Shape:", df.shape)
print(df.isnull().sum())

# -------------------------------
# 8. Save Cleaned Data
# -------------------------------
df.to_csv("student_data_cleaned.csv", index=False)

print("Data cleaning and preprocessing completed successfully.")
