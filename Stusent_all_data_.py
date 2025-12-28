#===========================================#
# University data taken first for cleaning analysing and maintaining then manipulation task

# ============================================
# University Student Data Cleaning & Processing
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
df = pd.read_csv("student_data.csv")

print("Initial Shape:", df.shape)
print(df.head())
print(df.info())

# --------------------------------------------
# 2. Standardize Column Names
# --------------------------------------------

df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

# --------------------------------------------
# 3. Expected University Parameters
# --------------------------------------------
expected_columns = [
    "student_id",
    "gender",
    "age",
    "department",
    "year_of_study",
    "attendance_hours",
    "total_class_hours",
    "attendance_percentage",
    "internal_marks",
    "external_marks",
    "cgpa",
    "assignments_submitted",
    "total_assignments",
    "quiz_score",
    "mid_sem_score",
    "final_exam_score",
    "backlogs",
    "study_hours_per_week",
    "extra_curricular_participation",
    "library_visits_per_month",
    "internet_usage_hours",
    "disciplinary_actions",
    "placement_status"
]

# Add missing columns (if dataset incomplete)
for col in expected_columns:
    if col not in df.columns:
        df[col] = np.nan

# --------------------------------------------
# 4. Attendance Feature Engineering
# --------------------------------------------
df["attendance_percentage"] = (
    df["attendance_hours"] / df["total_class_hours"]
) * 100

df["attendance_percentage"] = df["attendance_percentage"].clip(0, 100)

# Attendance Risk Flag
df["low_attendance_flag"] = np.where(df["attendance_percentage"] < 75, 1, 0)

# --------------------------------------------
# 5. Academic Performance Features
# --------------------------------------------
df["assignment_completion_rate"] = (
    df["assignments_submitted"] / df["total_assignments"]
) * 100

df["average_exam_score"] = df[
    ["quiz_score", "mid_sem_score", "final_exam_score"]
].mean(axis=1)

df["academic_risk_flag"] = np.where(
    (df["cgpa"] < 6.0) | (df["backlogs"] > 0), 1, 0
)

# --------------------------------------------
# 6. Behavioral & Engagement Metrics
# --------------------------------------------
df["engagement_score"] = (
    (df["study_hours_per_week"] * 0.4) +
    (df["library_visits_per_month"] * 0.3) +
    (df["extra_curricular_participation"] * 0.3)
)

df["discipline_risk_flag"] = np.where(
    df["disciplinary_actions"] > 0, 1, 0
)

# --------------------------------------------
# 7. Missing Value Treatment
# --------------------------------------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# --------------------------------------------
# 8. Duplicate Removal
# --------------------------------------------
df.drop_duplicates(subset=["student_id"], inplace=True)

# --------------------------------------------
# 9. Outlier Treatment (IQR)
# --------------------------------------------
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

# --------------------------------------------
# 10. Encode Categorical Variables
# --------------------------------------------
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --------------------------------------------
# 11. Feature Scaling
# --------------------------------------------
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# --------------------------------------------
# 12. Final University Risk Score
# --------------------------------------------
df["overall_risk_score"] = (
    df["low_attendance_flag"] * 0.3 +
    df["academic_risk_flag"] * 0.4 +
    df["discipline_risk_flag"] * 0.3
)

# --------------------------------------------
# 13. Train-Test Split (Optional)
# --------------------------------------------
target = "placement_status"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------
# 14. Save Outputs
# --------------------------------------------
df.to_csv("student_data_cleaned_full.csv", index=False)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("University-level student data preprocessing completed successfully.")
print("Final Shape:", df.shape)
