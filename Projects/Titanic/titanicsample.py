"""
Titanic - Machine Learning from Disaster
Kaggle Competition Solution

Structure:
    Phase 1: Load & Explore
    Phase 2: Clean & Feature Engineer
    Phase 3: Train & Validate
    Phase 4: Predict & Submit
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# PHASE 1: LOAD & EXPLORE
# ============================================================

# TODO: Download train.csv and test.csv from Kaggle and place
#       them in this folder before running.

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Save test PassengerIds for the submission file later
test_ids = test["PassengerId"]

print("=== Train shape:", train.shape)
print("=== Test shape: ", test.shape)
print()
print("=== First 5 rows ===")
print(train.head())
print()
print("=== Column types & missing values ===")
print(train.info())
print()
print("=== Survival rate ===")
print(train["Survived"].value_counts(normalize=True))
print()
print("=== Numerical stats ===")
print(train.describe())
print()
print("=== Missing values ===")
print(train.isnull().sum())
print("---")
print(test.isnull().sum())


# ============================================================
# PHASE 2: CLEAN & FEATURE ENGINEER
# ============================================================

# Combine train + test so we clean them identically
# We'll split them back later
train["is_train"] = 1
test["is_train"] = 0
# test doesn't have 'Survived', add a placeholder
test["Survived"] = np.nan

df = pd.concat([train, test], ignore_index=True)

# --- 2a. Extract Title from Name ---
# Names look like: "Braund, Mr. Owen Harris"
# We grab the title between ", " and "."
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()

# Group rare titles
title_map = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
}
df["Title"] = df["Title"].map(lambda t: title_map.get(t, "Rare"))

print("\n=== Title distribution ===")
print(df["Title"].value_counts())

# --- 2b. Fill missing Age using median per Title ---
# This is smarter than a global median because
# "Master" (boys) are much younger than "Mr" (men).
for title in df["Title"].unique():
    median_age = df.loc[df["Title"] == title, "Age"].median()
    df.loc[(df["Title"] == title) & (df["Age"].isnull()), "Age"] = median_age

# --- 2c. Fill missing Embarked (only 2 missing) ---
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# --- 2d. Fill missing Fare (only 1 missing in test) ---
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# --- 2e. Create FamilySize ---
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# --- 2f. IsAlone flag ---
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# --- 2g. Fare bins (quartile-based) ---
df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)

# --- 2h. Age bins ---
df["AgeBin"] = pd.cut(
    df["Age"],
    bins=[0, 12, 20, 40, 60, 80],
    labels=[0, 1, 2, 3, 4],
).astype(int)

# --- 2i. Encode categorical columns ---
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])  # female=0, male=1

le_embarked = LabelEncoder()
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

le_title = LabelEncoder()
df["Title"] = le_title.fit_transform(df["Title"])

# --- 2j. Select features ---
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone",
    "FareBin",
    "AgeBin",
]

print("\n=== Missing values after cleaning ===")
print(df[features].isnull().sum())

# --- Split back into train / test ---
train_clean = df[df["is_train"] == 1].copy()
test_clean = df[df["is_train"] == 0].copy()

X = train_clean[features].values          # numpy array
y = train_clean["Survived"].values.astype(int)  # numpy array
X_test = test_clean[features].values      # numpy array

print(f"\nX shape: {X.shape}  y shape: {y.shape}  X_test shape: {X_test.shape}")


# ============================================================
# PHASE 3: TRAIN & VALIDATE
# ============================================================

# --- 3a. Random Forest with cross-validation ---
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")

print("\n=== Cross-Validation Results ===")
print(f"Fold scores: {np.round(scores, 4)}")
print(f"Mean accuracy: {scores.mean():.4f}  (+/- {scores.std():.4f})")

# --- 3b. Train on full training set ---
rf.fit(X, y)

# Feature importances (helpful to understand what matters)
importances = rf.feature_importances_
for name, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {name:15s} {imp:.4f} {bar}")


# ============================================================
# PHASE 4: PREDICT & SUBMIT
# ============================================================

predictions = rf.predict(X_test).astype(int)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": predictions,
})

submission.to_csv("submission.csv", index=False)

print(f"\n=== Submission file created ===")
print(f"Shape: {submission.shape}")
print(f"Survived distribution:\n{submission['Survived'].value_counts()}")
print(f"\nFile saved: submission.csv")
print("Upload this to: https://www.kaggle.com/competitions/titanic/submit")
