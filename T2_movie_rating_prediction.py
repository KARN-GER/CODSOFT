import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Load Dataset ===
file_name = 'IMDb Movies India.csv'

if not os.path.exists(file_name):
    print(f"❌ File not found: {file_name}")
    print(f"Current directory: {os.getcwd()}")
    exit()
else:
    print("✅ File found. Loading...")

# Fix UnicodeDecodeError using latin1
df = pd.read_csv(file_name, encoding='latin1')

# === Step 2: Data Overview ===
print("First few rows of the dataset:")
print(df.head())
print("\nColumns:", df.columns)

# === Step 3: Data Cleaning ===

# Drop rows where Rating is missing (target column)
df.dropna(subset=['Rating'], inplace=True)

# Convert 'Votes' to numeric
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

# Convert 'Year' to numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Year'] = df['Year'].fillna(df['Year'].median())

# === Step 4: Feature Engineering ===

# Encode Director (top 10 as labels, rest as "Other")
df['Director'] = df['Director'].fillna('Unknown')
top_directors = df['Director'].value_counts().nlargest(10).index
df['Director_clean'] = df['Director'].apply(lambda x: x if x in top_directors else 'Other')
le_director = LabelEncoder()
df['Director_enc'] = le_director.fit_transform(df['Director_clean'])

# Encode Genre using MultiLabelBinarizer
df['Genre'] = df['Genre'].fillna('')
df['Genre_list'] = df['Genre'].apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(mlb.fit_transform(df['Genre_list']), columns=mlb.classes_, index=df.index)
df = pd.concat([df, genre_dummies], axis=1)

# Log-transform 'Votes' to normalize
df['Votes_log'] = np.log1p(df['Votes'])

# === Step 5: Prepare Features and Target ===
features = ['Year', 'Votes_log', 'Director_enc'] + list(mlb.classes_)
X = df[features]
y = df['Rating']

# === Step 6: Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 7: Train Model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 8: Evaluate ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n🎯 Model Evaluation:")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")

# === Step 9: Feature Importance ===
importances = model.feature_importances_
importance_series = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importance_series[:10], y=importance_series.index[:10])
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# === Step 10: Sample Prediction ===
sample = X_test.iloc[0]
actual = y_test.iloc[0]
predicted = model.predict([sample])[0]

print("\n🎬 Sample Prediction:")
print("Actual Rating   :", actual)
print("Predicted Rating:", round(predicted, 2))
