import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load dataset
df = pd.read_csv("Iris.csv")
print("✅ Dataset loaded successfully!\n")

# 2. View original column names
print("📌 Original Columns:", df.columns.tolist())

# 3. Clean column names
df.columns = df.columns.str.strip().str.lower()
print("✅ Cleaned Columns:", df.columns.tolist())

# 4. Drop 'id' column if present
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# 5. Encode species labels
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# 6. Split into features and target
X = df.drop('species', axis=1)
y = df['species']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 9. Predict
y_pred = model.predict(X_test)

# 10. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%\n")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# 11. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Greens', fmt='d',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Iris Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
