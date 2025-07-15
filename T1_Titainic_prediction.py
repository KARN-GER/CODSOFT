# Titanic Survival Prediction Project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Loading the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 2. Dropping the unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 3. Handling missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 4. Encoding categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 5. Defining features and target

X = df[features]
y = df['Survived']

# 6. Splitting into training and testing sets
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Training a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Validation Accuracy:", round(accuracy * 100, 2), "%")

# 9. Sample prediction | Sample input
sample = pd.DataFrame([{
    'Pclass': 3,
    'Sex': 0,
    'Age': 25,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 7.25,
    'Embarked': 0
}])
prediction = model.predict(sample)
print("Prediction for sample passenger:", "Survived" if prediction[0] == 1 else "Did not survive")
