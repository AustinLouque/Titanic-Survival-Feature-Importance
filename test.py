import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the Titanic dataset
data = sns.load_dataset('titanic')  # Use Kaggle CSV if seaborn doesn't work

# Step 2: Basic exploration
print(data.head())
print(data.info())
print(data.isnull().sum())

# Step 3: Data preprocessing
# Fill missing values
data['age'] = data['age'].fillna(data['age'].median())
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])

# Drop unnecessary columns
data.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], axis=1, inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['sex', 'embarked'], drop_first=True)

# Step 4: Feature selection and target variable
X = data.drop('survived', axis=1)
y = data['survived']

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Evaluate the Logistic Regression model
y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Train and evaluate a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# Step 9: Visualize feature importance for Random Forest
importances = rf_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances)
plt.title('Feature Importances')
plt.xticks(rotation=45)
plt.show()
