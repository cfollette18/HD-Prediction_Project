import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the testing data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))