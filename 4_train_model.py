import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the training data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train.values.ravel())

# Save the trained model to a file
joblib.dump(model, 'heart_disease_model.pkl')
print("Model trained and saved to 'heart_disease_model.pkl'.")