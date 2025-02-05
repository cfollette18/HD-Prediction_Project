import pandas as pd
from sklearn.model_selection import train_test_split

# Load the preprocessed data
X = pd.read_csv('preprocessed_features.csv')
y = pd.read_csv('preprocessed_targets.csv')

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split data to CSV files
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Training and testing sets saved to CSV files.")