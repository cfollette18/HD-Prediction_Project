import pandas as pd

# Load the dataset from CSV files
X = pd.read_csv('heart_disease_features.csv')
y = pd.read_csv('heart_disease_targets.csv')

# Check for missing values
print("Missing values in features:")
print(X.isnull().sum())

# Handle missing values (fill with median for numerical columns)
X = X.fillna(X.median())

# Check the distribution of the target variable
print("\nTarget variable distribution:")
print(y['num'].value_counts())  # Use 'num' instead of 'target'

# Save the preprocessed data to new CSV files
X.to_csv('preprocessed_features.csv', index=False)
y.to_csv('preprocessed_targets.csv', index=False)