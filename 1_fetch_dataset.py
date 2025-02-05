from ucimlrepo import fetch_ucirepo

# Fetch the Heart Disease dataset (ID: 45)
heart_disease = fetch_ucirepo(id=45)

# Extract features (X) and target (y)
X = heart_disease.data.features
y = heart_disease.data.targets

# Save the dataset to CSV files for later use
X.to_csv('heart_disease_features.csv', index=False)
y.to_csv('heart_disease_targets.csv', index=False)

# Print dataset information
print("Dataset Metadata:")
print(heart_disease.metadata)
print("\nDataset Variables:")
print(heart_disease.variables)