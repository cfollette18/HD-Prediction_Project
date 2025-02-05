import pandas as pd
import joblib
import shap

# Load the training data
X_train = pd.read_csv('X_train.csv')

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, X_train)