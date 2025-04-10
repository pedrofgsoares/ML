#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 23:12:41 2025

@author: pfsoares
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier



# Load the data
url = 'https://drive.google.com/uc?export=download&id=1TLTuZ-Zc0Mcl16aYLdHqVKHneXmCdR5t'
data = pd.read_csv(url)


# One-hot encoding of categorical columns
categorical_cols = data.select_dtypes(include='object').columns.tolist()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_cats = encoder.fit_transform(data[categorical_cols])
encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cat_cols)



# Normalization of numerical columns
numerical_cols = data.select_dtypes(include='int64').columns.tolist()
numerical_cols.remove('LABEL')
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(data[numerical_cols])
scaled_df = pd.DataFrame(scaled_nums, columns=numerical_cols)

# Dataset final
processed_df = pd.concat([scaled_df, encoded_df], axis=1)
processed_df['LABEL'] = data['LABEL'].values

# Creating features and target
X = processed_df.drop(columns='LABEL')
y = processed_df['LABEL'] - 1 # this is needed to use the XGBoost




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost':XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'Neural Network':MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
}

# Train and evaluate each model
model_results = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for AUC
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    
    model_results[model_name] = {
        'Accuracy': accuracy,
        'AUC-ROC': auc,
        'Classification Report': report
    }

# Display the results
# Prepare a DataFrame for better visualization
model_comparison = pd.DataFrame({
    'Model': model_results.keys(),
    'Accuracy': [results['Accuracy'] for results in model_results.values()],
    'AUC-ROC': [results['AUC-ROC'] for results in model_results.values()]
})


print(model_comparison)

# The best model is Random Forest