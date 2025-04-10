#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 23:12:41 2025

@author: pfsoares
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier



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
y = processed_df['LABEL']


#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, accuracy_score
#from xgboost import XGBClassifier
#from sklearn.neural_network import MLPClassifier



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the models
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)

"""
Best Parameters: {'bootstrap': True, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
Best Cross-Validation Accuracy: 0.76375
"""