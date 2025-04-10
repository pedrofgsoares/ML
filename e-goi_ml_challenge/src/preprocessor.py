#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 23:12:41 2025

@author: pfsoares
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


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


# Initialize the Random Forest model
rf = RandomForestClassifier(bootstrap=True,
                            max_depth=20,
                            max_features='sqrt',
                            min_samples_leaf=1,
                            min_samples_split=5,
                            n_estimators = 200)

rf.fit(X, y)

# Save the model and encoder to disk using joblib
joblib.dump(rf, '../models/random_forest_model.pkl')
joblib.dump(encoder, '../models/encoder.pkl')
joblib.dump(scaler, '../models/scaler.pkl')