"""
1. Use world_ds dataset containing socio-economic variables (features) of different countries and corresponding 
development_status as outcome (label).
2.Employ forward wrapper method to select best three features from the dataset.
3.Use a PCA model to create 3  new components from existing features.
4.Explain each PC (new features) based on the correlations with old features.
5.Use a LDA model to create 2 new components from existing features.
6.Compare the accuracy of a KNN classifier on new or selected features resulting by forward, PCA and LDA
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
world_ds = pd.read_csv("world_ds.csv")
# Define features and label
X = world_ds.drop(columns=['Country', 'development_status'])
Y = world_ds['development_status']

# Standardize the features for consistency
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# Initialize KNN model for forward selection
knn = KNeighborsClassifier()

# Apply forward selection to choose the best 3 features
sfs = SFS(knn, k_features='best', forward=True,scoring='accuracy',cv=5)
sfs.fit(X_train, Y_train)

selected_features_idx = list(sfs.k_feature_idx_)
selected_features = list(sfs.k_feature_names_)

# print("Selected feature indices:", selected_features_idx)
# print("Selected feature names:", selected_features)

# Apply PCA to reduce to 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame to display the correlation between the principal components and the original features
pca_components = pd.DataFrame(pca.components_, columns=X.columns, index=['PC1', 'PC2', 'PC3'])
print(pca_components.T)

# Apply LDA to reduce features to 2 components
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, Y)

# Moving to the next step: comparing accuracy using KNN on Forward Selection, PCA, and LDA features

# 1. Accuracy using Forward Selected Features
X_train_fs, X_test_fs = X_train[:, selected_features_idx], X_test[:, selected_features_idx]
knn.fit(X_train_fs, Y_train)
Y_pred_fs = knn.predict(X_test_fs)
accuracy_fs = accuracy_score(Y_test, Y_pred_fs)

# 2. Accuracy using PCA Components
X_train_pca, X_test_pca = X_pca[:X_train.shape[0]], X_pca[X_train.shape[0]:]
knn.fit(X_train_pca, Y_train)
Y_pred_pca = knn.predict(X_test_pca)
accuracy_pca = accuracy_score(Y_test, Y_pred_pca)

# 3. Accuracy using LDA Components
X_train_lda, X_test_lda = X_lda[:X_train.shape[0]], X_lda[X_train.shape[0]:]
knn.fit(X_train_lda, Y_train)
Y_pred_lda = knn.predict(X_test_lda)
accuracy_lda = accuracy_score(Y_test, Y_pred_lda)

# Compile results for comparison
accuracy = pd.DataFrame({
    'Method': ['Forward', 'PCA', 'LDA'],
    'Accuracy': [accuracy_fs, accuracy_pca, accuracy_lda]
})

print(accuracy)