"""
Train two SVM models, one with ‘linear’ kernel and another with ‘rbf’ kernel on dataset
( ‘label’ as label and all other attributes as features) using 75% of data as training and remaining data as test.
Calculate accuracy of both models and report the best accuracy.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data_path = 'non_linear.csv'
df = pd.read_csv(data_path)

# Separate features and label
feat_df = df.drop(columns=['label'])
label_df = df['label']

# Split the dataset into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(feat_df, label_df, test_size=0.25, random_state=42)

# Train the SVM model with linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(x_train, y_train)
y_pred_linear = svm_linear.predict(x_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)

# Train the SVM model with rbf kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(x_train, y_train)
y_pred_rbf = svm_rbf.predict(x_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

# Determine the best accuracy
best_accuracy = max(accuracy_linear, accuracy_rbf)
print(f'accuracy_linear:{accuracy_linear}, accuracy_rbf:{accuracy_rbf}, best_accuracy: {best_accuracy}')

# accuracy_linear:0.94, accuracy_rbf:1.0, best_accuracy: 1.0







