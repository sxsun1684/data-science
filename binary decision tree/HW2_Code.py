"""
Create a binary decision tree using Gini index impurity measure over the diabetes dataset to predict Outcome (as the label) by other attributes (as features).

Consider the following ranges of values for hyper-parameters:
max_depth = [3, 5]
min_sample_split= [5, 10]
min_samples_leaf= [3, 5]
min_impurity_decrease = [0.01, 0.001]
ccp_alpha = [0.001, 0.0001]

Spilt data into train, test, and validation (72%, 20%, 8%) and use validation data to select best hyper-parameter.
Calculate accuracy of the best Dtree on test data.
Extract three rules from the decision tree.
"""
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


max_depth = [3, 5]
min_sample_split= [5, 10]
min_samples_leaf= [3, 5]
min_impurity_decrease = [0.01, 0.001]
ccp_alpha = [0.001, 0.0001]
# random_state=32
# random_state=29
random_state=29

file_path = 'diabetes.csv'
diabetes_data = pd.read_csv(file_path)

# Define features to predict Resistance label
label_df = diabetes_data.loc[:, diabetes_data.columns == 'Outcome']
feat_df = diabetes_data.loc[:, diabetes_data.columns != 'Outcome']

# Split into train_val (80%) and test (20%)
train_feat, test_feat, train_label, test_label = train_test_split(feat_df, label_df, test_size=0.20, random_state=random_state)

# Further split train_val into train (72%) and validation (8%)
train_feat, val_feat, train_label, val_label = train_test_split(train_feat, train_label, test_size=0.10,
                                                                random_state=random_state)

# Create a hyperparameter grid
params = {
    'max_depth': max_depth,
    'min_samples_split': min_sample_split,
    'min_samples_leaf': min_samples_leaf,
    'min_impurity_decrease': min_impurity_decrease,
    'ccp_alpha': ccp_alpha
}

# Initialize the Decision Tree with Gini impurity
dtree = tree.DecisionTreeClassifier(criterion='gini', random_state=random_state)

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(dtree, params, cv=3, scoring='accuracy')
grid_search.fit(train_feat, train_label)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create the model using the best hyperparameters from grid search
treemodel = tree.DecisionTreeClassifier(criterion="gini",
                                        min_impurity_decrease=best_params['min_impurity_decrease'],
                                        max_depth=best_params['max_depth'],
                                        min_samples_leaf=best_params['min_samples_leaf'],
                                        ccp_alpha=best_params['ccp_alpha'],
                                        class_weight='balanced',
                                        random_state=random_state)

# Train the model on the training data
treemodel.fit(train_feat, train_label)

# Visualize the trained decision tree
plt.figure(figsize=(9, 9))
tree.plot_tree(treemodel, feature_names=train_feat.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.show()



# Accuracy Calculation
# Predict on test data
test_pred_label = treemodel.predict(test_feat)

# Confusion Matrix
confusion_mtx = confusion_matrix(test_label, test_pred_label)
ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=['No Diabetes', 'Diabetes']).plot()
plt.show()

# Accuracy = (True Positive + True Negative) / All
accuracy = accuracy_score(test_label, test_pred_label)
# Sensitivity (Recall) = True Positive / (True Positive + False Negative)
sensitivity = recall_score(test_label, test_pred_label)
# Specificity = True Negative / (True Negative + False Positive)
specificity = recall_score(test_label, test_pred_label, pos_label=0)


print("Accuracy=", accuracy)
print("Sensitivity=", sensitivity)
print("Specificity=", specificity)


"""
Accuracy= 0.8846153846153846
Sensitivity= 1.0
Specificity= 0.8
"""

print("Model Hyperparameters:")
print(treemodel.get_params())
'''
{'ccp_alpha': 0.001, 'class_weight': 'balanced', 'criterion': 'gini', 
'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.01, 
'min_samples_leaf': 5, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 
'monotonic_cst': None, 'random_state': 29, 'splitter': 'best'}
'''