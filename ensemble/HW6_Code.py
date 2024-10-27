"""
Train two RF models and two Adaboost models on all data in dataset  with ‘outcome’ as label
and all other attributes as features, using 3 and 50 as the number of estimators.

Using a cross-validation method, calculate scores of all four models for 5 folds.

For each pair of RF and Adaboost models with the same number of estimators compare mean of scores.
"""
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz
import graphviz

org_df = pd.read_csv("diabetes.csv")

feat_df = org_df.loc[:, org_df.columns != 'Outcome']  # 提取所有的特征
label_df = org_df.loc[:, org_df.columns == 'Outcome'].squeeze()  # 转换为一维标签

train_x, test_x, train_y, test_y = train_test_split(feat_df, label_df, test_size=0.25)

n_estimators_1,n_estimators_2 =3,50
#k-fold validation
kf = KFold(n_splits=5)


# RF Models
rf_model_1 = RandomForestClassifier(n_estimators=n_estimators_1)
rf_model_2 = RandomForestClassifier(n_estimators=n_estimators_2)

# AdaBoost models
adaboost_model_1 = AdaBoostClassifier(n_estimators=n_estimators_1,algorithm='SAMME')
adaboost_model_2 = AdaBoostClassifier(n_estimators=n_estimators_2,algorithm='SAMME')



# Random Forest with 3 estimators
rf_scores_1 = cross_val_score(rf_model_1,train_x, train_y, cv=kf)
print(f"Random Forest with {n_estimators_1} estimators scores: {rf_scores_1}")
print(f"Mean score: {rf_scores_1.mean()}")

# Random Forest with 50 estimators
rf_scores_2 = cross_val_score(rf_model_2, train_x, train_y, cv=kf)
print(f"Random Forest with {n_estimators_2} estimators scores: {rf_scores_2}")
print(f"Mean score: {rf_scores_2.mean()}")

# AdaBoost with 3 estimators
ad_1 = cross_val_score(adaboost_model_1, train_x, train_y, cv=kf)
print(f"AdaBoost with {n_estimators_1} estimators scores: {ad_1} Mean score: {ad_1.mean()}")
# print(f"Mean score: {ad_1.mean()}")

# AdaBoost with 50 estimators
ad_2 = cross_val_score(adaboost_model_2, train_x, train_y, cv=kf)
print(f"AdaBoost with {n_estimators_2} estimators scores: {ad_2} Mean score: {ad_2.mean()}")
# print(f"Mean score: {ad_2.mean()}")
def accuracy(model):
    """
        Trains the given model, makes predictions on the test data, and calculates accuracy.

        Parameters:
        model: A machine learning model instance (e.g., RandomForestClassifier, AdaBoostClassifier, etc.)
            The model that will be trained using the training data and tested on the test data.

        Returns:
        accuracy: float
            The accuracy score of the model on the test data.
        """

    # Train the model using the training data (train_x, train_y)
    model.fit(train_x,train_y)
    # Make predictions on the test data (test_x) using the trained model
    test_pred_y = model.predict(test_x)
    # Calculate the accuracy of the model by comparing the predictions to the true labels (test_y)
    accuracy = accuracy_score(test_y, test_pred_y)
    return accuracy


##Training model
print("accuracy of rf_model_1 with 3 estimators scores:",accuracy(rf_model_1))
print("accuracy of rf_model_2 with 50 estimators scores:",accuracy(rf_model_2))
print("accuracy of adaboost_model_1 with 3 estimators scores:",accuracy(adaboost_model_1))
print("accuracy of adaboost_model_2 with 50 estimators scores:",accuracy(adaboost_model_2))
# visualize Trees
# for i in range(3):
#     stump = adaboost_model_1.estimators_[i]
#     dot_data = export_graphviz(stump,
#                             feature_names=train_x.columns,
#                             filled=True,
#                             class_names=['No', 'Yes'],
#                             proportion=True)
#     graph = graphviz.Source(dot_data, filename=f"stump_{i}.png", format="png")
#     graph.view()