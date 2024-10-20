"""
Train two RF models and two Adaboost models on all data in dataset  with ‘outcome’ as label and all other attributes as features, using 3 and 50 as the number of estimators.
Using a cross-validation method, calculate scores of all four models for 5 folds.
For each pair of RF and Adaboost models with the same number of estimators compare mean of scores.
"""