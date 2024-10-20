"""
Use suitable data preprocessing methods including binning for numeric variables on  
dataset to prepare data for association rule mining.

Extract rules using all combination of the following hyper-parameters:

min_sup = [0.05, 0.1, 0.4]
min_conf= [0.70, 0.85, 0.95]
min_lift= [1.1, 1.5, 4]
Based on results, select a set of parameters which can extract 20 to 50 rules.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules
import itertools

org_df = pd.read_csv('amr_horse_ds.csv')
org_df= pd.get_dummies(org_df.loc[:,org_df.columns!='Age'])

#Extract Association Rules
min_sup_values = [0.05, 0.1, 0.4]
min_conf_values = [0.70, 0.85, 0.95]
min_lift_values = [1.1, 1.5, 4]
parameter_combinations = list(itertools.product(min_sup_values, min_conf_values, min_lift_values))

results = []

for min_sup, min_conf, min_lift in parameter_combinations:
    frequent_patterns = fpgrowth(org_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(frequent_patterns, metric='confidence', min_threshold=min_conf)
    filtered_rules = rules[rules['lift'] >= min_lift]
    results.append({
        'min_sup': min_sup,
        'min_conf': min_conf,
        'min_lift': min_lift,
        'num_rules': len(filtered_rules),
        'rules': filtered_rules
    })

#Based on results, select a set of parameters which can extract 20 to 50 rules.
selected_results = [result for result in results if 20 <= result['num_rules'] <= 50]

def results():
    if selected_results:
        selected_result = selected_results[0]
        print(
            f"min_sup: {selected_result['min_sup']}, min_conf: {selected_result['min_conf']}, min_lift: {selected_result['min_lift']}, num_rules: {selected_result['num_rules']}")

        selected_rules = selected_result['rules']
        selected_rules.to_csv('arules.csv', index=False)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(selected_rules['support'], selected_rules['confidence'], selected_rules['lift'], marker='o')
        ax.set_xlabel('support')
        ax.set_ylabel('confidence')
        ax.set_zlabel('lift')
        plt.show()
    else:
        return

results()

