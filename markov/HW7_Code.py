import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from pydtmc import *
from sklearn.naive_bayes import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score


org_df = pd.read_csv('amr_ds.csv')

# label_df = org_df.loc[:, org_df.columns != 'Not_MDR']
# feat_df = org_df['Not_MDR']
label_df = org_df[['Ampicillin', 'Penicillin']]
feat_df = org_df['Not_MDR']

x_train, x_test, y_train, y_test = train_test_split(label_df, feat_df, test_size=0.25, random_state=42)
#Create Naive Bayes Model
model = BernoulliNB()
model.fit(x_train, y_train)

#Accuracy of Model
print("Test accuracy:  ", model.score(x_test,y_test))
# Test accuracy:   0.945054945054945

amp_pen = len(org_df[(org_df['Ampicillin'] == 1) & (org_df['Penicillin'] == 1)])
amp_nmdr = len(org_df[(org_df['Ampicillin'] == 1) & (org_df['Not_MDR'] == 1)])
pen_nmdr = len(org_df[(org_df['Penicillin'] == 1) & (org_df['Not_MDR'] == 1)])


transition_matrix = [
    [0, amp_pen / (amp_nmdr + amp_pen), amp_nmdr / (amp_nmdr + amp_pen)],
    [amp_pen / (pen_nmdr + amp_pen), 0, pen_nmdr / (pen_nmdr + amp_pen)],
    [amp_nmdr / (amp_nmdr + pen_nmdr), pen_nmdr / (amp_nmdr + pen_nmdr), 0]
]
transition_matrix = np.array(transition_matrix)
# print(transition_matrix)
# [[0.         0.94690265 0.05309735]
#  [0.66049383 0.         0.33950617]
#  [0.09836066 0.90163934 0.        ]]
states = ['Ampicillin', 'Penicillin','Not_MDR']
# Create Markov Chain
mc = MarkovChain(transition_matrix, states)
# print(mc)

# Show stationary state
print(mc.steady_states)
#[array([0.33630952, 0.48214286, 0.18154762])]

observation_matrix = np.array([
    [0.4, 0.6],  # Ampicillin: [No Infection, Infection]
    [0.3, 0.7],  # Penicillin: [No Infection, Infection]
    [0.8, 0.2]   # Not_MDR: [No Infection, Infection]
])

observations = [
    "Infection after surgery",
    "No infection after surgery",
    "Infection after surgery"
]

sequence = []

for obs in observations:
    # Determine the column index based on the observation: 0 for "No Infection", 1 for "Infection"
    column_index = 1 if "Infection" in obs else 0
    # Extract probabilities for all states in the selected column
    state_probs = observation_matrix[:, column_index]
    # Find the index of the state with the highest probability
    most_probable_state_index = np.argmax(state_probs)
    # Retrieve the name of the most probable state
    most_probable_state = states[most_probable_state_index]
    # Add the most probable state to the sequence
    sequence.append(most_probable_state)

print("Observations:", observations)
print("Most probable sequence of states:", sequence)