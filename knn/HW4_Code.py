import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

"""
Train a multiple linear regression on Train Dataset  with ‘BloodPressure’ as outcome (dependent variable) 
and all other attributes as features (Independent variables).
"""

#Input Dateset
org_df = pd.read_csv("hw4_train.csv")

# Selection feature variable
X_selected = org_df[['Pregnancies', 'Glucose', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']]
y = org_df['BloodPressure']

# The data set is divided into the training set and the test set, and the test set accounts for 25%
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.25, random_state=42)

# Initialize the linear regression model
model_selected = LinearRegression()

# training model
model_selected.fit(X_train_selected, y_train)

# Output slope (coefficient) and intercept
print('Slope (Coefficients) =', model_selected.coef_)
print('Intercept =', model_selected.intercept_)

# Ensure that the characteristics of the test dataset are consistent with those of the training set
X_test_selected = X_test_selected[X_train_selected.columns]

# Make predictions with test dataset
predicted_blood_pressure = model_selected.predict(X_test_selected)

# plt.scatter(y_test, predicted_blood_pressure)
# plt.xlabel("Real BP")
# plt.ylabel("Predicted BP")
# plt.title("Real vs Predicted BP")
# plt.show()

'''
BloodPressure=38.42339234544128
+0.4834107*Pregnancies+0.04490215*Glucose+0.0134288*SkinThickness-0.01333235*Insulin+0.64051705*BMI
-1.27665774*DiabetesPedigreeFunction+0.23554899*Age−1.73213794*Outcome
'''

'''
Using the regression model, predict and set values of ‘BloodPressure’ in Test Dataset.
'''
test_data = pd.read_csv('hw4_test.csv')

# Select the same feature columns as the training set
test_features = test_data[['Pregnancies', 'Glucose', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']]
# Use models to make predictions
predicted_bp = model_selected.predict(test_features)
print(predicted_bp)

# Add the prediction to the test set
test_data['PredictedBloodPressure'] = predicted_bp

# Export the test set with the predicted values as a new CSV file
test_data.to_csv('hw4_test_predictions.csv', index=False)





print("-------------------------")

"""
Train 19 KNN models with k from 1 to 19 on Train Dataset  dataset with ‘Outcome’ as label and all other attributes 
as features.
"""

# Define a decorator to train and evaluate the KNN model
def knn_model_decorator(func):
    def wrapper(*args, **kwargs):
        # Initialize KNN model with the given number of neighbors
        knn_model = KNeighborsClassifier(n_neighbors=kwargs['n_neighbors'])

        # Train the model using training data
        knn_model.fit(kwargs['X_train'], kwargs['y_train'])

        # Predict outcomes on the test data
        y_pred = knn_model.predict(kwargs['X_test'])

        # Calculate accuracy of the model
        accuracy = accuracy_score(kwargs['y_test'], y_pred)

        # Pass accuracy and other arguments to the original function
        return func(accuracy, *args, **kwargs)

    return wrapper


# Define the features and target variable
X_knn = org_df.drop(columns=['Outcome'])
y_knn = org_df['Outcome']

# Split the data into training and test sets
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.25, random_state=42)

# List of k values to evaluate
k_values = list(range(1, 20))
accuracy_scores = []

@knn_model_decorator
def calculate_accuracy(accuracy, n_neighbors, *args, **kwargs):
    """
    Appends the accuracy of the KNN model for the given k (n_neighbors)
    to the accuracy_scores list.

    Parameters:
    - accuracy: Model accuracy calculated by the decorator.
    - n_neighbors: Number of neighbors (k) for the KNN model.
    """
    accuracy_scores.append(accuracy)

# Loop through k values to calculate accuracy
for k in k_values:
    calculate_accuracy(n_neighbors=k, X_train=X_train_knn, y_train=y_train_knn, X_test=X_test_knn, y_test=y_test_knn)

# Find the best k value (the one with the highest accuracy)
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print("-------------------------")
print(f"The best k is: {best_k}")

@knn_model_decorator
def conclusion(accuracy, n_neighbors, *args, **kwargs):
    """
    Prints the accuracy of the KNN model for the best k value.

    Parameters:
    - accuracy: Model accuracy calculated by the decorator.
    - n_neighbors: The best k value found.
    """
    print(f"Accuracy for k={n_neighbors}: {accuracy}")

# Call the conclusion function with the best k value
conclusion(n_neighbors=best_k, X_train=X_train_knn, y_train=y_train_knn, X_test=X_test_knn, y_test=y_test_knn)

"""
The best k is: 4
Accuracy for k=4: 0.8441558441558441
"""







