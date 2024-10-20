import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from kmodes.kmodes import KModes
from pyclustering.cluster.kmedians import kmedians
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
'''
Find an optimal number of clusters for k-means over the marketing dataset
Use the optimal cluster number in K-means to cluster the marketing data.
Visualize the relations between income and spending for all resulting clusters using a scatter plot.
Visualize the relations between income and age for all resulting clusters using another scatter plot.
Try to find names for different clusters based on these visualizations.
'''

#Input Dataset
data = pd.read_csv("market_ds.csv")

# Assume that the data set has three columns of 'Income', 'Spending', and 'age'
features = data[['Age','Income', 'Spending']]
train_feat = (features - features.mean()) / features.std()
print(data.columns)

# The elbow method is used to determine the optimal number of clusters
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i).fit(train_feat)
    inertias.append(kmeans.inertia_)

# draw elbow diagram
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Inertia)')
plt.show()

#Assuming that the optimal cluster number is 4 according to the elbow method, KMeans clustering is carried out
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters)
data['Cluster'] = kmeans.fit_predict(train_feat)

# Visualize the clustering relationship between income and expenditure
plt.scatter(data['Income'], data['Spending'], c=data['Cluster'], cmap='viridis')
plt.title('Clusters based on Income and Spending')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.show()

# Visualize the clustering relationship between age and income
plt.scatter(data['Age'], data['Income'], c=data['Cluster'], cmap='viridis')
plt.title('Clusters based on Age and Income')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

cluster_names = {
    0: 'High income, high spending group',
    1: 'Low income, low spending group',
    2: 'Low income, high spending group',
    3: 'High income, low spending group'
}

# Add the corresponding name of each cluster to the data bo
data['Cluster_Name'] = data['Cluster'].map(cluster_names)

data.to_csv('HW3_clusters_name.csv', index=False)
print(data.head(200))
print(f"Total number of rows: {len(data)}")