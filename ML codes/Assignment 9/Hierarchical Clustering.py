import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Load the dataset
shopping_data = pd.read_csv('Shopping_Trends.csv')

# Select relevant numerical features for clustering
features = shopping_data[["Purchase Amount (USD)", "Review Rating", 
                          "Previous Purchases", "Delivery Time", 
                          "Number of Items Purchased"]]

# Use a random subset of 30 customers to simplify the dendrogram
subset = features.sample(n=30, random_state=42)

# Standardize the data
scaler = StandardScaler()
subset_scaled = scaler.fit_transform(subset)

# Perform hierarchical clustering using the 'ward' method
linked = linkage(subset_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked,
           orientation='top',
           distance_sort='ascending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Subset of 30 Customers)')
plt.xlabel('Customer Index')
plt.ylabel('Distance')
plt.show()
