import numpy as np
from scipy.spatial.distance import cdist

def k_medoids(points, initial_medoids, max_iterations=100):
    # Number of clusters
    n_clusters = len(initial_medoids)
    # Initialize medoids
    medoids = initial_medoids
    # Initialize clusters
    clusters = np.zeros(len(points), dtype=int)

    for iteration in range(max_iterations):
        # Calculate distances from points to medoids
        distances = cdist(points, points[medoids], metric='euclidean')
        # Assign each point to the nearest medoid
        new_clusters = np.argmin(distances, axis=1)
        
        # Update medoids
        new_medoids = np.copy(medoids)
        for i in range(n_clusters):
            # Find all points in the current cluster
            cluster_points = points[new_clusters == i]
            if len(cluster_points) == 0:
                continue
            # Calculate the point within the cluster that minimizes the total distance to all other points in the cluster
            distance_matrix = cdist(cluster_points, cluster_points, metric='euclidean')
            medoid_index = np.argmin(distance_matrix.sum(axis=1))
            new_medoids[i] = np.where((points == cluster_points[medoid_index]).all(axis=1))[0][0]
        
        # Check for convergence
        if np.all(new_medoids == medoids):
            break
        medoids = new_medoids
        clusters = new_clusters

    return clusters, medoids

# Given data points
points = np.array([
    [0.1, 0.6],
    [0.15, 0.71],
    [0.08, 0.9],
    [0.16, 0.85],
    [0.2, 0.3],
    [0.25, 0.5],
    [0.24, 0.1],
    [0.3, 0.2]
])

# Initial medoids indices
initial_medoids = [0, 7]  # P1 and P8

# Perform K-Medoids clustering
clusters, final_medoids = k_medoids(points, initial_medoids)

# Output results
P6_cluster = clusters[5]
population_m2 = np.sum(clusters == 1)

print("Final Clusters:", clusters)
print("Final Medoids (indices):", final_medoids)
print(f"P6 belongs to Cluster: {P6_cluster + 1}")
print(f"Population around m2: {population_m2}")
print(f"Updated Medoids: {points[final_medoids]}")
