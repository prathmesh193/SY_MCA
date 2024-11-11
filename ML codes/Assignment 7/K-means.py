import numpy as np
from scipy.spatial.distance import cdist

def k_means_clustering(points, initial_centroids, max_iterations=10):
    centroids = initial_centroids

    for i in range(max_iterations):
        #calculate the distance of each point to the centroids
        distances = cdist(points, centroids, metric = 'euclidean')

        #Assign points to nearest centroids
        clusters = np.argmin(distances, axis=1)

        #calculate new centroids
        length_centroids = len(centroids)
        new_centroids = np.array([points[clusters == j].mean(axis=0) for j in range(length_centroids)])

        #check for convergence(if centroids don't change)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

        return clusters, centroids

#Given data points
points = np.array([[0.1,0.6],[0.15,0.71],[0.08,0.9],[0.16,0.85],[0.2,0.3],[0.25,0.5],[0.24,0.1],[0.3,0.2]])

initial_centroids = np.array([[0.1,0.6],[0.3,0.2]])

clusters, final_centroids = k_means_clustering(points, initial_centroids)

print("Final Clusters : ",clusters)
print("Final Centroids:\n ",final_centroids)

P6_cluster = clusters[5]
population_m2 = np.sum(clusters == 1)
print(f"\nP6 belongs to Cluster : ",P6_cluster+1)

print("Population around m2 :" ,population_m2)
print(f"Updated Centroids : m1 = {final_centroids[0]}, m2 = {final_centroids[1]}")
