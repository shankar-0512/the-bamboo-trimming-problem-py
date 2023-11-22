import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set the random seed to ensure reproducibility of results
np.random.seed(45)

def silhouette_score(X, cluster_assignments, centroids):
        num_clusters = len(np.unique(cluster_assignments))
        silhouette_scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # Compute mean distance to all other points in the same cluster
            a = np.mean(np.linalg.norm(X[cluster_assignments == cluster_assignments[i]] - X[i], axis=1))
            # Compute mean distance to all points in the nearest other cluster
            b = np.inf
            for j in range(num_clusters):
                if j != cluster_assignments[i] and np.sum(cluster_assignments == j) > 0:
                    b = min(b, np.mean(np.linalg.norm(X[cluster_assignments == j] - X[i], axis=1)))

            if a == 0 or b == 0 or np.isnan(a) or np.isnan(b) or np.isinf(a) or np.isinf(b):
                silhouette_scores[i] = 0
            else:
                silhouette_scores[i] = (b - a) / max(a, b)

        # Compute overall silhouette score
        silhouette_score = np.mean(silhouette_scores)
        return silhouette_score


class K_means_clustering:
    
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        
    def cluster(self, X, k):
        # Initialize centroids randomly
        centroids = X[np.random.choice(X.shape[0], k, replace=False), :]
        
        for iteration in range(self.max_iterations):
            # Assign data points to clusters
            distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
            cluster_assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            for j in range(k):
                cluster_points = X[cluster_assignments == j]
                if cluster_points.shape[0] > 0:
                    centroids[j] = np.mean(cluster_points, axis=0)
        
        return cluster_assignments, centroids

    
class K_means_pp_clustering:
    
    def __init__(self, max_iterations):
        
        # Initializing class attributes
        self.max_iterations = max_iterations
        self.silhouette_score = silhouette_score
    
    def cluster(self, X, k):
        # Select the first center uniformly at random
        centroids = [X[np.random.choice(X.shape[0])]]

        # Select the remaining k-1 centers using k-means++ algorithm
        for i in range(1, k):
            # Calculate distances to the nearest center for each point
            
            # Define an empty array to hold the minimum distances
            distances = np.empty(len(X))

            # Loop over each data point
            for i, x in enumerate(X):
                # Calculate the distance between the data point and each centroid
                distances_to_centroids = [np.linalg.norm(x - c)**2 for c in centroids]
                # Find the minimum distance among all centroids
                min_distance = min(distances_to_centroids)
                # Add the minimum distance to the distances array
                distances[i] = min_distance

            # Calculate probability weights for each point
            weights = distances / distances.sum()
            # Select next center randomly based on probability weights
            centroids.append(X[np.random.choice(X.shape[0], p=weights)])

        # Initialize cluster assignments
        cluster_assignments = np.zeros(X.shape[0], dtype=int)

        # Iterate until convergence or max iterations reached
        for iteration in range(self.max_iterations):
            # Assign data points to clusters
            for i in range(X.shape[0]):
                distances = np.array([np.linalg.norm(X[i] - c)**2 for c in centroids])
                cluster_assignments[i] = np.argmin(distances)

            # Update cluster centers
            for j in range(k):
                centroids[j] = np.mean(X[cluster_assignments == j], axis=0)

        silhouette_score = self.silhouette_score(X, cluster_assignments, centroids)
        
        return silhouette_score
    
class Bisecting_k_means_clustering:
    
    def __init__(self, max_iterations):
        
        # Initializing class attributes
        self.max_iterations = max_iterations
        self.silhouette_score = silhouette_score
    
    def cluster(self, X, k):
        
        # Initialize with all data points in one cluster
        cluster_assignments = np.zeros(X.shape[0], dtype=int)
        centroids = np.mean(X, axis=0, keepdims=True)

        # Perform bisecting k-means until the desired number of clusters is reached
        while centroids.shape[0] < k:
            # Find the cluster with the largest sum of squared distances to its centroid
            max_cluster = np.argmax([np.sum((X[cluster_assignments == i] - centroids[i])**2)
                                     for i in range(centroids.shape[0])])

            # Split the largest cluster into two
            split_indices = np.where(cluster_assignments == max_cluster)[0]
            split_data = X[split_indices]
            k_means_clustering = K_means_clustering(max_iterations = 100)
            split_assignments, split_centroids = k_means_clustering.cluster(split_data, 2)

            # Update the cluster assignments and centroids
            new_assignments = np.zeros(X.shape[0], dtype=int)
            new_assignments[split_indices[split_assignments == 0]] = max_cluster
            new_assignments[split_indices[split_assignments == 1]] = centroids.shape[0]
            cluster_assignments[cluster_assignments == max_cluster] = new_assignments[cluster_assignments == max_cluster]
            centroids[max_cluster] = split_centroids[0]
            centroids = np.vstack([centroids, split_centroids[1]])

        silhouette_score = self.silhouette_score(X, cluster_assignments, centroids)
        
        return silhouette_score


dataset = pd.read_csv("/Users/shankarnarayanan/Desktop/Important/SEMESTER 2/Data Mining/CA2/dataset", delimiter=" ", header=None)
dataset = (np.array(dataset)[:, 1:]).astype(np.float64)

k_means_silhouette = np.zeros(9)
k_means_pp_silhouette = np.zeros(9)
bisecting_k_means_silhouette = np.zeros(9)


for i in range(1, 10):
    
    clustering = K_means_clustering(max_iterations = 100)
    cluster_assignments, centroids = clustering.cluster(dataset, i)
    k_means_silhouette[i-1] = silhouette_score(dataset, cluster_assignments, centroids)
    
for i in range(1, 10):
    
    clustering = K_means_pp_clustering(max_iterations = 100)
    k_means_pp_silhouette[i-1] = clustering.cluster(dataset, i)

for i in range(1, 10):
    
    clustering = Bisecting_k_means_clustering(max_iterations = 100)
    bisecting_k_means_silhouette[i-1] = clustering.cluster(dataset, i)
    
print(k_means_silhouette)
print(k_means_pp_silhouette)
print(bisecting_k_means_silhouette)
    
# define the figure with one subplot
fig, ax = plt.subplots(3, 1, figsize=(10, 18))

# plot silhouette scores
ax[0].set_title("K-means", fontsize=20)
ax[0].set_xlabel("k", fontsize=16)
ax[0].set_ylabel("Silhouette Coefficient", fontsize=14)
ax[0].tick_params(axis="both", labelsize=14)
ax[0].grid(linestyle='--', alpha=0.7)
ax[0].plot(range(1, 10), k_means_silhouette, lw=2, color='purple')

# add background color
fig.patch.set_facecolor('#F5F5F5')
ax[0].set_facecolor('#FFFFFF')

# plot silhouette scores
ax[1].set_title("K-means ++", fontsize=20)
ax[1].set_xlabel("k", fontsize=16)
ax[1].set_ylabel("Silhouette Coefficient", fontsize=14)
ax[1].tick_params(axis="both", labelsize=14)
ax[1].grid(linestyle='--', alpha=0.7)
ax[1].plot(range(1, 10), k_means_pp_silhouette, lw=2, color='blue')

# add background color
fig.patch.set_facecolor('#F5F5F5')
ax[1].set_facecolor('#FFFFFF')

# plot silhouette scores
ax[2].set_title("Bisecting k-means", fontsize=20)
ax[2].set_xlabel("k", fontsize=16)
ax[2].set_ylabel("Silhouette Coefficient", fontsize=14)
ax[2].tick_params(axis="both", labelsize=14)
ax[2].grid(linestyle='--', alpha=0.7)
ax[2].plot(range(1, 10), bisecting_k_means_silhouette, lw=2, color='orange')

# add background color
fig.patch.set_facecolor('#F5F5F5')
ax[2].set_facecolor('#FFFFFF')