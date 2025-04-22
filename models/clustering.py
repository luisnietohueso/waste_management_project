"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This script performs K-Means clustering to group bins based on their geographical locations.
It begins by importing the `KMeans` algorithm from `sklearn.cluster`.
The function `apply_clustering` takes a dataset and an optional number of clusters (default is 5).
It checks whether the required `lat` (latitude) and `long` (longitude) columns are present in the dataset.
If these columns are missing, the function raises a `KeyError` to stop execution and alert the user.

Once validated, it applies the K-Means algorithm using the specified number of clusters, with fixed parameters
for reproducibility (`n_init=10` and `random_state=42`). The resulting cluster labels are stored in a new column called `'Cluster'`.
A message is printed to confirm successful clustering and to display the unique cluster IDs found.

This method is useful for spatial grouping, allowing bins to be categorised based on their physical locations.
"""

from sklearn.cluster import KMeans

# Clustering
def apply_clustering(data ,n_clusters=5):
    """
    Apply K-Means clustering to group bins geographically.
    """
    if 'lat' not in data.columns or 'long' not in data.columns:
        raise KeyError(" ERROR: 'lat' and 'long' columns are missing! Clustering cannot proceed.")

    print(" Running apply_clustering()...")

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['lat', 'long']])

    # Debugging: Ensure 'Cluster' column is created
    print(" Clustering completed! Unique clusters:", data['Cluster'].unique())

    return data
