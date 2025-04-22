# test_clustering.py
"""
Author: Luis Nieto Hueso  
Date: 28/03/2025  
Description:  
This test script verifies the functionality of the clustering module using Pytest.  
It generates sample spatial data, applies K-Means clustering, checks key outputs,  
and produces a visual plot for manual inspection.

Functionality Overview:

- `sample_data` (fixture):  
  Creates a synthetic dataset with latitude and longitude coordinates, simulated filling levels,  
  and timestamped entries. This mock dataset is used in all test functions to ensure repeatability.

- `test_apply_clustering`:  
  Verifies that the clustering function correctly assigns a 'Cluster' column and that the number  
  of clusters does not exceed the specified limit (3 in this case).

- `test_clustering_and_plot`:  
  - Applies clustering with default parameters.  
  - Confirms that multiple clusters are created and that the dataset remains complete.  
  - Generates a scatter plot visualising clusters by colour for manual validation.  
  - Saves the plot to the `tests/outputs` directory, creating it if necessary.

Testing Tools and Libraries:  
- `pytest` for testing framework and fixtures  
- `matplotlib` for cluster visualisation  
- `os` for file system operations  
- `pandas` and `numpy` for data handling and simulation

This script ensures that the clustering logic behaves as expected and produces clear, reproducible results.  
It supports both automated test verification and visual debugging.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.clustering import apply_clustering
import os


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "lat": np.random.uniform(40.0, 40.1, 100),
        "long": np.random.uniform(-3.7, -3.6, 100),
        "filling_level": np.random.randint(10, 100, 100),
        "Time": pd.date_range(start="2022-01-01", periods=100, freq="H")
    })

def test_apply_clustering(sample_data):
    clustered_data = apply_clustering(sample_data, n_clusters=3)

    assert "Cluster" in clustered_data.columns
    assert len(clustered_data["Cluster"].unique()) <= 3

def test_clustering_and_plot(sample_data):
    clustered_data = apply_clustering(sample_data)

    assert "Cluster" in clustered_data.columns
    n_clusters = clustered_data["Cluster"].nunique()

    assert n_clusters > 1
    assert clustered_data.shape[0] == 100

    # 📊 Plot clusters
    plt.figure(figsize=(8, 6))
    for cluster_id in clustered_data["Cluster"].unique():
        cluster_points = clustered_data[clustered_data["Cluster"] == cluster_id]
        plt.scatter(cluster_points["long"], cluster_points["lat"], label=f"Cluster {cluster_id}", alpha=0.6)

    plt.title("🧩 Clustering Visualization")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()

    #  Ensure folder exists
    os.makedirs("tests/outputs", exist_ok=True)

    output_path = "tests/outputs/clustering_test_plot.png"
    plt.savefig(output_path)
    plt.close()

    print(f"\n Clustering test passed. Plot saved to {output_path}")