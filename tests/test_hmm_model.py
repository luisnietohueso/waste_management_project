# tests/test_hmm_model.py
"""
Author: Luis Nieto Hueso  
Date: 28/03/2025  
Description:  
This test script validates the preprocessing and training components of the Hidden Markov Model (HMM) used in the  
smart waste bin system. It includes generation of test data, checks on encoded sequences, model training, and  
summary output for verification.

Fixtures:

- `sample_hmm_data`:  
  Produces synthetic fill-level data with 30 entries distributed evenly across 3 clusters.  
  Includes a date range for temporal ordering and values for the 'full-level' and 'Cluster' columns.

Tests:

- `test_prepare_hmm_data`:  
  Ensures that fill levels are properly encoded and grouped into sequences for HMM input.  
  Asserts that each sequence has at least 10 entries, as required for model stability.  
  Generates a plot showing how encoded levels vary over time for each cluster,  
  saving it to the `tests/outputs` directory for manual review.

- `test_train_hmm`:  
  Trains a Gaussian HMM on the prepared sequences.  
  Confirms that the model is trained and contains the required transition matrix attribute.  
  Outputs a markdown-formatted summary including the transition matrix and a link to the sequence plot.  
  The report is saved to the `tests/outputs` directory as `hmm_test_summary.md`.

Tools Used:  
- `pytest` for test execution and fixtures  
- `matplotlib` for time series plotting  
- `pandas` and `numpy` for data manipulation  
- HMM training and sequence preparation functions from the `models.hmm_model` module

This script ensures both automated and visual verification of HMM functionality in the system’s modelling pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from models.hmm_model import prepare_hmm_data, train_hmm

#  Ensure output dir exists
OUTPUT_DIR = "tests/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@pytest.fixture
def sample_hmm_data():
    """Creates sample data with at least 10+ data points per cluster."""
    return pd.DataFrame({
        "full-level": np.random.randint(0, 100, 30),
        "Cluster": [1]*10 + [2]*10 + [3]*10,
        "Time": pd.date_range(start="2022-01-01", periods=30, freq="D")
    })

def test_prepare_hmm_data(sample_hmm_data):
    data, sequences = prepare_hmm_data(sample_hmm_data)

    assert "filling_level_encoded" in data.columns
    assert len(sequences) > 0
    assert all(len(seq) >= 10 for seq in sequences)

    #  Plot encoded filling level per cluster
    plt.figure(figsize=(8, 5))
    for cluster in data["Cluster"].unique():
        cluster_data = data[data["Cluster"] == cluster]
        plt.plot(cluster_data["Time"], cluster_data["filling_level_encoded"], label=f"Cluster {cluster}")

    plt.title(" HMM - Encoded Filling Level per Cluster")
    plt.xlabel("Time")
    plt.ylabel("Encoded Level")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "hmm_encoded_sequence_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f" Plot saved to {plot_path}")

def test_train_hmm(sample_hmm_data):
    data, sequences = prepare_hmm_data(sample_hmm_data)
    hmm_model = train_hmm(data, sequences)

    assert hmm_model is not None
    assert hasattr(hmm_model, "transmat_")

    summary_path = os.path.join(OUTPUT_DIR, "hmm_test_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("#  HMM Model Test Summary\n")
        f.write(" Model trained successfully.\n\n")
        f.write("##  Transition Matrix\n")
        df = pd.DataFrame(hmm_model.transmat_)
        f.write(df.to_markdown(index=False))
        f.write("\n\n---\n![Encoded Sequences](hmm_encoded_sequence_plot.png)\n")
    print(f" Summary report saved to {summary_path}")
