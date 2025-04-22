"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This script prepares bin fill-level data for sequence modelling using a Hidden Markov Model (HMM),
trains the model, and applies it to assign predicted hidden states to each observation.

The necessary libraries are imported first:
- `numpy` for numerical operations
- `pandas` for data handling
- `hmmlearn` for building and training the HMM

The function `prepare_hmm_data` begins by verifying the presence of the 'full-level' column.
It converts this column to numeric, sets negative values to zero, and re-bins it into four categories
(low, medium, high, full) using predefined intervals. These are encoded into discrete integers for modelling.
Any missing or invalid values are replaced with the most frequent valid label.
The function then segments the data into sequences by bin cluster (using the 'Cluster' column),
sorted chronologically, filtering out sequences with fewer than 6 observations.

The `train_hmm` function filters out sequences with fewer than 10 steps to ensure robust model training.
It concatenates all valid sequences into a single training input, reshapes it, and determines the number
of unique fill-level states to use in the model (maximum of 6).
A Gaussian HMM is then trained using the combined sequence data and sequence lengths.
During training, the function handles potential errors in data formatting and logs the
initial state probabilities and transition matrix after completion.

The final function, `add_hmm_features`, applies the trained HMM model to the dataset and
predicts the hidden state sequence, storing it in a new column named 'HMM_State'.
This feature can be useful for downstream analysis or predictive modelling.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm

def prepare_hmm_data(data):
    if 'full-level' not in data.columns:
        raise KeyError(" ERROR: 'full-level' column is missing!")

    # Convert 'full-level' to numeric and handle negative values
    data['full-level'] = pd.to_numeric(data['full-level'], errors='coerce')
    data.loc[data['full-level'] < 0, 'full-level'] = 0  # Set negative values to 0

    #  Step 1: Reduce the Number of Categories (Re-Binning)
    bins = [0, 25, 50, 75, 100]  # Define bins (low, medium, high, full)
    labels = [0, 1, 2, 3]  # Encode as discrete states
    data['filling_level_encoded'] = pd.cut(data['full-level'], bins=bins, labels=labels, include_lowest=True)

    #  Fix: Handle NaNs by replacing them with the lowest category (0)
    data['filling_level_encoded'] = data['filling_level_encoded'].cat.codes  # Convert categorical to int

    print(" New Encoded Filling Level Distribution:\n", data['filling_level_encoded'].value_counts())
    most_frequent_label = data['filling_level_encoded'][data['filling_level_encoded'] != -1].mode()[0]
    data['filling_level_encoded'].replace(-1, most_frequent_label, inplace=True)

    #  Step 2: Prepare sequences for HMM
    sequences = []
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster].sort_values(by=['Time'])
        if len(cluster_data) > 5:  # Filter out short sequences
            sequences.append(cluster_data['filling_level_encoded'].values)

    if len(sequences) == 0:
        raise ValueError(" ERROR: No valid sequences found! Ensure enough data points.")

    return data, sequences

def train_hmm(data, sequences, max_states=10):
    if len(sequences) == 0:
        raise ValueError(" No sequences found! Ensure 'filling_level_encoded' is correctly processed.")

    # Remove sequences with <10 steps to improve model training
    sequences = [seq for seq in sequences if len(seq) >= 10]
    if len(sequences) == 0:
        raise ValueError(" No valid sequences found! Ensure enough data points per cluster.")

    lengths = [len(seq) for seq in sequences]
    print(f" Min sequence length: {min(lengths)}, Max: {max(lengths)}, Avg: {np.mean(lengths):.2f}")

    #  Ensure `combined_sequence` is properly initialized
    try:
        combined_sequence = np.concatenate(sequences).reshape(-1, 1)
    except ValueError as e:
        raise ValueError(" Error concatenating sequences. Check if sequences are empty or misaligned.") from e

    print("🔍 Unique states in combined sequence:", np.unique(combined_sequence))

    unique_states = np.unique(data['filling_level_encoded'])
    n_states = min(len(unique_states), 6)  # Use at most 6 states

    print(f"🚀 Adjusted number of states for HMM: {n_states}")

    # Train the model
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=5000, random_state=84)
    print(f"⏳ Training HMM Model with {n_states} states...")

    try:
        model.fit(combined_sequence, lengths)
    except Exception as e:
        raise RuntimeError(" HMM training failed. Check input sequences.") from e

    print(" HMM Model Training Completed!")
    print(" Start Probabilities:", model.startprob_)
    print(" Transition Matrix:\n", model.transmat_)

    return model

def add_hmm_features(data, hmm_model):
    data['HMM_State'] = hmm_model.predict(data[['filling_level_encoded']])
    return data
