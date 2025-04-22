"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This script integrates the full pipeline of the Smart Waste Management system — from data loading to predictive modelling,
route optimisation, and performance evaluation. It combines HMM and RNN models for time-series forecasting, applies spatial
clustering for bin grouping, and analyses route efficiency and fuel cost reductions.

Key Steps Overview:

1. Data Loading and Pre-processing:
   Loads bin data from a CSV file and cleans the 'Time' column. Ensures the data is ready for clustering and modelling.

2. Clustering:
   Applies K-Means to group bins geographically using latitude and longitude, supporting optimised route computation.

3. Hidden Markov Model (HMM):
   - Prepares encoded fill-level sequences for training.
   - Trains a Gaussian HMM to model bin behaviour patterns.
   - Adds predicted HMM states back into the dataset for use as features.

4. Recurrent Neural Network (RNN):
   - Prepares lagged and engineered features for training.
   - Trains an LSTM-based RNN model to predict bin filling behaviour.
   - Makes predictions on both training and test datasets.

5. Route Optimisation and Distance Calculation:
   - Uses OSMnx to generate the road graph and computes the shortest path between full bins.
   - Calculates both optimised and static route distances based on geographic coordinates.

6. Visualisation and Fuel Cost Analysis:
   - Plots cluster and route comparisons.
   - Estimates total fuel consumption and cost savings based on distance reduction.
   - Outputs comparative metrics to support logistical decisions.

7. Model Evaluation:
   - HMM accuracy is evaluated against actual encoded fill-levels.
   - RNN predictions are inverse-scaled and compared using Mean Absolute Error (MAE) and R² score.
   - Accuracy-like metrics (based on relative MAE) are also reported for interpretability.

Modules Used:
- `data_loader`, `clustering`, `hmm_model`, and `rnn_model` for modelling tasks
- `route_optimization` for mapping and distance analysis
- `helpers` for visualisation and fuel cost computation
- `sklearn` and `numpy` for error metrics and data handling

This script provides a complete offline evaluation flow for testing the effectiveness of the waste management forecasting
and optimisation system.
"""

import os
import numpy as np

from data.data_loader import load_and_preprocess_data
from models.clustering import apply_clustering
from models.hmm_model import prepare_hmm_data, train_hmm, add_hmm_features
from models.rnn_model import prepare_rnn_data, train_rnn, predict_hybrid
from routes.route_optimization import compute_routes, compute_static_route, compute_distance
from utils.helpers import analyze_fuel_cost_reduction, visualize_results
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, r2_score

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# File path
file_path = r"cleaned_bins_dataset.csv"

#  Step 1: Load and preprocess data
data = load_and_preprocess_data(file_path)

#  Step 2: Apply clustering
data = apply_clustering(data)

#  Step 3: Prepare HMM data
data, sequences = prepare_hmm_data(data)

#  Step 4: Train HMM model
hmm_model = train_hmm(data, sequences)

#  Step 5: Add HMM features
data = add_hmm_features(data, hmm_model)

#  Step 6: Prepare RNN data
X_train, X_test, y_train, y_test, scaler = prepare_rnn_data(data)

#  Step 7: Train RNN model
rnn_model = train_rnn(X_train, y_train, X_test, y_test)

#  Step 8: Predict using the hybrid model
predictions = predict_hybrid(rnn_model, X_test, scaler)

#  Step 9: Compute optimized route
route, route_coordinates = compute_routes(data)

#  Step 10: Compute static route
static_route_coordinates = compute_static_route(data)

#  Step 11: Compute distances (Fixing missing variables)
optimized_distance = compute_distance(route_coordinates) if route_coordinates else 0
static_distance = compute_distance(static_route_coordinates) if static_route_coordinates else 0

#  Step 12: Compare routes and visualize results (Now passing the missing variables)
visualize_results(data, route, route_coordinates, static_route_coordinates, optimized_distance, static_distance)

#  Step 13: Compute and analyze fuel cost reduction
fuel_efficiency = 0.35  # L/km
fuel_price = 1.50  # $/L

analyze_fuel_cost_reduction(optimized_distance, static_distance, fuel_efficiency, fuel_price)

# Get HMM Predictions
hmm_predictions = hmm_model.predict(data[['filling_level_encoded']])

# Compute Accuracy
hmm_accuracy = accuracy_score(data['filling_level_encoded'], hmm_predictions) * 100

print(f" HMM Model Accuracy: {hmm_accuracy:.2f}%")
# Predict on Training & Test Data
rnn_train_predictions = rnn_model.predict(X_train)
rnn_test_predictions = rnn_model.predict(X_test)

# Convert predictions back to original scale
rnn_train_predictions = scaler.inverse_transform(rnn_train_predictions)
rnn_test_predictions = scaler.inverse_transform(rnn_test_predictions)
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Compute Errors
rnn_train_mae = mean_absolute_error(y_train_original, rnn_train_predictions)
rnn_test_mae = mean_absolute_error(y_test_original, rnn_test_predictions)

# Compute R² Score
rnn_train_r2 = r2_score(y_train_original, rnn_train_predictions)
rnn_test_r2 = r2_score(y_test_original, rnn_test_predictions)

# Convert to accuracy-like percentage
rnn_train_accuracy = max(0, (1 - rnn_train_mae / np.mean(y_train_original)) * 100)
rnn_test_accuracy = max(0, (1 - rnn_test_mae / np.mean(y_test_original)) * 100)

print(f" RNN Train Accuracy: {rnn_train_accuracy:.2f}%")
print(f" RNN Test Accuracy: {rnn_test_accuracy:.2f}%")
print(f" RNN Train R² Score: {rnn_train_r2:.4f}")
print(f" RNN Test R² Score: {rnn_test_r2:.4f}")