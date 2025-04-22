# tests/test_rnn_model.py
"""
Author: Luis Nieto Hueso  
Date: 28/03/2025  
Description:  
This test script validates the functionality and performance of the Recurrent Neural Network (RNN) model  
used for predicting waste bin fill levels. It includes checks for data preparation, model training, metric evaluation,  
and output visualisation.

Fixtures:

- `sample_rnn_data`:  
  Generates synthetic data with 50 entries for testing. Each entry includes a random fill level, cluster ID,  
  and a timestamp for use in time-series modelling.

Tests:

- `test_prepare_rnn_data`:  
  Confirms that data preparation returns valid training and testing sets, ensuring that inputs and outputs are non-empty  
  and properly formatted for RNN training.

- `test_train_rnn_with_metrics_and_plot`:  
  - Trains the RNN model on the synthetic data.  
  - Makes predictions and applies inverse scaling to return to the original filling level scale.  
  - Calculates Mean Absolute Error (MAE) and R² score to assess model accuracy.  
  - Asserts that the model learns meaningful patterns (e.g. MAE < 30 and R² > 0).  
  - Produces a visual plot comparing actual and predicted values.  
  - Generates a markdown summary report including performance metrics and a link to the plot.

Tools Used:  
- `pytest` for test execution and fixtures  
- `matplotlib` for visualisation  
- `sklearn.metrics` for evaluation (MAE and R²)  
- `rnn_model` module for training and data preparation

The test provides both automated verification and a human-readable report to support reliable deployment  
and evaluation of the RNN model within the system.
"""

import os
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from models.rnn_model import prepare_rnn_data, train_rnn

@pytest.fixture
def sample_rnn_data():
    return pd.DataFrame({
        "filling_level": np.random.randint(0, 100, 50),
        "Cluster": np.random.randint(0, 5, 50),
        "Time": pd.date_range(start="2022-01-01", periods=50, freq="D")
    })

def test_prepare_rnn_data(sample_rnn_data):
    X_train, X_test, y_train, y_test, scaler = prepare_rnn_data(sample_rnn_data)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

def test_train_rnn_with_metrics_and_plot(sample_rnn_data):
    X_train, X_test, y_train, y_test, scaler = prepare_rnn_data(sample_rnn_data)
    rnn_model = train_rnn(X_train, y_train, X_test, y_test)

    assert rnn_model is not None

    #  Predict and inverse transform
    predictions = rnn_model.predict(X_test)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_orig = scaler.inverse_transform(predictions)

    #  Metrics
    mae = mean_absolute_error(y_test_orig, predictions_orig)
    r2 = r2_score(y_test_orig, predictions_orig)

    #  Assert model learned something (low MAE, decent R²)
    assert mae < 30
    assert r2 > 0  # Not negative R²

    #  Plot actual vs predicted
    os.makedirs("tests/outputs", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(y_test_orig, label="Actual", marker='o')
    plt.plot(predictions_orig, label="Predicted", marker='x')
    plt.title("📉 RNN Prediction vs Actual")
    plt.xlabel("Sample")
    plt.ylabel("Filling Level")
    plt.legend()
    plot_path = "tests/outputs/rnn_test_predictions_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    #  Summary Markdown
    summary_path = "tests/outputs/rnn_test_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("#  RNN Test Summary\n")
        f.write(f"- MAE: `{mae:.2f}`\n")
        f.write(f"- R² Score: `{r2:.4f}`\n")
        f.write(f"![Prediction Plot]({plot_path})\n")

    print(f"\n RNN test completed.\n Plot saved: {plot_path}\n Summary: {summary_path}")
