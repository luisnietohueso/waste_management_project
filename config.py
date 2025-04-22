"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This configuration block defines essential file paths and model parameters for use throughout the waste management project.

File Paths:
- `FILE_PATH`: Specifies the location of the cleaned bin dataset in CSV format.
  This file is expected to include latitude, longitude, fill levels, and other relevant fields.

Hidden Markov Model (HMM) Configuration:
- `MAX_STATES`: Defines the maximum number of hidden states to be used when training the HMM.
  This helps control model complexity and prevent overfitting.

Recurrent Neural Network (RNN) Training Parameters:
- `EPOCHS`: Number of training iterations for the RNN model.
- `BATCH_SIZE`: Number of samples processed before updating the model during training.

Fuel Analysis Settings:
- `FUEL_EFFICIENCY`: Represents the fuel consumption rate of the vehicle in litres per kilometre.
- `FUEL_PRICE`: Specifies the fuel cost per litre, used for estimating total trip expenses.

These parameters support consistency, easy modification, and reuse across the different modules of the system.
"""


# File Paths
FILE_PATH = r"cleaned_bins_dataset.csv"

# HMM Config
MAX_STATES = 10

# RNN Training Config
EPOCHS = 20
BATCH_SIZE = 32

# Fuel Efficiency (L/km)
FUEL_EFFICIENCY = 0.35
FUEL_PRICE = 1.50
