# Test file for data loader 
"""
Author: Luis Nieto Hueso  
Date: 28/03/2025  
Description:  
This test script verifies the functionality of the `load_and_preprocess_data` function from the data loading module.  
It ensures that a CSV file is correctly read and pre-processed — particularly focusing on the 'Time' column.

Function: `test_load_and_preprocess_data()`  
- Loads data from a sample CSV file (the path should be updated as required).  
- Asserts that the result is a valid pandas DataFrame.  
- Checks for the presence of the 'Time' column to confirm successful pre-processing.  
- Ensures that the dataset is not empty after loading.

Testing Tools and Libraries:  
- `pytest` for running the test  
- `pandas` for DataFrame assertions  
- `load_and_preprocess_data` from the data module for actual logic under test

This test helps confirm that the initial data ingestion and timestamp parsing behave reliably  
before further transformation or modelling takes place.
"""


import pytest
import pandas as pd
from data.data_loader import load_and_preprocess_data

def test_load_and_preprocess_data():
    file_path = r"C:\Users\Owner\Documents\waste_management_project\data\cleaned_bin_sensors_historical.csv"  # Replace with actual sample CSV file
    data = load_and_preprocess_data(file_path)

    assert isinstance(data, pd.DataFrame)
    assert "Time" in data.columns  # Ensure time column exists
    assert not data.empty  # Ensure data is loaded
