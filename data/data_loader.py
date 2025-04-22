"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This script manages the loading and pre-processing of time-related data from a CSV file.
It begins by importing the required libraries: `pandas` for data handling and `parser` from `dateutil`
for flexible date and time parsing.
The main function, `load_and_preprocess_data`, reads a CSV file and checks for the presence of a 'Time' column.
It then cleans the 'Time' column by removing timezone offsets (e.g., '+00:00') using a regular expression.
Each entry is then converted into a standard datetime object using the helper function `flexible_parse`.
Rows with invalid or unparseable 'Time' values are removed from the dataset.

The helper function `flexible_parse` attempts to parse a string into a datetime object using `dateutil.parser`.
If parsing fails, it outputs an error message for debugging and returns `None`.
This ensures that only clean, properly formatted date entries remain in the dataset for further analysis.
"""


import pandas as pd
from dateutil import parser

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    if 'Time' not in data.columns:
        raise ValueError("The dataset does not contain a 'Time' column.")

    # Clean and parse Time column
    data['Time'] = data['Time'].str.replace(r'\+\d{2}:\d{2}$', '', regex=True)
    data['Time'] = data['Time'].apply(flexible_parse)

    # Drop invalid 'Time' rows
    data = data.dropna(subset=['Time'])

    return data


def flexible_parse(time_str):
    try:
        return parser.parse(time_str)
    except Exception as e:
        print(f"Failed to parse: {time_str} -> {e}")
        return None
