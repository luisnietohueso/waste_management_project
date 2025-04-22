"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This utility script is responsible for generating both a markdown summary and a visual plot of fuel cost reduction analysis.
It is designed to complement the `analyze_fuel_cost_reduction()` function by exporting the results in a human-readable format
for reporting or testing purposes.

Function: `generate_fuel_summary_report()`
- Takes a dictionary of fuel-related metrics (e.g., fuel cost before/after optimisation, consumption, percentage reduction).
- Writes a markdown file (`fuel_summary.md`) summarising the results with a timestamp.
- The file is automatically saved in the `tests/outputs` directory, which is created if it doesn’t already exist.
- Useful for audit trails, documentation, and regression testing.

Function: `generate_fuel_summary_plot()`
- Produces a simple bar chart comparing fuel cost before and after optimisation.
- Labels each bar with the dollar amount and saves the figure as a PNG file in the specified path.
- Visually illustrates the benefit of route optimisation for presentations or summary reports.

The script uses:
- `os` for directory creation
- `datetime` to timestamp reports
- `matplotlib.pyplot` for visualisation

Together, these functions provide a lightweight reporting system for fuel and cost performance in waste management logistics.
"""


# utils/fuel_summary.py

import os
from datetime import datetime
import matplotlib.pyplot as plt

def generate_fuel_summary_report(metrics: dict, output_path="tests/outputs/fuel_summary.md"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#  Fuel Cost Reduction Summary\n")
        f.write(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("##  Metrics\n")
        for key, value in metrics.items():
            f.write(f"- **{key}**: {value:.2f}\n")

        f.write("\n *Generated automatically from `analyze_fuel_cost_reduction()` test.*\n")

    print(f"Fuel summary saved to {output_path}")


def generate_fuel_summary_plot(metrics: dict, save_path="tests/outputs/fuel_cost_plot.png"):
    before = metrics["Fuel Cost Before"]
    after = metrics["Fuel Cost After"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Before", "After"], [before, after], color=["red", "green"])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"${yval:.2f}", ha='center', fontsize=10)

    plt.title(" Fuel Cost Comparison")
    plt.ylabel("Fuel Cost ($)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f" Fuel cost plot saved to {save_path}")
