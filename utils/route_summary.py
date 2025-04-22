"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This script provides utilities for summarising and visualising the results of route optimisation tests.
It generates markdown reports and comparison plots that highlight key metrics between optimised and static collection routes.

Function: `generate_detailed_route_summary()`
- Creates a markdown summary file that outlines key results from route optimisation, including distances travelled,
  number of stops, and a preview of coordinates.
- The report includes a timestamp and embeds a route visualisation image for easy interpretation.
- The destination folder is automatically created if it doesn't exist.
- This function is especially useful for logging test results or compiling documentation during development.

Function: `generate_route_summary_plot()`
- Generates a bar chart comparing the optimised and static routes in terms of total distance and number of stops.
- Distances and stop counts are colour-coded (green for optimised, red for static) for clear visual comparison.
- Value labels are displayed on each bar to enhance readability.
- The plot is saved to the specified path, and the containing directory is created if necessary.

This script supports consistent and automated reporting of route efficiency testing, allowing for quick validation,
performance tracking, and inclusion in presentations or final reports.
"""


# utils/route_summary.py
import os
from datetime import datetime
import matplotlib.pyplot as plt

def generate_detailed_route_summary(metrics: dict, plot_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#  Route Optimization Test Summary\n")
        f.write(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("##  Key Metrics\n")
        f.write(f"-  Optimized Route Distance: {metrics['Optimized Route Distance (km)']} km\n")
        f.write(f"-  Static Route Distance: {metrics['Static Route Distance (km)']} km\n")
        f.write(f"-  Optimized Route Stops: {metrics['Optimized Stops']}\n")
        f.write(f"-  Static Route Stops: {metrics['Static Stops']}\n\n")

        f.write("##  Coordinates Preview\n")
        f.write(f"- Optimized (first 5): {metrics['Optimized Coords (first 5)']}\n")
        f.write(f"- Static (first 5): {metrics['Static Coords (first 5)']}\n\n")

        f.write("## 🖼️ Route Visualization\n")
        f.write(f"![Route Comparison]({os.path.basename(plot_path)})\n\n")

        f.write("✅ *Generated automatically by test script.*\n")




def generate_route_summary_plot(metrics: dict, save_path: str):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        ['Optimized Distance (km)', 'Static Distance (km)', 'Optimized Stops', 'Static Stops'],
        [
            metrics['Optimized Route Distance (km)'],
            metrics['Static Route Distance (km)'],
            metrics['Optimized Stops'],
            metrics['Static Stops']
        ],
        color=['green', 'red', 'green', 'red']
    )

    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}", ha='center', fontsize=10)

    plt.title(" Route Optimization Summary", fontsize=16)
    plt.ylabel("Distance (km) / Number of Stops")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f" Summary plot saved to {save_path}")
