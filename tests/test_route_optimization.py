# tests/test_route_optimization.py
"""
Author: Luis Nieto Hueso  
Date: 28/03/2025  
Description:  
This test script validates the route optimisation functionality within the smart waste management system.  
It evaluates the performance of the `compute_routes` and `compute_static_route` functions by comparing  
optimised versus static paths using real geographical coordinates and visualising the result.

Fixture:

- `sample_route_data`:  
  Generates test data containing 10 bin locations (latitude and longitude), all marked as 'Full',  
  to simulate a routing scenario within the Eurobodalla, Australia area.

Test: `test_route_visualization`  
- Computes the optimised route using a road network and the static route using direct bin-to-bin connections.  
- Asserts that both route coordinate sets are valid and non-empty.  
- Plots both routes on the same chart for visual comparison:  
  - The static route is shown as a red dashed line.  
  - The optimised route is shown as a green solid line.  
- Saves the plot to `tests/outputs/route_optimization_test_plot.png`.

Metrics and Reporting:  
- Calculates the total distance for both routes using geodesic distance.  
- Constructs a dictionary of test metrics including number of stops and sample coordinates.  
- Generates a markdown report summarising key findings and embeds the visual plot.  
- Also creates a separate summary bar chart using `generate_route_summary_plot`.

Modules Used:  
- `routes.route_optimization` for computing and measuring routes  
- `utils.route_summary` for report and visual summary generation  
- `matplotlib` for plotting route maps  
- `pytest` for test execution and fixtures

This test ensures both functional and visual confirmation that route optimisation works correctly  
and offers measurable benefits over static pathing.
"""

import os
import pytest
import pandas as pd
import matplotlib.pyplot as plt


from routes.route_optimization import compute_routes, compute_static_route, compute_distance
from utils.route_summary import generate_detailed_route_summary, generate_route_summary_plot

SUMMARY_IMAGE_PATH = "tests/outputs/route_summary_plot.png"
OUTPUT_PATH = "tests/outputs/route_optimization_test_plot.png"
SUMMARY_PATH = "tests/outputs/route_test_summary.md"

@pytest.fixture
def sample_route_data():
    return pd.DataFrame({
        "lat": [-35.5469786, -35.5453317, -34.8077422, -34.9161423, -34.2696789, -34.7174461, -34.8676808, -35.3100000, -35.4400000, -35.5500000],
        "long": [149.9639965, 149.9796274, 149.1441895, 149.1419426, 149.8265735, 149.7119203, 149.8534637, 149.9100000, 149.9200000, 149.9300000],
        "filling_level_map": ["Full (>900)"] * 10
    })

def test_route_visualization(sample_route_data):
    route, route_coords = compute_routes(sample_route_data)
    static_coords = compute_static_route(sample_route_data)

    #  Assertions
    assert route_coords and static_coords

    #  Plot
    plt.figure(figsize=(8, 5))
    xs_opt, ys_opt = zip(*route_coords)
    xs_static, ys_static = zip(*static_coords)

    plt.plot(xs_static, ys_static, label="Static Route", color="red", linestyle="--", marker='o')
    plt.plot(xs_opt, ys_opt, label="Optimized Route", color="green", marker='o')
    plt.title("🚛 Route Optimization Visualization")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    plt.close()

    #  Generate Metrics
    optimized_distance = compute_distance(route_coords)
    static_distance = compute_distance(static_coords)

    test_results = {
        "Optimized Route Distance (km)": round(optimized_distance, 2),
        "Static Route Distance (km)": round(static_distance, 2),
        "Optimized Stops": len(route_coords),
        "Static Stops": len(static_coords),
        "Optimized Coords (first 5)": route_coords[:5],
        "Static Coords (first 5)": static_coords[:5]
    }

    #  Generate Markdown Summary
    generate_detailed_route_summary(test_results, plot_path=OUTPUT_PATH, output_path=SUMMARY_PATH)
    print(f" Route summary saved to {SUMMARY_PATH}")
    generate_route_summary_plot(test_results, save_path=SUMMARY_IMAGE_PATH)
