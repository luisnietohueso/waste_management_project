"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This script provides functions for interactive and static visualisation of waste bin clusters, route comparisons,
and fuel cost analysis based on optimised and static collection routes.

Function: `display_map(data)`
- Uses the `folium` library to render an interactive map showing bin locations and their corresponding cluster assignments.
- The map is centred based on the average coordinates of the dataset and rendered in a Streamlit environment using `folium_static`.
- Each bin is marked with a popup label indicating its cluster number.

Function: `visualize_results(...)`
- Visualises the distance covered by the optimised and static routes using a bar chart.
- Also plots a scatter chart of all waste bins coloured by cluster, using latitude and longitude.
- Provides a visual understanding of route efficiency and the spatial distribution of bins.
- Automatically checks that route distances are valid before plotting.

Function: `analyze_fuel_cost_reduction(...)`
- Calculates and prints a breakdown of fuel consumption and cost before and after route optimisation.
- Takes into account the total distance, vehicle fuel efficiency (litres/km), and fuel price per litre.
- Outputs fuel usage in litres, fuel costs in currency, and the percentage reduction achieved.
- Returns all values for potential use in reports or further analysis.

This module integrates visual analytics and sustainability metrics, enabling better decision-making
in waste collection route planning.
"""
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt

def display_map(data):
    m = folium.Map(location=[data['lat'].mean(), data['long'].mean()], zoom_start=12)
    for _, row in data.iterrows():
        folium.Marker(location=[row['lat'], row['long']], popup=f"Cluster: {row['Cluster']}").add_to(m)
    folium_static(m)


def visualize_results(data, route, route_coordinates, static_route_coordinates, optimized_distance, static_distance):
    """
    Visualizes route distances and bin clusters.
    """
    print(f" Optimized Distance: {optimized_distance:.2f} km, Static Distance: {static_distance:.2f} km")

    #  Ensure values exist before plotting
    if optimized_distance > 0 and static_distance > 0:
        plt.figure(figsize=(8, 6))
        plt.bar(["Optimized Route", "Static Route"], [optimized_distance, static_distance], color=['blue', 'red'])
        plt.ylabel("Distance (km)")
        plt.title("Route Distance Comparison")
        plt.show()
    else:
        print(" Not enough data to plot distance comparison.")

    #  Additional visualizations for route clustering
    plt.figure(figsize=(10, 8))
    plt.scatter(data["long"], data["lat"], c=data["Cluster"], cmap="viridis", alpha=0.7)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Clustered Waste Bins")
    plt.colorbar(label="Cluster")
    plt.show()

def analyze_fuel_cost_reduction(optimized_distance, static_distance, fuel_efficiency, fuel_price):
    """
    Compute fuel consumption and cost before and after optimization.
    """
    fuel_consumption_before = static_distance * fuel_efficiency
    fuel_consumption_after = optimized_distance * fuel_efficiency

    fuel_cost_before = fuel_consumption_before * fuel_price
    fuel_cost_after = fuel_consumption_after * fuel_price

    cost_reduction_percentage = ((fuel_cost_before - fuel_cost_after) / fuel_cost_before) * 100 if fuel_cost_before > 0 else 0

    print("\n Fuel Cost Analysis:")
    print(f" Fuel Efficiency: {fuel_efficiency:.2f} L/km")
    print(f" Fuel Price: ${fuel_price:.2f}/L")
    print(f" Fuel Cost Before: ${fuel_cost_before:.2f}")
    print(f" Fuel Cost After: ${fuel_cost_after:.2f}")
    print(f" Cost Reduction: {cost_reduction_percentage:.2f}%\n")

    return fuel_consumption_before, fuel_consumption_after, fuel_cost_before, fuel_cost_after, cost_reduction_percentage