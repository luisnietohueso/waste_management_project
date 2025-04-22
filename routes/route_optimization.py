
"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This script handles route optimisation and fuel consumption analysis for waste collection vehicles.
It uses geographical data of waste bins to determine the most efficient travel paths, calculates the total distance travelled,
and estimates fuel usage and cost savings based on optimised vs static routing strategies.

The following libraries are used:
- `pandas` for data handling
- `osmnx` to retrieve and work with street networks
- `geopy` to compute geographic distances using the geodesic method

Routing Functions:
- `compute_routes(data)`: Builds a graph of the road network for a specified area (Eurobodalla, Australia) and identifies
  a route connecting bins marked as 'Full'. It finds the nearest nodes on the road network for each bin, removes duplicates,
  and returns both the list of graph nodes and their geographic coordinates.

- `compute_static_route(data)`: Generates a static list of bin coordinates for waste collection based on bins classified as
  'Full' or 'Medium'. This route does not account for road networks or path optimisation.

- `compute_distance(route_coordinates)`: Calculates the total geodesic distance (in kilometres) of a given route based
  on sequential coordinates using the `geopy` library.

Fuel Analysis Functions:
- `compute_fuel_consumption(distance, fuel_efficiency)`: Calculates the total fuel required to travel a given distance
  based on vehicle fuel efficiency (in litres per kilometre).

- `compute_fuel_cost(fuel_consumption, fuel_price)`: Computes the cost of fuel based on total consumption and the fuel price per litre.

- `analyze_fuel_cost_reduction(...)`: Compares fuel consumption and cost before and after route optimisation.
  It calculates the fuel and cost differences and outputs a percentage reduction in fuel expenses.

Overall, this module enables an effective evaluation of route planning strategies with direct implications
on cost efficiency and sustainability in waste collection logistics.
"""

import pandas as pd
import osmnx as ox
from geopy.distance import geodesic

def compute_routes(data):
    if not isinstance(data, pd.DataFrame):
        print("Data is not a DataFrame. Received type:", type(data))
        return None, None

    # Define area of interest
    area = "Eurobodalla, Australia"
    G = ox.graph_from_place(area, network_type='drive')

    # Add traffic weights (if available)
    for u, v, edge_data in G.edges(data=True):
        edge_data['weight'] = edge_data.get('length', 1)  # Use road length as weight

    if 'lat' in data.columns and 'long' in data.columns:
        full_bins = data[data['filling_level_map'] == 'Full (>900)']
        if len(full_bins) > 1:
            # Ensure route only considers full bins
            bin_coordinates = full_bins[['lat', 'long']].values.tolist()

            # Find the nearest network nodes for each bin
            bin_nodes = [ox.nearest_nodes(G, coord[1], coord[0]) for coord in bin_coordinates]

            # Compute the shortest path through all nodes
            route = []
            route_coordinates = []
            for node in bin_nodes:
                if node not in route:  # Avoid duplicate nodes
                    route.append(node)
                    route_coordinates.append((G.nodes[node]['y'], G.nodes[node]['x']))  # Get coordinates of the stop

            return route, route_coordinates
        else:
            print("Not enough 'Full' bins for routing.")
            return None, None
    else:
        print("Dataset is missing 'lat' and 'long' columns.")
        return None, None


def compute_static_route(data):
    """
    Define a static route as a list of coordinates for bins that need collection.
    """
    if 'lat' in data.columns and 'long' in data.columns:
        # Focus on bins that are full or nearly full
        static_bins = data[data['filling_level_map'].isin(['Full (>900)', 'Medium (<600)'])]

        # Ensure no missing values in latitude and longitude
        static_bins = static_bins.dropna(subset=['lat', 'long'])

        # Create a list of coordinates for these bins
        static_route_coordinates = static_bins[['lat', 'long']].values.tolist()
        return static_route_coordinates
    else:
        print("Dataset is missing 'lat' and 'long' columns.")
        return []


def compute_distance(route_coordinates):
    """
    Compute the total distance of a route based on its coordinates.
    """
    total_distance = 0
    for i in range(1, len(route_coordinates)):
        coord1 = route_coordinates[i - 1]
        coord2 = route_coordinates[i]
        # Use geodesic distance from geopy
        total_distance += geodesic(coord1, coord2).kilometers
    return total_distance

# Fuel Consumption and Cost Analysis

def compute_fuel_consumption(distance, fuel_efficiency):
    """
    Compute total fuel consumption based on distance traveled and vehicle fuel efficiency.
    :param distance: Total distance traveled (km)
    :param fuel_efficiency: Fuel efficiency of the vehicle (L/km)
    :return: Total fuel consumption (L)
    """
    return distance * fuel_efficiency


def compute_fuel_cost(fuel_consumption, fuel_price):
    """
    Compute the total fuel cost based on fuel consumption and fuel price per liter.
    :param fuel_consumption: Total fuel consumption (L)
    :param fuel_price: Fuel price per liter ($/L)
    :return: Total fuel cost ($)
    """
    return fuel_consumption * fuel_price


def analyze_fuel_cost_reduction(optimized_distance, static_distance, fuel_efficiency, fuel_price):
    """
    Perform a fuel consumption and cost analysis before and after route optimization.
    """
    # Compute fuel consumption
    fuel_consumption_before = compute_fuel_consumption(static_distance, fuel_efficiency)
    fuel_consumption_after = compute_fuel_consumption(optimized_distance, fuel_efficiency)

    # Compute fuel cost
    fuel_cost_before = compute_fuel_cost(fuel_consumption_before, fuel_price)
    fuel_cost_after = compute_fuel_cost(fuel_consumption_after, fuel_price)

    # Compute cost reduction percentage
    cost_reduction_percentage = ((fuel_cost_before - fuel_cost_after) / fuel_cost_before) * 100

    # Print results
    print("\nFuel Consumption and Cost Analysis:")
    print(f"Fuel Efficiency: {fuel_efficiency} L/km")
    print(f"Fuel Price: ${fuel_price:.2f} per liter")
    print(f"Total Fuel Consumption (Before Optimization): {fuel_consumption_before:.2f} L")
    print(f"Total Fuel Consumption (After Optimization): {fuel_consumption_after:.2f} L")
    print(f"Total Fuel Cost (Before Optimization): ${fuel_cost_before:.2f}")
    print(f"Total Fuel Cost (After Optimization): ${fuel_cost_after:.2f}")
    print(f"Fuel Cost Reduction: {cost_reduction_percentage:.2f}%\n")

    return fuel_consumption_before, fuel_consumption_after, fuel_cost_before, fuel_cost_after, cost_reduction_percentage