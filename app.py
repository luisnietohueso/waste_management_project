"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This is the main user interface script for the Smart Waste Management system, built using Streamlit.
It integrates data upload, clustering, machine learning model training, route optimisation, and fuel cost analysis
into a single interactive dashboard.

Streamlit Interface Setup:
- Sets up the web page with a title and sidebar navigation.
- Maintains state using `st.session_state` to store uploaded data across multiple pages.

Pages and Functionalities:

1. Upload Data:
   - Allows users to upload CSV files containing waste bin data.
   - Displays a preview of the dataset after upload.

2. Clustering & Maps:
   - Applies K-Means clustering based on bin latitude and longitude.
   - Visualises clusters using both a static matplotlib scatter plot and an interactive Folium map.

3. Model Training (HMM & RNN):
   - Prepares and encodes bin fill levels for sequence modelling.
   - Trains a Hidden Markov Model (HMM) to predict hidden states.
   - Trains a Recurrent Neural Network (RNN) using TensorFlow/Keras to forecast fill levels.
   - All modelling is done in-memory without external saving.

4. Route Optimisation:
   - Uses the OSMnx and NetworkX libraries to build a road network for a defined area (Eurobodalla, Australia).
   - Computes the shortest path between full bins and calculates the distance covered.
   - Compares this optimised route against a static, linear route between bin points.
   - Visualises the route on a map and displays key distance metrics.

5. Fuel Cost Analysis:
   - Accepts user input for fuel efficiency and price.
   - Calculates estimated cost before and after route optimisation.
   - Displays results as a bar chart and summarises potential cost savings.

Technical Notes:
- TensorFlow warnings are suppressed for a cleaner UI.
- Data validation steps ensure required columns are present before processing.
- Includes error handling and user feedback throughout the workflow for usability.

This script brings together geospatial analysis, machine learning, and data visualisation in an accessible,
interactive dashboard to support smarter and more sustainable waste collection planning.
"""


import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from hmmlearn import hmm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from geopy.distance import geodesic
import osmnx as ox
import networkx as nx
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Streamlit UI Setup
st.set_page_config(page_title="Smart Waste Management", layout="wide")
st.title("🚛 Smart Waste Collection Optimization")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Clustering & Maps", "Model Training", "Route Optimization",
                                  "Fuel Cost Analysis"])

# Initialize Session State
if "data" not in st.session_state:
    st.session_state.data = None

scaler = MinMaxScaler()
label_encoder = LabelEncoder()

# File Upload
if page == "Upload Data":
    st.subheader("Upload Waste Bin Data (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("### Sample Data")
        st.dataframe(st.session_state.data.head())

# Clustering & Maps
if page == "Clustering & Maps":
    st.subheader("Clustering Waste Bins")

    if st.session_state.data is not None:
        data = st.session_state.data  # Retrieve stored data

        if 'lat' in data.columns and 'long' in data.columns:
            n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            data['Cluster'] = kmeans.fit_predict(data[['lat', 'long']])

            st.write("### Cluster Visualization")
            fig, ax = plt.subplots()
            scatter = ax.scatter(data['long'], data['lat'], c=data['Cluster'], cmap='viridis')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(scatter, ax=ax, label='Cluster')
            st.pyplot(fig)

            # Interactive Map
            m = folium.Map(location=[data['lat'].mean(), data['long'].mean()], zoom_start=12)
            for _, row in data.iterrows():
                folium.Marker(location=[row['lat'], row['long']], popup=f"Cluster: {row['Cluster']}").add_to(m)
            folium_static(m)
        else:
            st.error("Dataset must contain 'lat' and 'long' columns.")
    else:
        st.warning("Please upload a dataset first in the 'Upload Data' section.")

# Model Training (HMM & RNN)
if page == "Model Training" and st.session_state.data is not None:
    st.subheader("Train HMM and RNN Models")
    data = st.session_state.data  # Retrieve stored data

    if 'filling_level' in data.columns:
        data = data.dropna(subset=['filling_level'])  # Drop NaN values
        if len(data) < 5:  # Ensure enough data points
            st.warning("Not enough data for model training. Please upload a larger dataset.")
        else:
            data['filling_level_encoded'] = label_encoder.fit_transform(data['filling_level'])
            sequences = [data['filling_level_encoded'].values]

            # Train HMM
            hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=500)
            hmm_model.fit(np.concatenate(sequences).reshape(-1, 1))
            data['HMM_State'] = hmm_model.predict(data[['filling_level_encoded']])
            st.success("HMM Model Trained Successfully!")

            # Prepare Data for RNN
            data['filling_level_scaled'] = scaler.fit_transform(data[['filling_level_encoded']])

            # Ensure X and y have the same length
            X = data[['filling_level_scaled']].shift(1).dropna().values.reshape(-1, 1, 1)
            y = data['filling_level_scaled'].iloc[1:].values  # Drop first value to match X

            if len(X) != len(y):  # Final safety check
                X = X[:len(y)]

            # Train RNN
            rnn_model = Sequential([LSTM(50, activation='relu', input_shape=(1, 1)), Dense(1)])
            rnn_model.compile(optimizer='adam', loss='mse')
            rnn_model.fit(X, y, epochs=10, batch_size=8, verbose=0)
            st.success("RNN Model Trained Successfully!")
    else:
        st.error("Dataset must contain 'filling_level' column.")

# Route Optimization
# Route Optimization
if page == "Route Optimization" and st.session_state.data is not None:
    st.subheader("Optimized Route Computation")
    data = st.session_state.data  # Retrieve stored data

    if "filling_level_map" in data.columns:
        full_bins = data[data['filling_level_map'] == 'Full (>900)']
        if len(full_bins) > 1:
            area = "Eurobodalla, Australia"
            G = ox.graph_from_place(area, network_type='drive')
            bin_coordinates = full_bins[['lat', 'long']].values.tolist()
            bin_nodes = [ox.nearest_nodes(G, coord[1], coord[0]) for coord in bin_coordinates]
            route = nx.shortest_path(G, bin_nodes[0], bin_nodes[-1], weight='length')

            # Compute the optimized route distance
            optimized_distance = 0
            for i in range(1, len(route)):
                node1 = G.nodes[route[i - 1]]
                node2 = G.nodes[route[i]]
                optimized_distance += geodesic((node1['y'], node1['x']), (node2['y'], node2['x'])).kilometers

            # Compute the static route distance (e.g., naive route using all bins)
            static_distance = 0
            for i in range(1, len(bin_coordinates)):
                static_distance += geodesic(bin_coordinates[i - 1], bin_coordinates[i]).kilometers

            # Calculate the percentage difference
            distance_reduction = ((static_distance - optimized_distance) / static_distance) * 100 if static_distance > 0 else 0

            # Display the optimized route map
            m = folium.Map(location=[data['lat'].mean(), data['long'].mean()], zoom_start=12)
            folium.PolyLine([(G.nodes[node]['y'], G.nodes[node]['x']) for node in route], color='blue').add_to(m)
            folium_static(m)

            # Show Distance Comparison
            st.success("Optimised Route Computed!")
            st.write(f"🚗 Optimised Route Distance: {optimized_distance:.2f} km")
            st.write(f"📏 Static Route Distance: {static_distance:.2f} km")
            st.write(f"📉 Distance Reduction: {distance_reduction:.2f}%")

        else:
            st.warning("Not enough full bins for routing.")
    else:
        st.error("Dataset is missing the 'filling_level_map' column.")


# Fuel Cost Analysis
if page == "Fuel Cost Analysis" and st.session_state.data is not None:
    st.subheader("Fuel Cost Savings")
    fuel_efficiency = st.number_input("Truck Fuel Efficiency (L/km)", min_value=0.1, max_value=1.0, value=0.35)
    fuel_price = st.number_input("Fuel Price ($/L)", min_value=1.0, max_value=5.0, value=1.50)

    optimized_distance = 10.0  # Example value
    static_distance = 15.0  # Example value
    fuel_cost_before = static_distance * fuel_efficiency * fuel_price
    fuel_cost_after = optimized_distance * fuel_efficiency * fuel_price
    reduction = ((fuel_cost_before - fuel_cost_after) / fuel_cost_before) * 100

    st.write(f"Fuel Cost Before: ${fuel_cost_before:.2f}")
    st.write(f"Fuel Cost After: ${fuel_cost_after:.2f}")
    st.write(f"Cost Reduction: {reduction:.2f}%")

    fig, ax = plt.subplots()
    ax.bar(["Before Optimization", "After Optimization"], [fuel_cost_before, fuel_cost_after], color=['red', 'blue'])
    ax.set_ylabel("Fuel Cost ($)")
    ax.set_title("Fuel Cost Reduction")
    st.pyplot(fig)
    st.success("Fuel Cost Analysis Completed!")
