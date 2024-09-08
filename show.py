





import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import time

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        required_columns = ['latitude', 'longitude', 'time', 'incident_type']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def optimize_dbscan_params(X, k_dist=4):
    neigh = NearestNeighbors(n_neighbors=k_dist)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations")
    plt.savefig('k_distance_graph.png')
    plt.close()
    
    knee = np.diff(distances, 2)
    knee = knee.argmax() + 2
    eps = distances[knee]
    print(f"Optimal eps: {eps}")
    return eps, k_dist

def perform_clustering(data, eps, min_samples):
    coordinates = data[['latitude', 'longitude']].values
    scaler = StandardScaler()
    scaled_coordinates = scaler.fit_transform(coordinates)
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_coordinates)
    data['cluster'] = db.labels_
    return data

def create_map(data):
    map_center = [data['latitude'].mean(), data['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=13)
    
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, row in data.iterrows():
        color = 'red' if row['cluster'] == -1 else 'blue'
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=5,
            popup=f"Incident: {row['incident_type']}<br>Time: {row['time']}<br>Cluster: {row['cluster']}",
            color=color,
            fill=True,
            fillColor=color
        ).add_to(marker_cluster)
    
    return m

def run_show():
    file_path = 'incident_data.csv'  # Update this to your file path
    data = load_data(file_path)
    if data is None:
        return

    print(f"Loaded {len(data)} incidents.")

    coordinates = data[['latitude', 'longitude']].values
    scaler = StandardScaler()
    scaled_coordinates = scaler.fit_transform(coordinates)

    eps, min_samples = optimize_dbscan_params(scaled_coordinates)

    clustered_data = perform_clustering(data, eps, min_samples)

    hotspots = clustered_data[clustered_data['cluster'] != -1]
    print(f"Detected {len(hotspots)} hotspots in {len(set(hotspots['cluster']))} clusters.")

    m = create_map(clustered_data)
    m.save('hotspot_map.html')
    print("Hotspot map created and saved as 'hotspot_map.html'.")

    # Delay the next run for a fixed period (e.g., 10 minutes)
    time.sleep(600)