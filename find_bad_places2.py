import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from street_graph import (create_graph, add_places_to_graph, find_alternative_for_overloaded_routes, summarize_traffic_data, plot_alternative_routes, 
                   assign_routes_to_population, calculate_population_loads, update_weights, plot_heatmap, add_population_column_to_houses )

# --- Load and preprocess data ---
house_path = "Векторные данные/Дома_исходные.shp"
bus_path = "Векторные данные/Остановки_ОТ.shp"
files = {"Streets_исходные": "Векторные данные/Streets_исходные.shp"}

houses = gpd.read_file(house_path).to_crs(epsg=4326)

houses["Apartments"] = houses["Apartments"].fillna(0) 
buses = gpd.read_file(bus_path).to_crs(epsg=4326)
streets = gpd.GeoDataFrame(pd.concat([
    gpd.read_file(path).to_crs(epsg=4326).query("Foot == 1")
    for path in files.values()
], ignore_index=True)).loc[lambda df: df.geometry.type == 'LineString']

point = gpd.GeoDataFrame(geometry=[Point(37.495, 55.555)], crs="EPSG:4326").to_crs(epsg=3857)
radius = 1500

houses = houses.to_crs(epsg=3857)
buses = buses.to_crs(epsg=3857)
streets = streets.to_crs(epsg=3857)

houses = houses[houses.geometry.distance(point.geometry.iloc[0]) <= radius]
buses = buses[buses.geometry.distance(point.geometry.iloc[0]) <= radius]
streets = streets[streets.geometry.distance(point.geometry.iloc[0]) <= radius]

houses = houses.to_crs(epsg=4326)
buses = buses.to_crs(epsg=4326)
streets = streets.to_crs(epsg=4326)

# --- Calculate population in each house ---
houses = add_population_column_to_houses(houses)  # Добавление столбца 'Total_People'

# --- Create graph and add places ---
G, nodes = create_graph(streets)
node_coords = np.array(nodes)
tree = cKDTree(node_coords)

add_places_to_graph(houses, G, tree, node_coords, 'house')
add_places_to_graph(buses, G, tree, node_coords, 'bus_stop')

# --- Assign routes and calculate loads based on population ---
route_distribution = assign_routes_to_population(G, houses, buses, tree, node_coords)
edge_loads = calculate_population_loads(G, route_distribution)


# --- Update weights based on loads and visualize heatmap ---
update_weights(G, edge_loads)

summary = summarize_traffic_data(G, edge_loads, route_distribution, buses)

# Находим альтернативные маршруты для перегруженных участков
alternative_routes = find_alternative_for_overloaded_routes(G, route_distribution, edge_loads, houses)

# Строим график с альтернативными маршрутами
plot_alternative_routes(G, alternative_routes)



#plot_heatmap(G, edge_loads)

