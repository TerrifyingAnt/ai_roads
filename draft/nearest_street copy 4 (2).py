import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# Загрузка данных (аналогично)
house_path = "Векторные данные/Дома_исходные.shp"
bus_path = "Векторные данные/Остановки_ОТ.shp"
files = {"Streets_исходные": r"Векторные данные\Streets_исходные.shp"}

houses = gpd.read_file(house_path).to_crs(epsg=4326)
buses = gpd.read_file(bus_path).to_crs(epsg=4326)
streets = gpd.GeoDataFrame(pd.concat([ 
    gpd.read_file(path).to_crs(epsg=4326)[lambda data: data['Foot'] == 1]
    for path in files.values()
], ignore_index=True))
streets = streets[streets.geometry.type == 'LineString']

point = gpd.GeoDataFrame(geometry=[Point(37.495, 55.555)], crs="EPSG:4326")
radius = 1000  # 1 км = 1000 метров

houses = houses.to_crs(epsg=3857)
buses = buses.to_crs(epsg=3857)
streets = streets.to_crs(epsg=3857)
point = point.to_crs(epsg=3857)

# Выбираем объекты, находящиеся в пределах 1 км от заданной точки
houses = houses[houses.geometry.distance(point.geometry.iloc[0]) <= radius]
buses = buses[buses.geometry.distance(point.geometry.iloc[0]) <= radius]
streets = streets[streets.geometry.distance(point.geometry.iloc[0]) <= radius]

houses = houses.to_crs(epsg=4326)
buses = buses.to_crs(epsg=4326)
streets = streets.to_crs(epsg=4326)

# Создаем граф с NetworkX
def create_graph(streets):
    G = nx.DiGraph()
    nodes = set()
    for _, row in streets.iterrows():
        coords = list(row.geometry.coords)
        for start, end in zip(coords[:-1], coords[1:]):
            distance = Point(start).distance(Point(end))
            G.add_edge(start, end, weight=distance)
            G.add_edge(end, start, weight=distance)
            nodes.add(start)
            nodes.add(end)
    return G, list(nodes)

def find_nearest_node(point, tree, node_coords):
    dist, idx = tree.query((point.x, point.y))
    return tuple(node_coords[idx]), dist

# Добавление домов и остановок
def add_places_to_graph(places, G, tree, node_coords, place_type):
    for _, place in tqdm(places.iterrows(), desc=f"Adding {place_type}", total=len(places)):
        point = place.geometry.centroid if place_type == "house" else place.geometry
        nearest_node, dist = find_nearest_node(point, tree, node_coords)
        new_node = (point.x, point.y)
        G.add_edge(nearest_node, new_node, weight=dist)
        G.add_edge(new_node, nearest_node, weight=dist)
        G.nodes[new_node]['type'] = place_type

# Перевод вычислений на CPU (с использованием NumPy)
def cpu_shortest_path_usage(houses, buses, G):
    house_nodes = np.array([(c.x, c.y) for c in houses.geometry.centroid])
    bus_nodes = np.array([(g.x, g.y) for g in buses.geometry])
    
    usage = defaultdict(int)
    
    for house_node in tqdm(house_nodes, desc="Calculating paths"):
        distances = []
        for bus_node in bus_nodes:
            try:
                length = nx.shortest_path_length(G, source=tuple(house_node), target=tuple(bus_node), weight='weight')
                distances.append((bus_node, length))
            except nx.NetworkXNoPath:
                continue
        
        nearest_stops = sorted(distances, key=lambda x: x[1])[:2]
        
        for bus_node, _ in nearest_stops:
            path = nx.shortest_path(G, source=tuple(house_node), target=tuple(bus_node), weight='weight')
            for start, end in zip(path[:-1], path[1:]):
                usage[(start, end)] += 1
    
    return usage

# Визуализация результата
def plot_street_usage(streets, street_usage):
    fig, ax = plt.subplots(figsize=(12, 12))
    streets.plot(ax=ax, color='lightgray', linewidth=0.5)
    
    max_usage = max(street_usage.values())
    for (start, end), usage in street_usage.items():
        line = LineString([start, end])
        usage_norm = usage / max_usage
        gpd.GeoSeries([line]).plot(ax=ax, color=cm.viridis(usage_norm), linewidth=2)
    
    houses.plot(ax=ax, color='blue', markersize=10, label='Houses')
    buses.plot(ax=ax, color='red', markersize=10, label='Bus Stops')
    plt.legend()
    plt.show()


G, nodes = create_graph(streets)

# cKDTree и поиск ближайших точек
node_coords = np.array(nodes)
tree = cKDTree(node_coords)

add_places_to_graph(houses, G, tree, node_coords, 'house')
add_places_to_graph(buses, G, tree, node_coords, 'bus_stop')

# Вычисление использования улиц
street_usage = cpu_shortest_path_usage(houses, buses, G)

plot_street_usage(streets, street_usage)

