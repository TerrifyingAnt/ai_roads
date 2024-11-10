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

# Загрузка данных
house_path = "Векторные данные/Дома_исходные.shp"
bus_path = "Векторные данные/Остановки_ОТ.shp"
files = {"Streets_исходные": r"Векторные данные\Streets_исходные.shp"}

houses = gpd.read_file(house_path).to_crs(epsg=4326)
houses["Apartments"] = houses["Apartments"].fillna(0)
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

# Добавление домов и остановок в граф
def add_places_to_graph(places, G, tree, node_coords, place_type):
    for _, place in tqdm(places.iterrows(), desc=f"Adding {place_type}", total=len(places)):
        point = place.geometry.centroid if place_type == "house" else place.geometry
        nearest_node, dist = find_nearest_node(point, tree, node_coords)
        new_node = (point.x, point.y)
        G.add_edge(nearest_node, new_node, weight=dist)
        G.add_edge(new_node, nearest_node, weight=dist)
        G.nodes[new_node]['type'] = place_type

# Поиск маршрута от дома до ближайшей остановки
# Поиск маршрута от дома до ближайшей остановки
def find_shortest_paths_to_bus_stops(houses, buses, G):
    house_locations = []
    bus_stops = {tuple(bus.geometry.coords[0]): bus for _, bus in buses.iterrows()}
    routes = {}
    
    for house in tqdm(houses, desc="Finding shortest paths"):
        house_point = house
        house_location = (house_point[0], house_point[1])
        house_locations.append(house_location)
        
        # Находим ближайшую остановку
        nearest_bus_stop = min(bus_stops.keys(), key=lambda bus: Point(house_point).distance(Point(bus_stops[bus].geometry)))
        
        # Проверим, существует ли путь между домом и ближайшей остановкой
        if nx.has_path(G, house_location, nearest_bus_stop):
            # Находим кратчайший путь
            shortest_path = nx.shortest_path(G, source=house_location, target=nearest_bus_stop, weight='weight')
            routes[house_location] = shortest_path
        else:
            # Если пути нет, можно добавить сообщение о невозможности найти путь
            routes[house_location] = None
    
    return routes, house_locations

def find_buildings_within_1km(houses, buses, G):
    bus_stops = {tuple(bus.geometry.coords[0]): bus for _, bus in buses.iterrows()}
    buildings_near_stops = []

    for _, house in tqdm(houses.iterrows(), desc="Finding buildings within 1km"):
        house_point = house.geometry.centroid
        house_location = (house_point.x, house_point.y)
        
        # Находим ближайшую остановку
        nearest_bus_stop = min(bus_stops.keys(), key=lambda bus: Point(house_point).distance(Point(bus_stops[bus].geometry)))
        
        # Проверяем, находится ли здание в пределах 1 км от остановки
        if Point(house_point).distance(Point(bus_stops[nearest_bus_stop].geometry)) <= 1.0:
            buildings_near_stops.append(house_location)
    
    return buildings_near_stops

G, nodes = create_graph(streets)

# cKDTree и поиск ближайших точек
node_coords = np.array(nodes)
tree = cKDTree(node_coords)

add_places_to_graph(houses, G, tree, node_coords, 'house')
add_places_to_graph(buses, G, tree, node_coords, 'bus_stop')

buildings_near_stops = find_buildings_within_1km(houses, buses, G)
# Найдем маршруты и отрисуем их
routes, house_locations = find_shortest_paths_to_bus_stops(buildings_near_stops, buses, G)

# Отрисовываем карты
fig, ax = plt.subplots(figsize=(10, 10))

# Отрисовка улиц
streets.plot(ax=ax, color='lightgray', linewidth=1)

# Отрисовка домов
houses.plot(ax=ax, color='blue', markersize=10, label='Houses')

# Отрисовка остановок
buses.plot(ax=ax, color='red', markersize=50, label='Bus Stops')

# Отрисовка маршрутов
for route in routes.values():
    if route is not None:
        route_line = LineString(route)
        ax.plot(*route_line.xy, color='green', linewidth=2)

ax.legend()
plt.show()