import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import Point, LineString
import time
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
import heapq
import progressbar


files = {
    "Streets_исходные": r"Векторные данные\Streets_исходные.shp"
}
start_time = time.time()

def create_graph(houses: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> nx.Graph:
    # Загружаем дома, метро и улицы
    # houses = gpd.read_file(house_path).to_crs(epsg=4326)
    # metroes = gpd.read_file(metro_path).to_crs(epsg=4326)  # Станции метро

    # Убедимся, что все геометрии - LineString
    streets = streets[streets.geometry.type == 'LineString']

    # Создаем граф
    G = nx.Graph()

    # Добавляем улицы в граф
    nodes = []
    edges = []
    for _, row in streets.iterrows():
        coords = list(row.geometry.coords)
        for i in range(len(coords) - 1):
            G.add_edge(coords[i], coords[i + 1], weight=Point(coords[i]).distance(Point(coords[i + 1])))
            nodes.append(coords[i])
            nodes.append(coords[i + 1])

    # Сортируем узлы
    nodes = sorted(set(nodes), key=lambda x: (x[0], x[1]))

    # Преобразуем узлы в массив для cKDTree
    node_coords = [(x, y) for x, y in nodes]
    tree = cKDTree(node_coords)

    # Рассчитываем центры домов
    houses['centroid'] = houses.geometry.centroid

    # Для каждого дома находим ближайшую точку
    for _, house in houses.iterrows():
        house_center = house['centroid']
        
        # Находим ближайшую точку на графе с использованием cKDTree
        nearest_point, nearest_distance = _find_nearest_node(house_center, tree, node_coords)
        
        # Добавляем центр дома в граф
        G.add_node((house_center.x, house_center.y), type='house')

        # Добавляем ребро между домом и ближайшей точкой на улице
        G.add_edge((house_center.x, house_center.y), nearest_point, weight=nearest_distance)
        # print((house_center.x, house_center.y))

    # Для каждой станции метро добавляем её в граф
    for _, metro in pois.iterrows():
        metro_coords = metro.geometry
        nearest_point, nearest_distance = _find_nearest_node(metro_coords, tree, node_coords)
        G.add_node((metro_coords.x, metro_coords.y), type='metro')

        # Добавляем ребро между домом и ближайшей точкой на улице
        G.add_edge((metro_coords.x, metro_coords.y), nearest_point, weight=nearest_distance)
        
    # Печать статистики
    logging.info(f"--> Граф содержит {len(G.nodes)} узлов и {len(G.edges)} ребер.")
    logging.info(f"--> Время выполнения: {time.time() - start_time:.2f} секунд")
    return G


# Функция для поиска ближайшей точки с помощью cKDTree
def _find_nearest_node(house_center, tree, node_coords):
    house_coords = (house_center.x, house_center.y)
    # Находим ближайшую точку
    distance, idx = tree.query(house_coords)
    nearest_node = node_coords[idx]
    return nearest_node, distance

# Визуализация маршрута
# fig, ax = plt.subplots(figsize=(10, 10))
# streets.plot(ax=ax, color='grey', linewidth=0.5)

# Генерация позиций узлов
def find_shortest_path(start_place: gpd.GeoDataFrame, end_place: gpd.GeoDataFrame, G: nx.Graph) -> list[tuple]:
    pos = {node: (node[0], node[1]) for node in G.nodes}
    start_node = start_place.geometry.centroid
    end_node = end_place.geometry.centroid
    # Создаем список ребер маршрута
    #start_node = (37.48997790288288, 55.553571725893136)
    #end_node = (37.470826631165075, 55.5599388466005)
    shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
    return [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]


# Функция для обработки всех маршрутов и подсчета популярных путей
def process_routes(houses, metro, stations, filtered_poi, G):
    with progressbar.ProgressBar(max_value=len(houses)) as bar:
        # Словарь для хранения путей и их популярности
        route_popularity = defaultdict(int)
        # Словарь для хранения путей
        all_paths = []

        # Проходим по всем домам и получаем маршруты
        for idx, house in houses.iterrows():
            assigned_routes = house['Assigned_Routes']  # Список маршрутов для текущего дома
            
            for route in assigned_routes:
                print(route)
                # Определяем начальную точку (дом) и конечную точку (место назначения)
                end_place = find_by_name(house, route, metro, stations, filtered_poi)
                start_place = gpd.GeoDataFrame(geometry=[Point(house['centroid'].x, house['centroid'].y)])

                end_place = gpd.GeoDataFrame(geometry=[Point(end_place['centroid'].x, end_place['centroid'].y)], crs=houses.crs)
                
                # Получаем путь для этого маршрута
                path_edges = find_shortest_path(start_place, end_place, G)
                
                # Конвертируем путь в кортеж для дальнейшего подсчета популярности
                path_tuple = tuple(path_edges)
                
                # Увеличиваем счетчик популярности для этого пути
                route_popularity[path_tuple] += 1
                all_paths.append(path_tuple)
            bar.update(idx)

        # Получаем наиболее популярные пути
        popular_routes = heapq.nlargest(10, route_popularity.items(), key=lambda x: x[1])

        # Формируем результаты
        popular_routes_with_count = [(route, count) for route, count in popular_routes]
    
    return popular_routes_with_count


def find_by_name(house: gpd.GeoDataFrame, route: tuple[str], metro: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, houses: gpd.GeoDataFrame):
    # Вычисляем центроид для дома, если он еще не рассчитан
    house_centroid = house.geometry.centroid

    if route is not None and len(route) == 1:
        if route[0] == "Метро":
            # Ищем ближайшую станцию метро
            metro['distance'] = metro.geometry.centroid.distance(house_centroid)
            if metro.empty:
                return None
            need_station = metro.loc[metro['distance'].idxmin()]
            return need_station
        elif route[0] == "Остановка":
            # Ищем ближайшую станцию (не метро)
            stations['distance'] = stations.geometry.centroid.distance(house_centroid)
            if stations.empty:
                return None
            need_station = stations.loc[stations['distance'].idxmin()]
            return need_station
        else:
            filtered_houses = houses[(houses["Type"] == route[0])]
        
        # Проверка на наличие данных в filtered_houses
        if filtered_houses.empty:
            return None

        # Если есть данные, ищем ближайший
        filtered_houses['distance'] = filtered_houses.geometry.centroid.distance(house_centroid)
        need_station = filtered_houses.loc[filtered_houses['distance'].idxmin()]
        return need_station
    else:
        # Фильтруем здания по типу и назначению
        filtered_houses = houses[(houses["Type"] == route[0])]
        print(houses)
        
        # Проверка на наличие данных в filtered_houses
        if filtered_houses.empty:
            return None

        # Если есть данные, ищем ближайший
        filtered_houses['distance'] = filtered_houses.geometry.centroid.distance(house_centroid)
        need_station = filtered_houses.loc[filtered_houses['distance'].idxmin()]
        return need_station

            