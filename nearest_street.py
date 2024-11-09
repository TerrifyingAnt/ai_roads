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
import warnings
import progressbar
import networkx as nx

warnings.simplefilter(action='ignore')


files = {
    "Streets_исходные": r"Векторные данные\Streets_исходные.shp"
}
start_time = time.time()

def create_graph(houses: gpd.GeoDataFrame, filtered_poi: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> nx.Graph:
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
        G.add_node((house_center.x, house_center.y), type="house")

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

    for _, filtered_poi_1 in filtered_poi.iterrows():
        filtered_poi_center = filtered_poi_1.geometry.centroid
        nearest_point, nearest_distance = _find_nearest_node(filtered_poi_center, tree, node_coords)
        
        # Добавляем центр дома в граф
        G.add_node((filtered_poi_center.x, filtered_poi_center.y), type='poi_house_' + str(house["Type"]) + "_" + str(house["Purpose"]))

        # Добавляем ребро между домом и ближайшей точкой на улице
        G.add_edge((filtered_poi_center.x, filtered_poi_center.y), nearest_point, weight=nearest_distance)
        
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
    # pos = {node: (node[0], node[1]) for node in G.nodes}

    start_node = start_place.geometry.centroid
    end_node = end_place.geometry.centroid
    # Создаем список ребер маршрута
    start_node = (start_node.x, start_node.y)
    end_node = (end_node.x, end_node.y)
    shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
    return [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

# Функция для обработки всех маршрутов и подсчета популярных путей
def process_routes(houses, G):
        # Словарь для хранения путей и их популярности
    route_popularity = defaultdict(int)
        # Список для хранения всех маршрутов
    all_paths = []

        # Проходим по всем домам и получаем маршруты
    with progressbar.ProgressBar(max_value=len(houses) + 1) as bar:
        bar.update(0)
        for idx, house in houses.iterrows():
            assigned_routes = house['Assigned_Routes']  # Список маршрутов для текущего дома

            # Определяем начальную точку (дом) как объект Point
            start_point = Point(house['centroid'].x, house['centroid'].y)

            # Обработка каждого маршрута и количества людей
            for route_data in assigned_routes:
                route_name = route_data['Маршрут']      # Название маршрута
                print(route_name)
                people_count = route_data['Количество_людей']  # Количество людей, использующих этот маршрут

                try:
                    # Находим кратчайший путь до ближайшей точки интереса по маршруту
                    path_edges = find_shortest_path_to_poi(G, start_point, route_name)

                    # Конвертируем путь в кортеж для подсчета популярности
                    path_tuple = tuple(path_edges)

                    # Увеличиваем счетчик популярности для этого пути на количество людей
                    route_popularity[path_tuple] += people_count
                    all_paths.append((path_tuple, people_count))
                    
                except ValueError as e:
                    # Обрабатываем случаи, когда маршрут не найден
                    print(f"Маршрут не найден для дома {idx} с маршрутом {route_name}: {e}")
                    
            bar.update(idx)

    # Получаем 10 самых популярных маршрутов
    popular_routes = heapq.nlargest(10, route_popularity.items(), key=lambda x: x[1])

    # Формируем результаты
    popular_routes_with_count = [(route, count) for route, count in popular_routes]
    
    return popular_routes_with_count


# Обновленная функция для поиска кратчайшего пути до точки интереса
def find_shortest_path_to_poi(graph: nx.Graph, start_point: Point, route_name: str) -> list[tuple]:
    """
    Находит кратчайший путь до ближайшей точки интереса по маршруту в графе.
    
    :param graph: Граф NetworkX, содержащий все дома и точки интереса с их типами и целями.
    :param start_point: Точка отсчета (например, дом) в виде объекта Point.
    :param route_name: Название маршрута, соответствующее точке интереса.
    :return: Список кортежей, представляющих путь от стартовой точки до ближайшей точки интереса.
    """
    # Определяем координаты стартовой точки в графе
    start_node = (start_point.x, start_point.y)
    
    # Проверка наличия стартовой точки в графе
    if start_node not in graph:
        raise ValueError("Стартовая точка не найдена в графе.")
    temp_x = [data for node, data in graph.nodes(data=True)]
    print(temp_x)
    # Находим узлы графа, соответствующие заданному маршруту
    target_nodes = [
        node for node, data in graph.nodes(data=True)
        if str(data.get('type')).find(route_name) != -1
    ]
    
    # Проверка наличия подходящих точек
    if not target_nodes:
        raise ValueError(f"Нет точек интереса с маршрутом '{route_name}' в графе.")

    # Инициализация переменных для хранения ближайшей точки и кратчайшего пути
    shortest_path = None
    shortest_distance = float('inf')

    # Поиск кратчайшего пути до ближайшей точки интереса
    for target_node in target_nodes:
        try:
            # Находим путь и его длину от стартовой точки до текущей целевой точки
            path = nx.shortest_path(graph, source=start_node, target=target_node, weight='weight')
            path_length = nx.shortest_path_length(graph, source=start_node, target=target_node, weight='weight')
            
            # Сравниваем длину пути с текущим кратчайшим путём
            if path_length < shortest_distance:
                shortest_path = path
                shortest_distance = path_length
        except nx.NetworkXNoPath:
            # Если пути нет, пропускаем
            continue
    
    if shortest_path is None:
        raise ValueError("Кратчайший путь не найден.")
    
    # Возвращаем кратчайший путь
    return [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
 