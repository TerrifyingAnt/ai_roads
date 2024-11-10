# utils.py

import networkx as nx
from shapely.geometry import Point, LineString
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from tqdm import tqdm


# --- Create Graph from Streets ---
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

# --- Find nearest node ---
def find_nearest_node(point, tree, node_coords):
    dist, idx = tree.query((point.x, point.y))
    return tuple(node_coords[idx]), dist

# --- Add Places (houses, bus stops) to Graph ---
def add_places_to_graph(places, G, tree, node_coords, place_type):
    for _, place in tqdm(places.iterrows(), desc=f"Adding {place_type}", total=len(places)):
        point = place.geometry.centroid if place_type == "house" else place.geometry
        nearest_node, dist = find_nearest_node(point, tree, node_coords)
        new_node = (point.x, point.y)
        G.add_edge(nearest_node, new_node, weight=dist)
        G.add_edge(new_node, nearest_node, weight=dist)
        G.nodes[new_node]['type'] = place_type
        if place_type == "house":
            G.nodes[new_node]['total_people'] = place["Total_People"]
        else: 
            G.nodes[new_node]['total_people'] = 0
        

# --- Compute paths and loads ---
def compute_paths_and_loads(G, sources, targets):
    flow_distribution = {source: {target: np.random.randint(800, 1000) for target in targets} for source in sources}
    paths = {source: nx.single_source_dijkstra_path(G, source) for source in sources}

    edge_loads = {edge: 0 for edge in G.edges}
    
    for source, target_flows in tqdm(flow_distribution.items(), desc="Calculating loads"):
        if source not in paths:
            continue
        for target, flow in target_flows.items():
            if target not in paths[source]:
                continue
            path = paths[source][target]
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                edge_loads[edge] += flow
                
    return edge_loads

# --- Update Edge Weights based on Loads ---
def update_weights(G, edge_loads, capacity=300):
    for edge, load in edge_loads.items():
        weight = G[edge[0]][edge[1]]['weight']
        congestion = load / capacity
        G[edge[0]][edge[1]]['weight'] = weight * (1 + congestion * 2)

# --- Plot Heatmap for Edge Loads ---
def plot_heatmap(G, edge_loads):
    loads = np.array(list(edge_loads.values()))
    norm = plt.Normalize(vmin=0, vmax=loads.max())
    cmap = plt.cm.Reds

    fig, ax = plt.subplots(figsize=(12, 8))
    for (u, v, data) in G.edges(data=True):
        load = edge_loads[(u, v)]
        color = cmap(norm(load))
        ax.plot([u[0], v[0]], [u[1], v[1]], color=color, linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Load Intensity')
    plt.title("Heatmap of Pedestrian Congestion")
    plt.show()

# --- Calculate people in one apartment ---
def calculate_people_in_apartment(mean=2, std_dev=1):
    people = int(np.random.normal(mean, std_dev))
    return max(1, min(people, 4))  # Limit to 1-4 people

# --- Calculate total people in a house based on apartments ---
def calculate_total_people_in_house(house, mean=2, std_dev=1):
    """
    Рассчитывает общее количество людей в доме на основе количества квартир.
    Параметры:
    - house: строка датафрейма с данными о доме
    - mean: среднее количество людей в квартире
    - std_dev: стандартное отклонение для нормального распределения
    Возвращает:
    - Общее количество людей в доме
    """
    apartments = int(house['Apartments'])  # Количество квартир
    total_people = sum([calculate_people_in_apartment(mean, std_dev) for _ in range(apartments)])  # Суммируем количество людей
    return total_people

# --- Добавление столбца с количеством людей в домах ---
def add_population_column_to_houses(houses, mean=2, std_dev=1):
    """
    Для каждого дома в датафрейме houses рассчитывает общее количество людей и добавляет новый столбец 'Total_People'.
    Параметры:
    - houses: GeoDataFrame с данными о домах
    - mean: среднее количество людей в квартире
    - std_dev: стандартное отклонение для нормального распределения
    Возвращает:
    - GeoDataFrame с добавленным столбцом 'Total_People', содержащим количество людей в доме
    """
    houses['Total_People'] = houses.apply(lambda row: calculate_total_people_in_house(row, mean, std_dev), axis=1)
    return houses

# --- Calculate population for all houses in GeoDataFrame ---
def calculate_population(houses, mean=2, std_dev=1):
    if 'Apartments' in houses.columns:
        houses['Total_People'] = houses.apply(lambda row: calculate_total_people_in_house(row, mean, std_dev), axis=1)
    else:
        print("Column 'Apartments' missing in data.")
    return houses

# Внутри функции assign_routes_to_population
def assign_routes_to_population(G, houses, buses, tree, node_coords):
    """
    Назначение маршрутов для населения, идущего от домов к ближайшим остановкам.
    """
    route_distribution = {}

    # Итерация по домам
    for _, house in houses.iterrows():
        house_location = house.geometry.centroid if house.geometry.geom_type != "Point" else house.geometry
        nearest_house_node, _ = find_nearest_node(house_location, tree, node_coords)
        
        # Проверка, что house_node существует в графе
        if nearest_house_node not in G:
            continue

        # Получаем количество людей в доме
        total_people = house['Total_People'] * 0.51 / 60

        # Итерация по остановкам
        for _, bus_stop in buses.iterrows():
            # Здесь bus_stop — это строка GeoDataFrame и мы обращаемся к bus_stop.geometry
            bus_stop_location = bus_stop.geometry.centroid if bus_stop.geometry.geom_type != "Point" else bus_stop.geometry
            nearest_bus_node, _ = find_nearest_node(bus_stop_location, tree, node_coords)
            
            # Проверка, что bus_node существует в графе
            if nearest_bus_node not in G:
                continue

            # Если путь существует, добавляем информацию о маршруте
            if nearest_house_node in G and nearest_bus_node in G:
                try:
                    # Находим путь от дома до остановки
                    path = nx.shortest_path(G, source=nearest_house_node, target=nearest_bus_node, weight="weight")
                    
                    # Добавляем новый маршрут в route_distribution с ключом (nearest_house_node, nearest_bus_node)
                    route_distribution[(nearest_house_node, nearest_bus_node)] = {
                        'path': path,
                        'total_people': total_people
                    }

                except nx.NetworkXNoPath:
                    # Если нет пути, пропускаем
                    continue

    return route_distribution


def calculate_population_loads(G, route_distribution):
    """
    Рассчитывает нагрузку на ребра графа на основе распределения маршрутов от домов к остановкам.
    Каждый маршрут имеет количество людей, идущих по пути.
    """
    # Словарь для хранения нагрузки на ребра
    edge_loads = {edge: 0 for edge in G.edges}
    
    # Итерация по маршрутам в route_distribution
    for (house_node, bus_stop_node), route_info in tqdm(route_distribution.items(), desc="Calculating loads"):
        path = route_info['path']
        total_people = route_info['total_people']
        
        # Для каждого пути добавляем нагрузку только один раз для каждого ребра
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            
            # Увеличиваем нагрузку на ребро с учетом количества людей
            edge_loads[edge] += total_people
        
    return edge_loads

def summarize_traffic_data(G, edge_loads, route_distribution, buses):
    """
    Возвращает сводную аналитику по загруженности дорог, основанной на информации о маршрутах и нагрузках.
    
    Параметры:
    - G: граф, в котором происходят маршруты.
    - edge_loads: словарь, содержащий нагрузку на каждое ребро графа.
    - route_distribution: словарь маршрутов с количеством людей для каждого пути.

    Возвращает:
    - Словарь с аналитической информацией:
      1. Количество значимых объектов (остановок)
      2. Сумма людей, которые прошли по дорогам
      3. Количество перегруженных участков дорог (нагрузка > 800)
      4. Наибольший загруженный участок (длина самого длинного перегруженного участка)
    """
    # 1. Количество участков дорог
    edges = len(route_distribution)

    num_bus_stops = buses.shape[0]
    
    # 2. Сумма людей, которые прошли по дорогам
    total_people = sum(route_info['total_people'] for route_info in route_distribution.values())

    # 3. Количество перегруженных участков дорог (нагрузка на участке больше 800)
    overloaded_edges_count = sum(1 for load in edge_loads.values() if load > 40)

    # 4. Наибольший загруженный участок (длина самого длинного перегруженного участка)
    longest_overloaded_edge_length = 0
    for (u, v), load in edge_loads.items():
        if load > 40:
            # Расчет длины ребра, если его нагрузка больше 800
            edge_length = np.linalg.norm(np.array(u) - np.array(v))  # Используем евклидово расстояние для длины ребра
            if edge_length > longest_overloaded_edge_length:
                longest_overloaded_edge_length = edge_length

    # Возвращаем словарь с аналитикой
    summary = {
        "num_bus_stops": num_bus_stops,  # Количество значимых объектов (остановок)
        "total_people": total_people,    # Сумма людей, которые прошли по дорогам
        "overloaded_edges_count": overloaded_edges_count,  # Количество перегруженных участков дорог
        "longest_overloaded_edge_length": longest_overloaded_edge_length,  # Длина самого длинного перегруженного участка
        "sytem_score": (edges - overloaded_edges_count) / edges
    }
    
    return summary

def find_overloaded_edges(G, edge_loads, load_threshold):
    """
    Находит перегруженные рёбра в графе на основе порога нагрузки.
    """
    overloaded_edges = []
    for (u, v, data) in G.edges(data=True):
        if data['weight'] > load_threshold:  # Если нагрузка на ребро превышает порог
            overloaded_edges.append((u, v))
    return overloaded_edges

def is_walkable(G, edge, buildings):
    """
    Проверяет, можно ли пройти по ребру, не попадя в здание.
    """
    road_segment = LineString([edge[0], edge[1]])
    for _, building in buildings.iterrows():
        building_geom = building.geometry
        if road_segment.intersects(building_geom):  # Если дорога пересекает здание
            return False
    return True

def create_alternative_route(G, overloaded_edge, buildings, penalty_radius=50):
    """
    Создает альтернативный маршрут, отступая от перегруженного ребра.
    """
    u, v = overloaded_edge
    # Получаем координаты узлов
    u_pos, v_pos = G.nodes[u]['geometry'], G.nodes[v]['geometry']
    
    # Найдем ближайшие соседние точки для обхода перегруженного участка
    # Отступаем от перегруженного сегмента (используем простую эвристическую логику)
    u_new_point = Point(u_pos.x + 0.01, u_pos.y + 0.0001)  # Пример смещения
    v_new_point = Point(v_pos.x + 0.01, v_pos.y - 0.0001)  # Пример смещения

    # Проверяем, не пересекают ли новые участки зданий
    if is_walkable(G, (u_new_point, v_new_point), buildings):
        # Соединяем новые точки
        alternative_path = nx.shortest_path(G, source=u_new_point, target=v_new_point, weight='weight')
        return alternative_path
    return None
def find_alternative_for_overloaded_routes(G, routes, edge_loads, buildings, load_threshold=10):
    """
    Для каждого маршрута ищем альтернативные пути для перегруженных участков.
    """
    alternative_routes = {}
    for route_id, route in routes.items():
        # Убедимся, что маршрут представляет собой список узлов
        path_route = route["path"]
        if isinstance(path_route, list):
            # Перебираем ребра маршрута
            for i in range(len(path_route) - 1):
                u, v = path_route[i], path_route[i + 1]

                # Проверим, если ребро перегружено
                if (u, v) in find_overloaded_edges(G, edge_loads, load_threshold):
                    # Если перегружено, ищем альтернативный маршрут
                    alternative_route = create_alternative_route(G, [u, v], buildings)

                    if alternative_route:
                        alternative_routes[route_id] = alternative_route  # Добавляем альтернативный маршрут
                    break  # Переходим к следующему маршруту после первого найденного перегруженного участка
        else:
            print(f"Warning: Route {route_id} is not a list of nodes, skipping it.")
            continue  # Пропускаем маршрут, если он не является списком узлов

    return alternative_routes

def plot_alternative_routes(G, routes, ax=None):
    """
    Строит график с альтернативными маршрутами
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    pos = {node: (node[0], node[1]) for node in G.nodes}

    # Отобразим граф
    # nx.draw(G, pos, with_labels=True, node_size=10, font_size=6, ax=ax)

    # Отобразим альтернативные маршруты
    for route in routes:
        x = [node[0] for node in route]
        y = [node[1] for node in route]
        ax.plot(x, y, marker='o', markersize=3, label="Alternative Route", color='orange')  # Альтернативный путь

    ax.set_title("Alternative Routes")
    plt.legend()
    plt.show()



