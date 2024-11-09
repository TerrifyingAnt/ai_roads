import geopandas as gpd
from data_processing import filter_places, calculate_population, assign_routes_to_all_houses, check_apartements_count
import logging
import time
from utils.poi_type import PlaceType
from nearest_street import create_graph, find_shortest_path, process_routes


logging.basicConfig(level=logging.INFO)

start_time = time.time()
logging.info(f"Начало загрузки данных")

# Загружаем файлы с домами и улицами
houses = gpd.read_file(r"Векторные данные\Дома_исходные.shx").to_crs(epsg=4326)
streets = gpd.read_file(r"Векторные данные\Streets_исходные.shx").to_crs(epsg=4326)[lambda data: data['Foot'] == 1]
metro = gpd.read_file(r"Векторные данные\Выходы_метро.shx").to_crs(epsg=4326)
stations = gpd.read_file(r"Векторные данные\Остановки_ОТ.shx").to_crs(epsg=4326)

end_time = time.time()
logging.info(f"Загрузка завершена:  {round(end_time - start_time, 2)}")


start_time = time.time()
logging.info(f"Начало фильтрации данных")

filtered_houses = filter_places(houses, PlaceType.home)
filtered_poi = filter_places(houses, PlaceType.poi)
print(filtered_poi)
end_time = time.time()
logging.info(f"Данные отфильтрованы: {round(end_time - start_time, 2)}")

start_time = time.time()
logging.info(f"Начало проверки данных")

filtered_houses = check_apartements_count(filtered_houses)

end_time = time.time()
logging.info(f"Данные проверены: {round(end_time - start_time, 2)}")

start_time = time.time()
logging.info("Начало дополнения данных")
# Применение функции
houses_with_population = calculate_population(filtered_houses, mean=2, std_dev=1)
# Применение функции
houses = assign_routes_to_all_houses(houses_with_population, mean=1, std_dev=0.2)

# Вывод результатов
end_time = time.time()
logging.info(f"Данные дополнены: {round(end_time - start_time, 2)}")

start_time = time.time()
logging.info(f"Начало создания графа путей")
path_graph = create_graph(houses, metro, streets)

end_time = time.time()
logging.info(f"Граф создан: {round(end_time - start_time, 2)}")


# Пример использования:
# houses_with_routes - датафрейм, который был получен ранее после выполнения assign_routes_to_all_houses
# G - граф, созданный через create_graph

popular_routes = process_routes(houses, metro, stations, filtered_poi, path_graph)

# Выводим наиболее популярные маршруты и количество людей, которые их используют
for route, count in popular_routes:
    logging.info(f"Маршрут: {route}, Число людей: {count}")







