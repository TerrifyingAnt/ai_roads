import geopandas as gpd
import matplotlib.pyplot as plt
from utils.poi_type import PlaceType
from utils.constants import poi_type, poi_purpose, home_type, home_purpose, routes
import logging
import numpy as np
import random


def check_apartements_count(houses: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Обновле количества квартир для домов"""
    # Проверка наличия нужных столбцов
    if 'Apartments' in houses.columns and 'Elevation' in houses.columns and 'Type' in houses.columns:
       
        # Рассчитываем среднее количество квартир на этаж для типов зданий, кроме "Частных домов"
        known_apartments = houses[
            (houses['Apartments'].notna()) &
            (houses['Elevation'].notna()) &
            (houses['Type'] != 'Частные дома')
        ]
        known_apartments['Apartments_per_floor'] = known_apartments['Apartments'] / known_apartments['Elevation']
        avg_apartments_per_floor = known_apartments['Apartments_per_floor'].mean()
        
        # Функция для заполнения пропусков в зависимости от типа здания
        def fill_apartments(row):
            if np.isnan(row['Apartments']):  # Если значение пропущено
                if row['Type'] == 'Частные дома':
                    return 1  # Для частных домов ставим 1
                elif row['Type'] in ['Дома_новостройки', 'Жилые дома'] and not np.isnan(row['Elevation']):
                    return avg_apartments_per_floor * row['Elevation']  # Для остальных типов используем среднее количество квартир на этаж
            return row['Apartments']  # Если значение не пропущено, оставляем его

        # Применяем функцию для заполнения пропусков
        houses['Apartments'] = houses.apply(fill_apartments, axis=1)
        logging.info("--> Обновлено количество квартир в домах")
    else:
        logging.info("--> В данных отсутствуют необходимые столбцы 'Apartments', 'Elevation' или 'Type'.")
    
    return houses


def filter_places(geo_data: gpd.GeoDataFrame, place_type: PlaceType) ->  gpd.GeoDataFrame:
    """Фильтрация строк для нахождения входных\выходных мест в маршрутах"""
    if place_type == PlaceType.home:
        types = geo_data[geo_data["Type"].isin(home_type)]
        purposes = geo_data[geo_data["Purpose"].isin(home_purpose)]
    else:
        types = geo_data[geo_data["Type"].isin(poi_type)]
        purposes = geo_data[geo_data["Purpose"].isin(poi_purpose)]
        print(types)
        print(purposes)

    filtered_houses = set(zip(types["Type"], purposes["Purpose"]))

    # Фильтрация строк, где (Type, Purpose) содержатся в filtered_houses
    filtered_rows = geo_data[geo_data.apply(lambda row: (row['Type'], row['Purpose']) in filtered_houses, axis=1)]
    if place_type == PlaceType.home:
        logging.info(f"--> Найдены жилые дома")
    else:
        logging.info(f"--> Найдены точки интереса")
    return filtered_rows

# Функция для расчета количества людей в одной квартире
def calculate_people_in_apartment(mean=2, std_dev=1):
    """
    Генерирует случайное количество людей, проживающих в одной квартире.
    Параметры:
    - mean: среднее количество людей в квартире
    - std_dev: стандартное отклонение для нормального распределения
    Возвращает:
    - Количество людей в квартире (1-4)
    """
    # Генерация случайного количества людей в квартире с нормальным распределением
    people = int(np.random.normal(mean, std_dev))
    
    # Ограничиваем количество людей от 1 до 4
    if people < 1:
        people = 1
    elif people > 4:
        people = 4
    
    return people

# Функция для расчета общего количества людей в доме
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
    # Получаем количество квартир в доме
    
    apartments = int(house['Apartments'])
    
    # Рассчитываем общее количество людей в доме
    total_people = sum([calculate_people_in_apartment(mean, std_dev) for _ in range(apartments)])
    
    return total_people


# Основная функция для расчета людей по всем домам
def calculate_population(houses, mean=2, std_dev=1):
    """
    Для каждого дома в датафрейме houses рассчитывает общее количество людей.
    Параметры:
    - houses: GeoDataFrame с данными о домах
    - mean: среднее количество людей в квартире
    - std_dev: стандартное отклонение для нормального распределения
    Возвращает:
    - GeoDataFrame с новым столбцом 'Total_People', содержащим количество людей в доме
    """
    # Проверка наличия столбца 'Apartments'
    if 'Apartments' in houses.columns:
        # Применяем расчет для всех домов
        houses['Total_People'] = houses.apply(lambda row: calculate_total_people_in_house(row, mean, std_dev), axis=1)
    else:
        print("В данных отсутствует столбец 'Apartments'.")
    
    return houses

# Функция для генерации маршрута для человека
def generate_route(person_type, is_child=False, mean=1, std_dev=0.2):
    """
    Генерирует случайный маршрут для человека, исходя из его типа.
    - person_type: тип человека ('взрослый', 'школьник', 'маленький ребенок', 'пенсионер')
    - is_child: если это ребенок, то определяет, идет ли он в детский сад
    - routes: доступные маршруты
    - mean и std_dev: параметры нормального распределения для случайного выбора
    """
    # if person_type == 'маленький ребенок':
    #     # Маленькие дети идут в детский сад, и их должен кто-то вести
    #     route = random.choice(['Дошкольные', 'Метро', 'Остановка'])
    # el
    if person_type == 'школьник':
        # Школьники могут идти в школу или к метро, либо на остановку
        route = random.choice(routes)  # Выбираем случайный маршрут
    elif person_type == 'взрослый':
        # Взрослые могут использовать разные маршруты
        route = random.choice(routes)  # Выбираем случайный маршрут
    else:
        route = random.choice(routes)

    return route

# Функция для распределения людей по категориям с учетом их типа и направления
def assign_routes_to_house(house, routes, mean=1, std_dev=0.2):
    """
    Распределяет людей по маршрутам в зависимости от их типа (взрослый, школьник, маленький ребенок, пенсионер)
    Параметры:
    - house: строка из датафрейма с данными о доме
    - routes: доступные маршруты
    - mean, std_dev: параметры нормального распределения
    Возвращает:
    - Словарь с маршрутом для каждого человека
    """
    assigned_routes = []  # Список маршрутов для всех людей в доме
    
    # Получаем количество квартир и общее количество людей в доме
    total_people = house['Total_People']
    
    # Распределение людей по категориям
    child_percent = 0.15  # 15% детей и пенсионеров
    adult_personal_transport_percent = 0.30  # 30% взрослых с личным транспортом
    adult_public_transport_percent = 0.45  # 45% взрослых с общественным транспортом
    adult_car_sharing_percent = 0.10  # 10% взрослых, использующих каршеринг
    
    # Количество людей по категориям
    children_and_seniors_count = int(total_people * child_percent)
    adults_personal_transport_count = int(total_people * adult_personal_transport_percent)
    adults_public_transport_count = int(total_people * adult_public_transport_percent)
    adults_car_sharing_count = int(total_people * adult_car_sharing_percent)
    
    # Генерация маршрутов для детей (маленькие дети идут в детский сад, а взрослые ведут их)
    for _ in range(children_and_seniors_count // 2):  # Каждого ребенка ведет взрослый
        assigned_routes.append(generate_route('маленький ребенок', routes))
        assigned_routes.append(generate_route('взрослый', routes))  # Взрослый ведет ребенка
    
    # Генерация маршрутов для школьников
    for _ in range(children_and_seniors_count % 2):  # Остальные дети могут быть школьниками
        assigned_routes.append(generate_route('школьник', routes))
    
    # Генерация маршрутов для взрослых с личным транспортом
    for _ in range(adults_personal_transport_count):
        assigned_routes.append(generate_route('взрослый', routes))
    
    # Генерация маршрутов для взрослых с общественным транспортом
    for _ in range(adults_public_transport_count):
        assigned_routes.append(generate_route('взрослый', routes))
    
    # Генерация маршрутов для взрослых с каршерингом
    for _ in range(adults_car_sharing_count):
        assigned_routes.append(generate_route('взрослый', routes))
    
    return assigned_routes

# Основная функция для распределения маршрутов по всем домам
def assign_routes_to_all_houses(houses, mean=1, std_dev=0.2):
    """
    Применяет функцию распределения маршрутов для всех домов в датафрейме houses
    Параметры:
    - houses: GeoDataFrame с данными о домах
    - routes: список маршрутов
    - mean, std_dev: параметры нормального распределения
    Возвращает:
    - GeoDataFrame с новыми столбцами для маршрутов
    """
    # Проверка наличия столбца 'Total_People'
    if 'Total_People' in houses.columns:
        # Применяем расчет для всех домов
        houses['Assigned_Routes'] = houses.apply(lambda row: assign_routes_to_house(row, routes, mean, std_dev), axis=1)
    else:
        print("В данных отсутствует столбец 'Total_People'.")
    
    return houses

