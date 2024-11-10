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





