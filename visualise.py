import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Пути к данным
file_paths = {
    "House_1очередь_ЖК": r"Векторные данные\House_1очередь_ЖК.shx",
    "House_2очередь_ЖК": r"Векторные данные\House_2очередь_ЖК.shx",
    "House_3очередь_ЖК": r"Векторные данные\House_3очередь_ЖК.shx",
    "Streets_1очередь": r"Векторные данные\Streets_1очередь.shx",
    "Streets_2очередь": r"Векторные данные\Streets_2очередь.shx",
    "Streets_3очередь": r"Векторные данные\Streets_3очередь.shx",
    "Streets_исходные": r"Векторные данные\Streets_исходные.shx",
    "Выходы_метро": r"Векторные данные\Выходы_метро.shx",
    "Дома_исходные": r"Векторные данные\Дома_исходные.shx",
    "Остановки_ОТ": r"Векторные данные\Остановки_ОТ.shx"
}

# Загружаем файлы с домами и улицами
houses = gpd.read_file(r"Векторные данные\Дома_исходные.shx")
streets = gpd.read_file(r"Векторные данные\Streets_исходные.shx")

houses = houses[houses['Elevation'].notna()]

# Подсчёт строк, где в столбце 'Apartments' пустые значения (NaN)
num_empty_apartments = houses['Apartments'].isna().sum()

# Вывод результата
print("Количество строк с пустым значением в столбце 'Apartments':", num_empty_apartments)

# Создание фигуры и осей
fig, ax = plt.subplots(figsize=(12, 10))

# Условие для отображения на основе высоты
# Нормализация значений высоты для цветовой шкалы
norm = Normalize(vmin=houses['Elevation'].min(), vmax=houses['Elevation'].max())
cmap = cm.viridis  # Выбор цветовой карты

# Отображение домов с использованием градации цвета по высоте
houses.plot(ax=ax, column='Elevation', cmap=cmap, legend=True, norm=norm,
            legend_kwds={'label': "Высота зданий", 'orientation': "vertical"})

# Отображение остальных слоев данных (например, улиц и метро), если это нужно
other_layers = {
    "Streets_исходные": "gray",
    "Выходы_метро": "black",
    "Остановки_ОТ": "pink"
}

# Загрузка и отображение дополнительных слоёв данных
for layer, color in other_layers.items():
    data = gpd.read_file(file_paths[layer])
    data.plot(ax=ax, color=color, label=layer)

# Настройка заголовка и легенды
plt.title("Карта с подсветкой по высоте зданий")
plt.legend(loc="upper right")
plt.show()