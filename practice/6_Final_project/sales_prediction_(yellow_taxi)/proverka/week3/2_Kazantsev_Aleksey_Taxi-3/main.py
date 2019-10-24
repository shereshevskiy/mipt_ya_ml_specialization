from datetime import datetime
from tqdm import tqdm_notebook as log_progress
from tqdm import tqdm, tqdm_pandas

import multiprocessing as mp

from collections import Counter, namedtuple, defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.ndimage.filters import convolve
from matplotlib import pyplot as plt
import seaborn as sns
import pyproj  # библотека для работы с картографическими данными

from bokeh import plotting as bk
from bokeh.models.tiles import WMTSTileSource as TileSource

import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import inferno, viridis, Hot, Greys9
from datashader.bokeh_ext import InteractiveImage

import imageio

# зададим немного вспомогательных объектов для удобства работы с картой
Point = namedtuple(
    'Point',
    ['latitude', 'longitude']
)

BoundingBox = namedtuple(
    'BoundingBox',
    ['lower_left', 'upper_right']
)

data_files = {
    "regions": "data/regions.csv",
    "200905": "data/yellow_tripdata_2009-05.csv",
    "201605": "data/yellow_tripdata_2016-05.csv"
}


WIDTH = 700

KERNEL = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

# Инициализируем некоторые константы
# источники карты OpenStreetMap
LIGHT_CARTO = TileSource(
    url='http://tiles.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png'
)
DARK_CARTO = TileSource(
    url='http://tiles.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'
)

# координаты расположения города Нью Йорка
NEW_YORK_BOX = BoundingBox(
    Point(40.496119999999998, -74.255589999999998),
    Point(40.915529999999997, -73.700009999999992)
)

# вспомогательные функции для работы с большим объемом данных


def chunck_generator(path, header=False, chunk_size=10**5):
    """
    Генератор, считывает из файла данные кусками
    Inputs:
    - path: путь к файлу
    - header: признак загрузки заголовка
    - chunk_size: размер куска файла
    """

    for chunk in pd.read_csv(path,
                             delimiter=',',
                             iterator=True,
                             chunksize=chunk_size,
                             parse_dates=['tpep_pickup_datetime',
                                          'tpep_dropoff_datetime'],
                             dtype={'VendorID': 'int8',
                             'passenger_count': 'int8',
                              'RatecodeID': 'int8',
                              'payment_type': 'int8'
                              }):
        yield (chunk)


def convolve_aggregate(aggregate, kernel=KERNEL):
    coords = aggregate.coords
    aggregate = convolve(aggregate, kernel)
    return xr.DataArray(aggregate, coords)


def decorate_callback(callback):
    def function(x_range, y_range, width, height, **kwargs):
        canvas = ds.Canvas(
            plot_width=width, plot_height=height,
            x_range=x_range, y_range=y_range
        )
        return callback(canvas, **kwargs)
    return function


def overlay_image(map, callback, **kwargs):
    callback = decorate_callback(callback)
    return InteractiveImage(map, callback, **kwargs)

WORLD = pyproj.Proj(init='epsg:4326')
MERCATOR = pyproj.Proj(init='epsg:3857')


def convert_point(point, source=WORLD, target=MERCATOR):
    """
    @brief      Преобразование координат  формата EPSG: 4326
    в Open Street Map EPSG 3857

    @param      point   Координаты точки
    @param      source  Из какой систему координат
    @param      target  В какую систему координат

    @return     преобразованная точка
    """
    try:
        longitude, latitude = pyproj.transform(
            source, target,
            point.longitude, point.latitude
        )
    except RuntimeError:
        raise ValueError(point)
    return Point(latitude, longitude)


def convert_box(box):
    return BoundingBox(
        convert_point(box.lower_left),
        convert_point(box.upper_right)
    )


def convert_regions_features_table(table, convert=True):

    lower_left = Point(
        table['south'].values,
        table['west'].values
    )
    if convert:
        lower_left = convert_point(lower_left, source=WORLD, target=MERCATOR)

    upper_right = Point(
        table['north'].values,
        table['east'].values
    )
    if convert:
        upper_right = convert_point(upper_right, source=WORLD, target=MERCATOR)

    boxes = BoundingBox(lower_left, upper_right)

    coords = {
        'region': {
            'number': list(table.index.values),
            'box': ['lower_left', 'upper_right']
        }
    }
    # df.index.values
    dims = ('region', 'box')

#    return xr.DataArray(box, dims=('box', 'lower_left', 'upper_right',
#                                   'latitude', 'longitude'))
    # table['box'] = box
    table['lower_left_latitude'] = lower_left.latitude
    table['lower_left_longitude'] = lower_left.longitude
    table['upper_right_latitude'] = upper_right.latitude
    table['upper_right_longitude'] = upper_right.longitude
    return table


def convert_taxi_features_table(table):
    """
    @brief      Предобработка данных

    @param      table  The table

    @return     { description_of_the_return_value }
    """
    point = Point(
        table.pickup_latitude.values,
        table.pickup_longitude.values
    )
    point = convert_point(point)
    table['pickup_latitude'] = point.latitude
    table['pickup_longitude'] = point.longitude

    point = Point(
        table.dropoff_latitude.values,
        table.dropoff_longitude.values
    )
    point = convert_point(point)
    table['dropoff_latitude'] = point.latitude
    table['dropoff_longitude'] = point.longitude

    return table


def clean_taxi_data(table):
    """
    @brief      Фильтрация данных
    - Отбросить минуты и секунды во времени начала поездки
    Удалить поездки с:
    - нулевой длительностью
    - нулевым количеством пассажиров
    - нулевым расстоянием поездки по счётчику
    - координатами начала, не попадающими в прямоугольник Нью-Йорка

    @param      table  Данных, котороые надо обработать

    @return     обработанные данные
    """
    def clean_datetime64(timestamp):
        timestamp -= np.timedelta64(timestamp.minute, 'm')
        timestamp -= np.timedelta64(timestamp.second, 's')

        return timestamp

    # убираем тех кто не вписался в регион
    drop_idx = table.index[(table['pickup_longitude'] < NEW_YORK_BOX.lower_left.longitude) | (table['pickup_longitude'] > NEW_YORK_BOX.upper_right.longitude) | (table['pickup_latitude'] < NEW_YORK_BOX.lower_left.latitude) | (table['pickup_latitude'] > NEW_YORK_BOX.upper_right.latitude)]
    table.drop(list(drop_idx), inplace=True)

    # без пассажиров и нулевым расстоянием
    drop_idx = table.index[(table['passenger_count'] == 0) | (table['trip_distance'] == 0)]
    table.drop(list(drop_idx), inplace=True)

    # с нулевым временем поездки
    drop_idx = table.index[(table['tpep_dropoff_datetime'] - table['tpep_pickup_datetime']) == "00:00:00"]
    table.drop(list(drop_idx), inplace=True)

    # убираем минуты и секунды
    table['tpep_pickup_datetime'] = table['tpep_pickup_datetime'].dt.floor(freq='h')


    return table


def init_aggregated_df_by_month(year, month, regions):
    '''
    Подготавливаем таблицу для агрегации данных по часам и регионам
    в разрезе месяца и года
    '''
    columns = ['yearmonth', 'region']
    columns = np.concatenate((columns, [str(np.timedelta64(i, 'h')) for i in range(24)]))

    data = []
    date = np.datetime64('%s-%s'%(year, month))
    for ind in regions.index.values:
        data.append(np.concatenate(([date, ind], [0 for _ in range(24)])))
    return pd.DataFrame(data, columns=columns)


def init_aggregated_df_by_day(year, month, regions):
    '''
    Подготавливаем таблицу для агрегации данных по часам и регионам
    в разрезе месяца и года
    '''
    date_from = np.datetime64('%s-%s' % (year, month))
    date_to = date_from + np.timedelta64(1, 'M')

    month_days = np.arange(date_from,
                           date_to, dtype='datetime64[D]')
    columns = ['date', 'region']
    columns = np.concatenate((columns, [str(np.timedelta64(i, 'h')) for i in range(24)]))

    data = []
    for day in month_days:
        for ind in regions.index.values:
            data.append(np.concatenate(([day, ind], [0 for _ in range(24)])))

    return pd.DataFrame(data, columns=columns)


def aggregate_taxi_data(regions, table, aggregate, ag_type='month'):
    '''
    # определять регион будем из таблицы статистик
    # проставим из какого региона была поездка
    def get_region_id(trip):
        return regions[
                (trip[0] >= regions['lower_left_longitude']) &
                (trip[0] <= regions['upper_right_longitude']) &
                (trip[1] >= regions['lower_left_latitude']) &
                (trip[1] <= regions['upper_right_latitude'])
                ].index[0]

    trips = table[['pickup_longitude', 'pickup_latitude']].values
    table['from_region'] = np.apply_along_axis(get_region_id, 1, trips)
    '''
    # Регионы разбиты в системе
    #  X: слева направа
    #  Y: снизу вверх

    # с юга на север
    bins_lat = np.concatenate([regions['lower_left_latitude'][:1].values,
                              regions['upper_right_latitude'][:50].values])

    # с запада на восток
    bins_lon = np.concatenate([regions['lower_left_longitude'][:1].values,
                              regions['upper_right_longitude'][::50].values])

    # группируем данные по часам из даты
    if ag_type == 'month':
        grouped_data = table.groupby(table['tpep_pickup_datetime'].map(lambda h: h.hour))
    else:
        # группируем по датам
        grouped_data = table.groupby(table['tpep_pickup_datetime'])

    # теперь аггрегируем по каждой группе днные по регионам
    for name, group in grouped_data:
        # считаем количество поездок из каждого региона
        # считаем статистику по Регионам
        ret = stats.binned_statistic_2d(group['pickup_longitude'].values,  # X
                                        group['pickup_latitude'].values,  # Y
                                        None,
                                        'count', bins=[bins_lon, bins_lat],
                                        expand_binnumbers=True)

        # пишем статистику по регионам в разрезе часов
        if ag_type == 'month':
            hour_id = str(np.timedelta64(int(name), 'h'))
            aggregate[hour_id] = ret.statistic.flatten()
        else:
            date_idx = pd.DatetimeIndex([np.datetime64(name)])[0]
            date = date_idx.floor(freq='D')
            hour_id = str(np.timedelta64(int(date_idx.hour), 'h'))
            aggregate.loc[aggregate['date'] == date, hour_id] = ret.statistic.flatten()

    '''
    # статистика по часам
    pickup_datetime_idx = pd.DatetimeIndex(table['tpep_pickup_datetime'])
    # тут берем только диагональные элементы
    ret_hour = stats.binned_statistic_2d(pickup_datetime_idx.hour,
                                         pickup_datetime_idx.hour,
                                         None,
                                         statistic='count', bins=24)

    # берем все записи по региону и считаем по ним статистику по часам
    '''
    return aggregate


def get_box_canvas(box, width=WIDTH):
    box = convert_box(box)
    y_min = box.lower_left.latitude
    y_max = box.upper_right.latitude
    x_min = box.lower_left.longitude
    x_max = box.upper_right.longitude
    x_range = (x_min+17000, x_max-17000)
    y_range = (y_min+17000, y_max-17000)
    height = int(width / (x_max - x_min) * (y_max - y_min))
    return ds.Canvas(
            plot_width=width, plot_height=height,
            x_range=x_range, y_range=y_range
        )


def get_figure(box, width=WIDTH):
    box = convert_box(box)
    y_min = box.lower_left.latitude
    y_max = box.upper_right.latitude
    x_min = box.lower_left.longitude
    x_max = box.upper_right.longitude
    x_range = (x_min+17000, x_max-17000)
    y_range = (y_min+17000, y_max-17000)
    height = int(width / (x_max - x_min) * (y_max - y_min))

    figure = bk.figure(
        tools='pan, wheel_zoom',
        active_scroll='wheel_zoom',
        plot_width=width, plot_height=height,
        x_range=x_range, y_range=y_range,
    )
    figure.axis.visible = True
    return figure


def get_map(box, tiles, width=WIDTH):
    """
    Получение и отображение карты с помощью библиотеки bokeh
    """
    figure = get_figure(box, width=width)
    figure.add_tile(tiles)
    return figure
