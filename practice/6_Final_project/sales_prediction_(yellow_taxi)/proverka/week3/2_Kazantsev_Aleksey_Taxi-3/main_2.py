import multiprocessing as mp
from collections import namedtuple
from datetime import datetime
from pathlib import Path
import pyproj  # библотека для работы с картографическими данными
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm_notebook as log_progress
from bokeh import plotting as bk
from bokeh.models.tiles import WMTSTileSource as TileSource
from bokeh.io import show
from bokeh.palettes import Inferno8 as palette
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LogColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
)

DATA_ROOT = f'data'
TEST = False
NUM_CHUNKS = 10
WORLD = pyproj.Proj(init='epsg:4326')
MERCATOR = pyproj.Proj(init='epsg:3857')
WIDTH = 700

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
    "201601": "data/yellow_tripdata_2016-01.csv",
    "201602": "data/yellow_tripdata_2016-02.csv",
    "201603": "data/yellow_tripdata_2016-03.csv",
    "201604": "data/yellow_tripdata_2016-04.csv",
    "201605": "data/yellow_tripdata_2016-05.csv",
    "201606": "data/yellow_tripdata_2016-06.csv"
}

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


def save_data(folder, file_name, data):
    print('[{}] Saving {}...'.format(str(datetime.now()), file_name))
    data_path = Path(DATA_ROOT)
    folder_path = data_path / folder
    folder_path.mkdir(exist_ok=True, parents=True)
    data.to_csv(str(folder_path / (file_name + '.csv')), index=False)


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


def chunck_generator(path, header=False, chunk_size=10**5):
    """
    Генератор, считывает из файла данные кусками
    Inputs:
    - path: путь к файлу
    - header: признак загрузки заголовка
    - chunk_size: размер куска файла
    """

    columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime',
                'pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude']

    for chunk in pd.read_csv(path,
                             delimiter=',',
                             iterator=True,
                             chunksize=chunk_size,
                             #names=columns,
                             parse_dates=['tpep_pickup_datetime',
                                          'tpep_dropoff_datetime'],
                             dtype={'VendorID': 'int8',
                                    'passenger_count': 'int8',
                                    'RatecodeID': 'int8',
                                    'payment_type': 'int8'
                                    }
                            ):
        yield (chunk)


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


def overlay_data(map, regions, data=None):
    source = ColumnDataSource(data=dict(
        left=regions.lower_left_longitude.values,
        right=regions.upper_right_longitude.values,
        top=regions.upper_right_latitude.values,
        bottom=regions.lower_left_latitude.values,
        region=regions.index.values,
        mean=data
    ))

    color_mapper = LogColorMapper(palette=palette)

    map.quad('left', 'right', 'top', 'bottom', source=source,
             fill_color={'field': 'mean', 'transform': color_mapper},
             fill_alpha=0.4, line_color="white", line_width=0.5, line_alpha=0.)

    color_bar = ColorBar(color_mapper=color_mapper,
                         major_label_text_font_size="5pt",
                         ticker=BasicTicker(desired_num_ticks=len(palette)),
                         formatter=PrintfTickFormatter(format="%d%%"),
                         label_standoff=6, border_line_color=None,
                         location=(0, 0))

    map.add_layout(color_bar, 'right')


    hover = map.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [
        ("Region id", "@region"),
        ("Mean pickups", "@mean"),
        ("(Long, Lat)", "($x, $y)"),
    ]

    # Empire State Building
    # Latitude: 40.748817
    # Longitude: -73.985428
    ESB_POINT = Point(
        40.748817,
        -73.985428
    )

    # Statue of Liberty
    # Latitude: 40.6892494
    # Longitude: -74.0445004
    SOL_POINT = Point(
        40.6892494,
        -74.0445004
    )

    ESB_POINT = convert_point(ESB_POINT)
    SOL_POINT = convert_point(SOL_POINT)

    map.inverted_triangle(ESB_POINT.longitude, ESB_POINT.latitude, size=8,
                          color="blue", alpha=0.8)
    map.inverted_triangle(SOL_POINT.longitude, SOL_POINT.latitude, size=8,
                          color="red", alpha=0.8)


def get_figure(box, width=WIDTH, tools='pan, wheel_zoom, hover', active_scroll='wheel_zoom'):
    box = convert_box(box)
    y_min = box.lower_left.latitude
    y_max = box.upper_right.latitude
    x_min = box.lower_left.longitude
    x_max = box.upper_right.longitude
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    height = int(width / (x_max - x_min) * (y_max - y_min))

    figure = bk.figure(
        tools=tools,
        active_scroll=active_scroll,
        plot_width=width, plot_height=height,
        x_range=x_range, y_range=y_range,
        toolbar_location='above',
    )
    figure.axis.visible = True
    return figure


def get_map(box, tiles=None, width=WIDTH, tools='pan, wheel_zoom, hover', active_scroll='wheel_zoom'):
    """
    Получение и отображение карты с помощью библиотеки bokeh
    """
    figure = get_figure(box, width=width, tools=tools, active_scroll=active_scroll)
    if tiles:
        figure.add_tile(tiles)
    return figure


def main():
    # загружаем разбивку по регионам, первая колонка #-региона будет индексом
    regions = pd.read_csv(data_files['regions'], sep=';', index_col=0)
    # переведем регионы в более удобный вид
    regions = convert_regions_features_table(regions)


if __name__ == "__main__":
    mp.set_start_method('fork')
    main()
