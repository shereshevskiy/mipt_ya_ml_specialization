import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product as prd
from glob import glob
from tqdm import tqdm_notebook as log_progress
from scipy import stats
from statsmodels.tsa import seasonal, stattools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics import tsaplots


def load_data(root_dir='data/aggregate/'):
    files = sorted(glob(root_dir + '*'))
    file_names = [x.split('/')[-1] for x in files]

    data_df = pd.DataFrame()
    for f in log_progress(files, total=len(files)):
        df = pd.read_csv(f, parse_dates=['date'])
        data_df = data_df.append(df, ignore_index=True)

    return data_df, file_names


def generate_sin_reg(T, K, period_len=168):
    """
    @brief      Создает регриссионные признаки моделируемого ряда по синусоиде

    @param      T  Длинна моделируемого ряда
    @param      K  Размер
    @param      period_len  Длительность в часах

    @return     регрессионные признаки
    """
    data = np.array([np.sin((x*2*np.pi*np.arange(1, T+1))/period_len) for x in np.arange(1, K+1)])
    return data


def generate_cos_reg(T, K, period_len=168):
    """
    @brief      Создает регриссионные признаки моделируемого ряда
                по косинусоиде

    @param      T  Длинна моделируемого ряда
    @param      K  Размер
    @param      period_len  Длительность в часах

    @return     регрессионные признаки
    """
    data = np.array([np.cos((x*2*np.pi*np.arange(1, T+1))/period_len) for x in np.arange(1, K+1)])
    return data
