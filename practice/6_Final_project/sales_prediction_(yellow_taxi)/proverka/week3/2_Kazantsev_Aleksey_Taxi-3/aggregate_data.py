import argparse
from pathlib import Path
from main_2 import *
from tqdm import tqdm


# оставлю этот код для будущих поколений
def compute_chunk(year, month, ag_type, regions, chunk):
    #  проводим фильтрацию данных
    chunk = clean_taxi_data(chunk)

    # предобработка данных
    chunk = convert_taxi_features_table(chunk)

    # агрегация данных
    if ag_type == 'month':
        aggregate = init_aggregated_df_by_month(year, month, regions)
    else:
        aggregate = init_aggregated_df_by_day(year, month, regions)

    aggregate = aggregate_taxi_data(regions, chunk, aggregate, ag_type)
    return chunk, aggregate


# считаем пока немного данных для анализа
def read_data(filename, regions, year, month, ag_type='month', chunk_size=10):
    print('Read data file - ', filename)
    data_chunk = chunck_generator(filename, header=False, chunk_size=chunk_size)
    funclist = []

    num_cpu = mp.cpu_count() - 2 if TEST is not True else 1
    with mp.Pool(processes=num_cpu) as pool:
        # считываем все данные
        if TEST:
            f = pool.apply_async(compute_chunk, args=[year, month, ag_type, regions, next(data_chunk)])
            funclist.append(f)
            for _ in range(NUM_CHUNKS-1):
                f = pool.apply_async(compute_chunk, args=[year, month, ag_type, regions, next(data_chunk)])
                funclist.append(f)
        else:
            for chunk in tqdm(data_chunk, desc='Chunking...'):
                f = pool.apply_async(compute_chunk, args=[year, month, ag_type, regions, chunk])
                funclist.append(f)

        # добавляем результат
        data = pd.DataFrame()

        timed_out_results = []
        if ag_type == 'month':
            aggregate = init_aggregated_df_by_month(year, month, regions)
        else:
            aggregate = init_aggregated_df_by_day(year, month, regions)

        for f in tqdm(funclist, total=len(funclist), desc='Aggregate'):

            try:
                table, agg = f.get()
                # agg_data = np.sum([aggregate.iloc[:,2:].values, agg.iloc[:,2:].values], axis=0)
                aggregate.iloc[:, 2:] += agg.iloc[:, 2:]
                data = pd.concat([data, table], ignore_index=True)
            except mp.TimeoutError:
                timed_out_results.append(f)

    return data, aggregate, timed_out_results


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--root', type=str, default='data', help='data root')
    arg('--filename', type=str, help='filename to aggregate')
    arg('--year', type=str, default='2016', help='year to aggregate')
    arg('--month', type=str, default='01', help='month to aggregate')
    args = parser.parse_args()

    data_path = Path(args.root)

    # загружаем разбивку по регионам, первая колонка #-региона будет индексом
    region_path = data_path / 'regions.csv'
    regions = pd.read_csv(region_path, sep=';', index_col=0)
    # переведем регионы в более удобный вид
    regions = convert_regions_features_table(regions)
    # считываем данные
    data_path = data_path / args.filename
    data, agregate_df, errors = read_data(data_path, regions,
                                          args.year, args.month, 'hour', 10**6)
    # сохраняем
    save_data('aggregate', args.year + args.month, agregate_df)

if __name__ == "__main__":
    main()
