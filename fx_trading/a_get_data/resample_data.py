import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import fx_trading.utils.constants as cst
from fx_trading.utils.data_interact import read_data
from fx_trading.utils.parallel_tasks import AsyncMP


def resample_data(df, freq='15min', pip_units=10_000):

    df = df.resample(freq, label='right', closed='right').agg(
        ['min', 'max', 'last', 'first']

    ).dropna(how='all')
    df.rename(columns={'min': 'Low', 'max': 'High', 'last': 'Close', 'first': 'Open'},
              level=1, inplace=True)
    df.columns = [i[1] + i[0] for i in df.columns]

    return df


def read_raw_data(pair: str, period):

    path = 'D:/Blackfire_Projects/Blackfire_fx_algo_trading/fx_data/{}/NT/{}/{}.csv'
    if len(period) == 6:
        file_name = 'DAT_NT_{}_T_{}_'+ period[:4] + '0' + period[-1]
    else:
        file_name = 'DAT_NT_{}_T_{}_'+ period[:4] + period[-2:]
    
    r = []
    try:
        for i in ['BID', 'ASK', 'LAST']:
            _file_name = file_name.format(pair.upper(), i)
            _path = path.format(pair, period, _file_name)
            df = pd.read_csv(_path, delimiter=';', index_col=0, header=None)
            df.index.name = cst.DATE
            df.index = pd.to_datetime(df.index, format='%Y%m%d %H%M%S')
            df.rename(columns={1: i}, inplace=True)
            r.append(df[[i]])
        r = pd.concat(r, axis=1).sort_index().replace({0: np.NaN})
    except FileNotFoundError:
        r = pd.DataFrame(None)
    finally:
        return r


def write_chunk(pair, period, chunk):

    df = read_raw_data(pair=pair, period=period)

    if len(df) != 0:

        df['CHUNK'] = chunk
        df['YEAR'] = df.index.year
        df['FX_PAIR'] = pair
        chunk += 1
        df.to_parquet(
            path=cst.FX_TICK_CHUNK_DATA_PATH,
            engine='pyarrow',
            compression='snappy',
            partition_cols=['FX_PAIR', 'CHUNK', 'YEAR']
        )
        del df
    return [pair, period, chunk]


def get_all_tick_data():
    #169/170 sgd/jpy
    pairs = pd.read_csv('pairs.csv')

    for fx_pair in pairs['currency_pair_code'][25:]:
        print(f'Pair: {fx_pair}')
        params = [[str(i) + 'M' + str(month)] for i in range(2000, 2023) for month in range(1, 13)]
        params = pd.DataFrame(params, columns=['PERIOD'])
        params['CHUNK'] = params.index + 1
        params = [(fx_pair, p['PERIOD'], p['CHUNK']) for key, p in params.iterrows()]
        AsyncMP().exec(params, write_chunk)

        for year in range(2000, 2023, 5):
            y = list(range(year, year + 5)) if year + 5 < 2022 else list(range(year, 2023))
            df = re_sample_tick_data([fx_pair], y, '15min')


def re_sample_tick_data(fx_pair, year, freq):

    try:
        df = read_data(cst.FX_TICK_CHUNK_DATA_PATH, fx_pair, year)
        df.drop(columns=['FX_PAIR', 'CHUNK', 'YEAR'])

        df = resample_data(df, freq=freq, pip_units=10_000)
        df['YEAR'] = df.index.year
        df['FX_PAIR'] = fx_pair[0]

        df.to_parquet(
            path=cst.FX_15MIN_DATA_PATH,
            engine='pyarrow',
            compression='snappy',
            partition_cols=['YEAR', 'FX_PAIR']
        )
        del df
        return [fx_pair, year]
    except ValueError:
        print(f'No values `{fx_pair}, {year}')


def write_all_resample_data(freq='15min'):

    pairs = pd.read_csv('pairs.csv')
    for year in range(2000, 2023, 5):
        y = list(range(year, year + 5)) if year + 5 < 2022 else list(range(year, 2023))
        for fx_pair in pairs['currency_pair_code']:
            print(fx_pair)
            df = re_sample_tick_data([fx_pair], y, freq)
        # df = re_sample_tick_data(['eurusd'], [1980, 2001, 2002, 2003, 2004], freq)
        # params = [(fx_pair, year, freq) for fx_pair in pairs['currency_pair_code'] ]
        # fx_quote = pd.concat(AsyncMP().exec(params, re_sample_tick_data))
        print(y)

    # return fx_quote



if __name__ == '__main__':

    get_all_tick_data()
    # write_all_resample_data()
    # for i in range(2000, 2023):
    #     for month in range(1, 13):
    #         df = read_raw_data(pair='usdhkd', period=str(i) + 'M' + str(month))
    #         print(str(i) + 'M' + str(month))
    #         print(df.head())