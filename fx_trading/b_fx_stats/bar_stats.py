import os
import pandas as pd
import fx_trading.utils.constants as cst
from fx_trading.utils.data_interact import read_data, save_data

CANDLE_SIZE = 'CANDLE_SIZE'

def stats(df, groupby):
    # Nombre de bars.
    return df.groupby(groupby).agg(
        {cst.CLOSE: 'count', 'SPREAD': 'mean', 'SPREAD_MAX': 'mean', CANDLE_SIZE: 'mean'}
    ).rename(
        columns={cst.CLOSE: 'N. Obs', 'SPREAD': 'Moy. Spread', 'SPREAD_MAX': 'Max. Spread', CANDLE_SIZE: 'Taille'}
    )

def prepare_data_for_stats(df, fx_pair):

    df[CANDLE_SIZE] = (df[cst.HIGH] - df[cst.LOW]).abs() * cst.FX_PIP[fx_pair]
    df['HOUR'] = df[cst.DATE].dt.hour
    df.loc[(df['HOUR'] >= 8) & (df['HOUR'] < 16), 'SESSION'] = 'NY'
    df.loc[(df['HOUR'] >= 16) & (df['HOUR'] < 24), 'SESSION'] = 'AS'
    df.loc[(df['HOUR'] >= 0) & (df['HOUR'] < 8), 'SESSION'] = 'LO'

    df['SPREAD'] = (df['OpenASK'] + df['CloseASK'] - df['OpenBID'] - df['CloseBID'] ).abs() * cst.FX_PIP[fx_pair] * 0.5
    df['SPREAD_MAX'] = (df['HighASK'] - df['LowBID'] ).abs() * cst.FX_PIP[fx_pair]

    t = stats(df, groupby=['YEAR']).round(1)
    spread = pd.concat({'Total': t}, axis=1)

    session = stats(df, groupby=['YEAR', 'SESSION']).round(1)
    session = session.unstack(level=-1).swaplevel(axis=1)

    level_0 = ['Total', 'LO', 'NY', 'AS']
    level_1 = ['N. Obs', 'Moy. Spread', 'Taille']

    a = pd.concat([spread, session], axis=1).reindex(level_1, axis=1, level=1).reindex(level_0, axis=1, level=0)

    return a


def build_stats_analysis(fx_pair, date_range, path=cst.FX_15MIN_DATA_PATH):

    fx_quote = read_data(path, [fx_pair], date_range)
    fx_quote[cst.FX_PAIR] = fx_quote[cst.FX_PAIR].astype(str)
    fx_quote['YEAR'] = fx_quote['YEAR'].astype(str)
    fx_quote.reset_index(inplace=True)

    to_save = cst.FX_STATS_DESCRIPTIVE.format(cst.ASSET[fx_pair], fx_pair)
    if not os.path.exists(to_save):
        os.makedirs(to_save)

    fx_quote.set_index(cst.DATE)[[cst.CLOSE]].to_parquet(
        path=to_save + 'plot.parquet'
    )

    df = prepare_data_for_stats(fx_quote, fx_pair).reset_index()

    df.to_parquet(
        path=to_save + 'stats.parquet'
    )


if __name__ == '__main__':

    DATE_RANGE = list(range(2000, 2023))

    for fx_pair in cst.FX_PIP:
        build_stats_analysis(fx_pair, DATE_RANGE, path=cst.FX_15MIN_DATA_PATH)
        print(f'Done for {fx_pair}')