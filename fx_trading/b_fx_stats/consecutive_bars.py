import warnings
import numpy as np
import pandas as pd
import fx_trading.utils.constants as cst
from fx_trading.utils.data_interact import read_data, save_data
from fx_trading.utils.parallel_tasks import AsyncMP

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def add_count_bar_pattern(df):

    df['CandleSIZE'] = df[cst.CLOSE] - df[cst.OPEN]
    df['CandleCOLOR'] = 1
    df.loc[df['CandleSIZE'] <= 0, 'CandleCOLOR'] = 0

    df['Group'] = df['CandleCOLOR'] != df['CandleCOLOR'].shift()
    df['Group'] = df['Group'].cumsum()
    df['Cons_count'] = df.groupby('Group')['CandleCOLOR'].cumcount() + 1

    df['Flag'] = np.NaN
    df.loc[df.drop_duplicates(subset=['Group'], keep='last').index, 'Flag'] = 1

    return df


def prepare_data_for_stats(df, fx_pair, forecast_period):
    """
    Description:
    ------------
    Prepare the data set to compute the stats on accuracy.
    """
    df.reset_index(inplace=True)
    df[['CandleCOLOR', 'Cons_count', 'Flag']] = df[['CandleCOLOR', 'Cons_count', 'Flag']].shift(1)

    df['FORECAST_PERIOD_LOW'] = df[cst.LOW].rolling(forecast_period).apply(
        lambda x: x.idxmin()).shift(-forecast_period + 1)
    df['FORECAST_PERIOD_HIGH'] = df[cst.HIGH].rolling(forecast_period).apply(
        lambda x: x.idxmax()).shift(-forecast_period + 1)

    df['BEAR_STOP'] = np.NaN
    df['BULL_STOP'] = np.NaN

    h = [cst.LOW, cst.HIGH, 'FORECAST_PERIOD_HIGH', 'FORECAST_PERIOD_LOW', 'BEAR_STOP', 'BULL_STOP']

    for i, row in df[h].iterrows():

        if ~np.isnan(row['FORECAST_PERIOD_HIGH']):
            df.loc[i, 'BULL_STOP'] = df[h].iloc[i:int(row['FORECAST_PERIOD_HIGH']) + 1, 0].min()
        if ~np.isnan(row['FORECAST_PERIOD_LOW']):
            df.loc[i, 'BEAR_STOP'] = df[h].iloc[i:int(row['FORECAST_PERIOD_HIGH']) + 1, 1].max()

    df['BEAR_TP'] = df[cst.LOW].rolling(forecast_period).min().shift(-forecast_period + 1)
    df['BEAR_TP'] = (df[cst.OPEN] - df['BEAR_TP']) * cst.FX_PIP[fx_pair]
    df['BEAR_STOP'] = (df['BEAR_STOP'] - df[cst.OPEN]) * cst.FX_PIP[fx_pair]

    df['BULL_TP'] = df[cst.HIGH].rolling(forecast_period).max().shift(-forecast_period + 1)
    df['BULL_TP'] = (df['BULL_TP'] - df[cst.OPEN]) * cst.FX_PIP[fx_pair]
    df['BULL_STOP'] = (df[cst.OPEN] - df['BULL_STOP']) * cst.FX_PIP[fx_pair]

    df.set_index([cst.FX_PAIR, cst.DATE], inplace=True)
    df = df[~df['Flag'].isna()]

    return df.dropna(subset=['BULL_TP'])


def get_optimal_entry_stats(df, pip_target):

    stats = {}
    df['YEAR'] = df[cst.DATE].dt.strftime('%Y').astype(str)
    group_by = [cst.FX_PAIR, 'Candle_type', 'Cons_count', 'YEAR']
    index = [cst.FX_PAIR, 'Candle_type', 'Cons_count', 'YEAR']
    group_by_total = [cst.FX_PAIR, 'Candle_type', 'Cons_count']

    for e in [0, 5, 10, 15]:

        df['TARGET'] = df['TP'] + e
        df['NEW_STOP'] = df['STOP'] - e
        df.loc[df['NEW_STOP'] < 0, 'TARGET'] = np.NaN
        df.loc[df['NEW_STOP'] < 0, 'NEW_STOP'] = np.NaN

        count = df.groupby(group_by)[['TARGET']].count().reset_index()
        tot = df.groupby(group_by_total)[['TARGET']].count().reset_index()
        tot['YEAR'] = 'Total'

        accuracy = {}
        accuracy['N. Obs'] = pd.concat([count, tot]).set_index(index)['TARGET']

        for i in pip_target:
            _df = df[(df['TARGET'] >= i) & (df['NEW_STOP'] < 20)]
            tot = _df.groupby(group_by_total)[['TARGET', 'NEW_STOP']].agg(
                {'TARGET': 'count', 'NEW_STOP': 'mean'}).reset_index()
            tot['YEAR'] = 'Total'
            tot.rename(columns={'TARGET': 'Accuracy Target', 'NEW_STOP': 'Stop Loss'},
                       inplace=True)

            _df = _df.groupby(group_by)[['TARGET', 'NEW_STOP']].agg(
                {'TARGET': 'count', 'NEW_STOP': 'mean'}).reset_index()

            _df.rename(columns={'TARGET': 'Accuracy Target', 'NEW_STOP': 'Stop Loss'},
                       inplace=True)
            accuracy[str(i)] = pd.concat([_df, tot]).set_index(index)['Accuracy Target']

        accuracy = pd.concat(accuracy, axis=1)

        accuracy = accuracy[accuracy['N. Obs'] != 0]
        accuracy[list(map(str, pip_target))] = (100 * accuracy[list(map(str, pip_target))].div(
            accuracy['N. Obs'], axis=0)).round(1)
        stats[f'Open - {e}'] = accuracy

    return pd.concat(stats, axis=1)


def run_potential_entry(df, pip_target):

    header = ['TP', 'STOP', 'Candle_type', 'Cons_count']
    stats = {}

    df['Candle_type'] = np.NaN
    df.loc[df['CandleCOLOR'] == 1, 'Candle_type'] = cst.BULLISH
    df.loc[df['CandleCOLOR'] == 1, 'TP'] = df['BEAR_TP']
    df.loc[df['CandleCOLOR'] == 1, 'STOP'] = df['BEAR_STOP']

    df.loc[df['CandleCOLOR'] == 0, 'Candle_type'] = cst.BEARISH
    df.loc[df['CandleCOLOR'] == 0, 'TP'] = df['BULL_TP']
    df.loc[df['CandleCOLOR'] == 0, 'STOP'] = df['BULL_STOP']

    st = get_optimal_entry_stats(df[header].reset_index(), pip_target)

    return st
    # stats[('Candlestick_Entry', candle)] = st


    for candle in CANDLE_NAMES:

        df['Candle_type'] = np.NaN
        df.loc[df[candle].fillna(-1) > 0, 'Candle_type'] = cst.BULLISH
        df.loc[df[candle].fillna(-1) > 0, 'TP'] = df['BULL_TP']
        df.loc[df[candle].fillna(-1) > 0, 'STOP'] = df['BULL_STOP']

        df.loc[df[candle].fillna(1) < 0, 'Candle_type'] = cst.BEARISH
        df.loc[df[candle].fillna(1) < 0, 'TP'] = df['BEAR_TP']
        df.loc[df[candle].fillna(1) < 0, 'STOP'] = df['BEAR_STOP']

        _df = df.copy()
        _df.dropna(subset=['Candle_type'], inplace=True)
        _df.set_index('Candle_type', inplace=True, append=True)

        if len(_df) != 0:
            st = get_optimal_entry_stats(_df[header].reset_index(), pip_target)
            stats[('Candlestick_Entry', candle)] = st

    stats = pd.concat(stats)
    stats.index.names = ['Stats', 'Candle_name', cst.FX_PAIR, 'Candle_type', cst.DATE]

    return stats.reset_index()


def build_bar_count_analysis(fx_pairs, date_range, path=cst.FX_15MIN_DATA_PATH, forecasting_period=16):
    """
    Description:
    ------------
    Compute some simple stats on the predictive power of the consecutive bar
    that closes in the same direction.
    """

    fx_quote = read_data(path, fx_pairs, date_range)
    fx_quote[cst.FX_PAIR] = fx_quote[cst.FX_PAIR].astype(str)
    fx_quote = fx_quote.set_index(cst.FX_PAIR, append=True).swaplevel(axis=0)
    fx_quote.sort_index(inplace=True)

    #############################################################################
    # Add consecutive bars
    #############################################################################
    params = [(df,) for (name, df) in fx_quote.groupby(level=0)]
    fx_quote = pd.concat(AsyncMP().exec(params, add_count_bar_pattern))
    del params

    #############################################################################
    # Prepare data set for the stats
    #############################################################################
    params = [(df, name, forecasting_period) for (name, df) in fx_quote.groupby(level=0)]
    fx_quote = pd.concat(AsyncMP().exec(params, prepare_data_for_stats))
    del params

    #############################################################################
    # Stats for each FX pair
    #############################################################################
    # per_fx_stats = run_max_gain_stats(fx_quote, pip_target=[5, 10, 15, 20])
    per_fx_stats = run_potential_entry(fx_quote, pip_target=[5, 10, 15, 20])
    # save_data(per_fx_stats, path=cst.FX_STATS_CANDLESTICK, partition_cols=['Stats'])
    # all_stats = run_max_gain_stats(fx_quote, pip_target=[5, 10, 15, 20])
    return per_fx_stats

    return fx_quote

if __name__ == '__main__':

    FX_PAIRS = ['eurusd']
    DATE_RANGE = list(range(2022, 2023))
    count_stats = build_bar_count_analysis(FX_PAIRS, DATE_RANGE, path=cst.FX_15MIN_DATA_PATH)